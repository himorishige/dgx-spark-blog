"""End-to-end: simulator CSV -> Chronos-2 multivariate -> anomaly score -> JSON / NPZ.

Run on DGX Spark inside the SKAB chronos2 venv:

    cd ~/works/private/workspace-dgx/workspace/blog/scripts/chronos2-plc-sim
    ~/works/timeseries-fm-bench/chronos2/.venv/bin/python predict_pipeline.py \
        --input data/sim_72h.csv \
        --model chronos2-28m
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_anomaly_score import (
    aggregate,
    align_labels_to_windows,
    auc_roc,
    fit_zscore,
    residual_score,
    sliding_windows,
    threshold_metrics,
)
from lib_chronos2_pipeline import MODEL_REGISTRY, Chronos2Predictor


SENSORS = ("motor_current_a", "bearing_temp_c", "vibration_mm_s", "ambient_temp_c")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chronos-2 multivariate prediction over simulator CSV.")
    p.add_argument("--input", type=Path, default=Path("data/sim_72h.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/predictions"))
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="chronos2-28m",
    )
    p.add_argument("--context", type=int, default=256,
                   help="Context length (rows). Default 256 (~4.3 min at 1 Hz).")
    p.add_argument("--horizon", type=int, default=16,
                   help="Forecast horizon (rows). Default 16 (~16 s at 1 Hz).")
    p.add_argument("--stride", type=int, default=16,
                   help="Stride between windows. Default 16 (non-overlapping).")
    p.add_argument(
        "--anomaly-free-hours",
        type=float,
        default=6.0,
        help="Initial hours used as the anomaly-free section for the z-score scaler.",
    )
    p.add_argument(
        "--aggregations",
        nargs="+",
        default=["mean", "max", "pca"],
        help="Aggregation strategies to evaluate.",
    )
    p.add_argument(
        "--positive-kinds",
        nargs="+",
        default=["spike", "wear"],
        choices=["spike", "wear"],
        help="Which label_kind values count as the positive class. "
             "Default is both. Use 'spike' alone to test pure spike detection.",
    )
    p.add_argument("--max-windows", type=int, default=0,
                   help="If >0, truncate to this many windows (smoke test).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).parent
    in_path = args.input if args.input.is_absolute() else root / args.input
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {in_path}")
    df = pd.read_csv(in_path)
    before = len(df)
    df = df.dropna(subset=list(SENSORS)).reset_index(drop=True)
    print(f"[load] {before:,} rows, dropped {before - len(df):,} dropout rows -> {len(df):,}")
    arr = df[list(SENSORS)].to_numpy(dtype=np.float32)
    row_labels = df["label_kind"].isin(args.positive_kinds).to_numpy(dtype=np.int8)
    print(f"[labels] positive kinds = {args.positive_kinds}; "
          f"row-positive = {int(row_labels.sum()):,}/{len(row_labels):,} "
          f"({row_labels.mean() * 100:.1f}%)")

    if len(df) > 1:
        dt_s = float(df["timestamp_s"].iloc[1] - df["timestamp_s"].iloc[0])
    else:
        dt_s = 1.0
    af_steps = max(args.context, int(args.anomaly_free_hours * 3600 / dt_s))
    af_steps = min(af_steps, len(arr))
    mean, std = fit_zscore(arr[:af_steps])
    print(f"[scaler] anomaly-free window = first {af_steps:,} rows ({af_steps * dt_s / 3600:.2f} h)")
    print(f"[scaler] mean = {mean.tolist()}")
    print(f"[scaler] std  = {std.tolist()}")

    print(f"[windows] ctx={args.context}, h={args.horizon}, stride={args.stride}")
    X, Y, end_idx = sliding_windows(arr, args.context, args.horizon, args.stride)
    win_labels = align_labels_to_windows(row_labels, end_idx, args.horizon)
    if args.max_windows > 0 and args.max_windows < len(X):
        X = X[: args.max_windows]
        Y = Y[: args.max_windows]
        end_idx = end_idx[: args.max_windows]
        win_labels = win_labels[: args.max_windows]
    print(f"[windows] N={len(X):,}, positive={int(win_labels.sum()):,} "
          f"({win_labels.mean() * 100:.1f}%)")

    print(f"[model] loading {args.model} ({MODEL_REGISTRY[args.model]})")
    predictor = Chronos2Predictor(model_name=args.model)
    t0 = time.monotonic()
    predictor.load()
    print(f"[model] loaded in {time.monotonic() - t0:.1f}s")

    print("[warmup] 3 dummy predictions")
    predictor.warmup(n_var=arr.shape[1], context_len=args.context, horizon=args.horizon)

    print(f"[predict] running {len(X):,} windows")
    t0 = time.monotonic()
    preds, latencies = predictor.predict_multivariate(X, args.horizon)
    wall = time.monotonic() - t0
    lat = np.asarray(latencies)
    print(f"[predict] wall={wall:.1f}s; "
          f"per-window median={np.median(lat) * 1000:.2f}ms "
          f"p95={np.quantile(lat, 0.95) * 1000:.2f}ms "
          f"min={lat.min() * 1000:.2f}ms")

    print("[score] residual (z-score space, MAE)")
    per_sensor = residual_score(preds, Y, std=std, metric="mae")

    summary: dict[str, dict] = {}
    for strat in args.aggregations:
        scores = aggregate(per_sensor, strategy=strat)
        auc = auc_roc(scores, win_labels)
        metrics = threshold_metrics(scores, win_labels)
        summary[strat] = {"auc": auc, **metrics}
        print(f"[metrics:{strat:>4}] AUC={auc:.4f}  F1={metrics['f1']:.4f}  "
              f"FAR={metrics['far']:.4f}  MAR={metrics['mar']:.4f}  "
              f"thresh={metrics['threshold']:.4f} "
              f"(TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']})")

    npz_path = out_dir / f"{args.model}_predictions.npz"
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        preds=preds.astype(np.float32),
        end_indices=end_idx,
        labels=win_labels,
        per_sensor_residual=per_sensor,
        scaler_mean=mean,
        scaler_std=std,
    )
    print(f"[save] {npz_path}")

    summary_path = out_dir / f"{args.model}_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "hf_repo": MODEL_REGISTRY[args.model],
                "input": str(in_path),
                "context": args.context,
                "horizon": args.horizon,
                "stride": args.stride,
                "anomaly_free_steps": af_steps,
                "positive_kinds": args.positive_kinds,
                "n_windows": int(len(X)),
                "n_positive": int(win_labels.sum()),
                "wall_s": wall,
                "latency_p50_ms": float(np.median(lat) * 1000),
                "latency_p95_ms": float(np.quantile(lat, 0.95) * 1000),
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
