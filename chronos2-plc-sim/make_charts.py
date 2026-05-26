"""Generate the figures embedded in the blog article.

Reads from data/sim_72h.csv + data/predictions/*.npz + data/llm_comments.jsonl
and writes PNGs to ../../images/chronos2-plc-sim/.

Run on Mac (matplotlib only, no GPU needed):
    cd workspace/blog/scripts/chronos2-plc-sim
    uv run python make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PRED_DIR = DATA_DIR / "predictions"
IMG_DIR = (ROOT / "../../images/chronos2-plc-sim").resolve()
IMG_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = ("motor_current_a", "bearing_temp_c", "vibration_mm_s", "ambient_temp_c")
SENSOR_LABELS = {
    "motor_current_a": "Motor current (A)",
    "bearing_temp_c": "Bearing temperature (°C)",
    "vibration_mm_s": "Vibration (mm/s)",
    "ambient_temp_c": "Ambient temperature (°C)",
}
SENSOR_COLORS = {
    "motor_current_a": "#1f77b4",
    "bearing_temp_c": "#d62728",
    "vibration_mm_s": "#2ca02c",
    "ambient_temp_c": "#ff7f0e",
}


def chart_timeline(df: pd.DataFrame) -> Path:
    hours = df["timestamp_s"].to_numpy() / 3600.0
    fig, axes = plt.subplots(6, 1, figsize=(11, 11), sharex=True)

    for ax, sensor in zip(axes[:4], SENSORS):
        ax.plot(hours, df[sensor], lw=0.5, color=SENSOR_COLORS[sensor])
        ax.set_ylabel(SENSOR_LABELS[sensor])

    axes[4].plot(hours, df["wear_truth"], lw=0.8, color="#9467bd", label="wear truth")
    axes[4].axhline(0.5, color="grey", ls="--", lw=0.5, label="label threshold")
    axes[4].axhline(1.0, color="black", ls=":", lw=0.5, label="nonlinear threshold")
    axes[4].set_ylabel("Wear (truth)")
    axes[4].legend(loc="upper left", fontsize=8)

    color_map = {"normal": "#cccccc", "wear": "#ffcc66", "spike": "#cc3333"}
    colors = df["label_kind"].map(color_map).fillna("#cccccc").to_numpy()
    axes[5].scatter(hours, np.ones_like(hours), c=colors, s=2, marker="|")
    axes[5].set_yticks([])
    axes[5].set_ylabel("Label")
    axes[5].set_xlabel("Hours")

    counts = df["label_kind"].value_counts().to_dict()
    total = len(df)
    title = (
        f"72h PLC-like simulator output  —  "
        f"normal {counts.get('normal', 0):,} ({counts.get('normal', 0)/total*100:.1f}%) / "
        f"wear {counts.get('wear', 0):,} ({counts.get('wear', 0)/total*100:.1f}%) / "
        f"spike {counts.get('spike', 0):,} ({counts.get('spike', 0)/total*100:.1f}%)"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = IMG_DIR / "timeline-72h.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def chart_pred_vs_truth(npz: dict, window_idx: int = 5000) -> Path:
    X = npz["X"]
    Y = npz["Y"]
    preds = npz["preds"]
    ctx_len = X.shape[1]
    horizon = Y.shape[1]
    total_len = ctx_len + horizon

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    axes = axes.flatten()
    t_ctx = np.arange(ctx_len)
    t_fc = np.arange(ctx_len, total_len)

    for j, sensor in enumerate(SENSORS):
        ax = axes[j]
        color = SENSOR_COLORS[sensor]
        ax.plot(t_ctx, X[window_idx, :, j], color=color, lw=0.8, label="context")
        ax.plot(t_fc, Y[window_idx, :, j], color=color, lw=1.5, label="truth")
        ax.plot(
            t_fc,
            preds[window_idx, :, j],
            color="black",
            lw=1.5,
            ls="--",
            label="Chronos-2 median",
        )
        ax.axvline(ctx_len - 0.5, color="grey", ls=":", lw=0.5)
        ax.set_title(SENSOR_LABELS[sensor], fontsize=10)
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)
    axes[2].set_xlabel("Steps (1 Hz)")
    axes[3].set_xlabel("Steps (1 Hz)")
    fig.suptitle(
        f"Chronos-2 28M multivariate prediction (window {window_idx}, ctx={ctx_len}, h={horizon})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = IMG_DIR / "chronos2-pred-vs-truth.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def chart_anomaly_score(npz: dict, df: pd.DataFrame) -> Path:
    per_sensor = npz["per_sensor_residual"]
    labels = npz["labels"]
    end_indices = npz["end_indices"]
    scores = per_sensor.mean(axis=1)

    # Map end_indices (row idx in CSV) to hours
    valid_rows = df.dropna(subset=list(SENSORS)).reset_index(drop=True)
    timestamps = valid_rows["timestamp_s"].to_numpy()
    win_hours = timestamps[end_indices - 1] / 3600.0

    # Threshold via the same 50-quantile sweep predict_pipeline uses,
    # but plot a representative threshold = F1-best for spike-only labels.
    label_kind = valid_rows["label_kind"].to_numpy()
    spike_row = (label_kind == "spike").astype(np.int8)
    spike_labels = np.zeros(len(end_indices), dtype=np.int8)
    h = npz["Y"].shape[1]
    for i, e in enumerate(end_indices):
        e = int(e)
        spike_labels[i] = int(spike_row[e - h : e].sum() > 0)

    qs = np.linspace(0.50, 0.99, 50)
    candidates = np.quantile(scores, qs)
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        pred = scores >= t
        tp = int(np.sum(pred & (spike_labels == 1)))
        fp = int(np.sum(pred & (spike_labels == 0)))
        fn = int(np.sum(~pred & (spike_labels == 1)))
        if tp + fp == 0 or tp + fn == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1]})

    ax[0].plot(win_hours, scores, color="#1f77b4", lw=0.5, alpha=0.7, label="anomaly score (mean residual z)")
    ax[0].scatter(
        win_hours[spike_labels == 1],
        scores[spike_labels == 1],
        color="#cc3333",
        s=10,
        label=f"spike-window truth ({int(spike_labels.sum())})",
        zorder=3,
    )
    ax[0].axhline(best_t, color="grey", ls="--", lw=1.0, label=f"F1-best threshold = {best_t:.3f}")
    ax[0].set_ylabel("Anomaly score")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title(
        f"Chronos-2 28M residual anomaly score (spike-only labels, F1={best_f1:.3f})",
        fontsize=11,
    )

    # Lower panel: per-sensor residual breakdown for the 4 sensors
    for j, sensor in enumerate(SENSORS):
        ax[1].plot(win_hours, per_sensor[:, j], lw=0.4, alpha=0.8,
                   color=SENSOR_COLORS[sensor], label=SENSOR_LABELS[sensor])
    ax[1].set_yscale("symlog", linthresh=1.0)
    ax[1].set_ylabel("Per-sensor residual (z)")
    ax[1].set_xlabel("Hours")
    ax[1].legend(loc="upper left", fontsize=8, ncol=2)

    fig.tight_layout()
    out = IMG_DIR / "anomaly-score-threshold.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> int:
    print(f"[load] CSV: {DATA_DIR / 'sim_72h.csv'}")
    df = pd.read_csv(DATA_DIR / "sim_72h.csv")
    print(f"[load] npz: {PRED_DIR / 'chronos2-28m_predictions.npz'}")
    npz_data = np.load(PRED_DIR / "chronos2-28m_predictions.npz")

    out1 = chart_timeline(df)
    print(f"[chart] timeline -> {out1}")

    # Pick a window that captures one of the top-spike events. Use index ~228
    # (highest score positive in our earlier comment_pipeline sample).
    out2 = chart_pred_vs_truth({"X": npz_data["X"], "Y": npz_data["Y"], "preds": npz_data["preds"]}, window_idx=228)
    print(f"[chart] pred vs truth (window 228) -> {out2}")

    out3 = chart_anomaly_score(
        {
            "per_sensor_residual": npz_data["per_sensor_residual"],
            "labels": npz_data["labels"],
            "end_indices": npz_data["end_indices"],
            "Y": npz_data["Y"],
        },
        df,
    )
    print(f"[chart] anomaly score -> {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
