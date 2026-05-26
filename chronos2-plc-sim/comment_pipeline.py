"""Pick representative anomaly windows and have the local LLM produce a
Japanese maintenance comment for each.

Workflow:
  1. Load CSV + Chronos-2 predictions (.npz)
  2. Aggregate per-sensor residual into a single score per window
  3. Sample windows for evaluation:
       - top-K positive (largest score among true anomalies)
       - top-K negative (largest score among true normals = false alarms)
  4. For each sample, build a JSON observation context + system prompt and
     call the local vLLM endpoint
  5. Optionally repeat each sample N times to inspect output variance / hallucination
  6. Save a JSONL log of (window, label, score, response, observation)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_llm_client import (
    PROMPT_HEADER_JA,
    SYSTEM_PROMPT_JA,
    VLLMClient,
    build_user_prompt,
)


SENSORS = ("motor_current_a", "bearing_temp_c", "vibration_mm_s", "ambient_temp_c")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/sim_72h.csv"))
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/predictions/chronos2-28m_predictions.npz"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/llm_comments.jsonl"),
    )
    p.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    p.add_argument("--model", default="nemotron-3-nano-nvfp4-local")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="LLM calls per sample (for variance/hallucination inspection).",
    )
    p.add_argument(
        "--top-k-positive",
        type=int,
        default=3,
        help="Number of true-positive windows to sample (highest score).",
    )
    p.add_argument(
        "--top-k-negative",
        type=int,
        default=3,
        help="Number of false-alarm windows to sample (high-score true-normal).",
    )
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Nemotron's reasoning mode (slower, longer output).",
    )
    p.add_argument("--max-tokens", type=int, default=200,
                   help="Max output tokens per LLM call.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).parent
    in_path = args.input if args.input.is_absolute() else root / args.input
    preds_path = args.predictions if args.predictions.is_absolute() else root / args.predictions
    out_path = args.output if args.output.is_absolute() else root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] CSV: {in_path}")
    df = pd.read_csv(in_path).dropna(subset=list(SENSORS)).reset_index(drop=True)

    print(f"[load] predictions: {preds_path}")
    data = np.load(preds_path)
    X = data["X"]
    per_sensor = data["per_sensor_residual"]
    labels = data["labels"]
    scaler_mean = data["scaler_mean"]
    scaler_std = data["scaler_std"]

    scores = per_sensor.mean(axis=1)
    print(f"[stats] N={len(labels):,}  positive={int(labels.sum()):,}  "
          f"score range [{scores.min():.3f}, {scores.max():.3f}]")

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    top_pos = pos_idx[np.argsort(-scores[pos_idx])][: args.top_k_positive]
    top_neg = neg_idx[np.argsort(-scores[neg_idx])][: args.top_k_negative]
    sample_idx = list(top_pos) + list(top_neg)
    print(f"[sample] {len(top_pos)} true-positive + {len(top_neg)} false-alarm = "
          f"{len(sample_idx)} windows")

    client = VLLMClient(base_url=args.vllm_url, served_model_name=args.model)
    if not client.health():
        print(f"[error] vLLM at {args.vllm_url} not reachable")
        return 1

    results: list[dict] = []
    for s_idx in sample_idx:
        s_idx = int(s_idx)
        recent_values = {sensor: round(float(X[s_idx, -1, j]), 4)
                         for j, sensor in enumerate(SENSORS)}
        per_sensor_z = {sensor: round(float(per_sensor[s_idx, j]), 4)
                        for j, sensor in enumerate(SENSORS)}
        baseline = {
            sensor: {
                "mean": round(float(scaler_mean[j]), 4),
                "std": round(float(scaler_std[j]), 4),
            }
            for j, sensor in enumerate(SENSORS)
        }
        observation = {
            "window_index": s_idx,
            "anomaly_score_mean_zscore": round(float(scores[s_idx]), 4),
            "per_sensor_residual_zscore": per_sensor_z,
            "recent_sensor_values": recent_values,
            "scaler_baseline_anomaly_free_6h": baseline,
            "ground_truth_label": int(labels[s_idx]),
        }
        user_msg = build_user_prompt(observation)

        print(f"\n[{s_idx:6d}] label={int(labels[s_idx])} score={float(scores[s_idx]):.3f}")
        for rep in range(args.repeats):
            t0 = time.monotonic()
            try:
                resp = client.chat(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_JA},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
                wall = time.monotonic() - t0
                content = resp["choices"][0]["message"]["content"].strip()
                usage = resp.get("usage", {})
            except Exception as e:
                wall = time.monotonic() - t0
                content = f"<<error: {e}>>"
                usage = {}

            row = {
                "window_index": s_idx,
                "label": int(labels[s_idx]),
                "score": float(scores[s_idx]),
                "repeat": rep,
                "temperature": args.temperature,
                "wall_s": wall,
                "response": content,
                "usage": usage,
                "observation": observation,
            }
            results.append(row)
            print(f"   rep={rep} t={args.temperature} wall={wall:.1f}s "
                  f"tokens={usage.get('completion_tokens', '?')}")
            for line in content.splitlines():
                print(f"     {line}")

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[save] {out_path}  ({len(results)} responses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
