"""Quick eyeball plot for the simulator output (Phase 1 sanity check).

Reads the CSV produced by run_simulation.py and saves a 6-panel figure:
  1) motor current
  2) bearing temperature
  3) vibration
  4) ambient temperature
  5) wear (ground truth)
  6) label kind as a colour band
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/sim_72h.csv"))
    p.add_argument("--output", type=Path, default=Path("data/sim_72h_timeline.png"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).parent
    in_path = args.input if args.input.is_absolute() else root / args.input
    out_path = args.output if args.output.is_absolute() else root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    hours = df["timestamp_s"].to_numpy() / 3600.0

    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(hours, df["motor_current_a"], lw=0.6, color="#1f77b4")
    axes[0].set_ylabel("Current (A)")
    axes[1].plot(hours, df["bearing_temp_c"], lw=0.6, color="#d62728")
    axes[1].set_ylabel("Bearing Temp (°C)")
    axes[2].plot(hours, df["vibration_mm_s"], lw=0.4, color="#2ca02c")
    axes[2].set_ylabel("Vibration (mm/s)")
    axes[3].plot(hours, df["ambient_temp_c"], lw=0.6, color="#ff7f0e")
    axes[3].set_ylabel("Ambient (°C)")
    axes[4].plot(hours, df["wear_truth"], lw=0.8, color="#9467bd")
    axes[4].set_ylabel("Wear (truth)")
    axes[4].axhline(0.5, color="grey", ls="--", lw=0.5, label="label threshold")
    axes[4].axhline(1.0, color="black", ls=":", lw=0.5, label="nonlinear threshold")
    axes[4].legend(loc="upper left", fontsize=8)

    color_map = {"normal": "#cccccc", "wear": "#ffcc66", "spike": "#cc3333"}
    label_colors = df["label_kind"].map(color_map).fillna("#cccccc").to_numpy()
    axes[5].scatter(hours, np.ones_like(hours), c=label_colors, s=2, marker="|")
    axes[5].set_yticks([])
    axes[5].set_ylabel("Label")
    axes[5].set_xlabel("Hours")

    counts = df["label_kind"].value_counts().to_dict()
    total = len(df)
    title = (
        f"72h simulator output  "
        f"normal={counts.get('normal', 0):,} ({counts.get('normal', 0) / total * 100:.1f}%)  "
        f"wear={counts.get('wear', 0):,} ({counts.get('wear', 0) / total * 100:.1f}%)  "
        f"spike={counts.get('spike', 0):,} ({counts.get('spike', 0) / total * 100:.1f}%)"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    print(f"[plot_timeline] saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
