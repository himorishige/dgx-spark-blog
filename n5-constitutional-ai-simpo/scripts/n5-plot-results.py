#!/usr/bin/env python3
"""N5 Results Visualization

Generate radar chart comparing CLP compliance scores
before and after Constitutional AI + SimPO training.

Usage:
  python n5-plot-results.py \
    --baseline ./data/n5/clp_baseline.json \
    --aligned ./data/n5/clp_aligned.json \
    --output ./data/n5/clp_radar.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# CLP principle labels (Japanese)
CLP_LABELS = {
    "leadership": "リーダーシップ",
    "partnership": "パートナーシップ",
    "diversity": "ダイバーシティ",
    "professional": "プロフェッショナル",
    "gratitude": "感謝",
    "customer_focus": "顧客視点",
    "feedback": "フィードバック",
    "information_sharing": "情報発信",
    "try_it": "やってみる",
    "enjoy": "楽しむ",
}

# Ordered list of CLP IDs for consistent radar chart
CLP_ORDER = [
    "leadership", "partnership", "diversity", "professional", "gratitude",
    "customer_focus", "feedback", "information_sharing", "try_it", "enjoy",
]


def plot_radar(baseline_data: dict, aligned_data: dict, output_path: str):
    """Generate radar chart comparing baseline and aligned CLP scores."""
    b_avg = baseline_data["clp_averages"]
    a_avg = aligned_data["clp_averages"]

    # Extract scores in order
    categories = [CLP_LABELS.get(cid, cid) for cid in CLP_ORDER]
    baseline_scores = [b_avg.get(cid, 0) for cid in CLP_ORDER]
    aligned_scores = [a_avg.get(cid, 0) for cid in CLP_ORDER]

    # Number of variables
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the radar chart
    baseline_scores += baseline_scores[:1]
    aligned_scores += aligned_scores[:1]
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Try to use Japanese font
    try:
        font_prop = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    except Exception:
        font_prop = None

    # Plot baseline
    ax.plot(angles, baseline_scores, "o-", linewidth=2, label=f"Baseline ({baseline_data['model']})", color="#1976d2")
    ax.fill(angles, baseline_scores, alpha=0.15, color="#1976d2")

    # Plot aligned
    ax.plot(angles, aligned_scores, "o-", linewidth=2, label=f"Aligned ({aligned_data['model']})", color="#d32f2f")
    ax.fill(angles, aligned_scores, alpha=0.15, color="#d32f2f")

    # Set category labels
    ax.set_xticks(angles[:-1])
    if font_prop:
        ax.set_xticklabels(categories, fontproperties=font_prop, fontsize=11)
    else:
        ax.set_xticklabels(categories, fontsize=11)

    # Set score range
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9)
    ax.set_rlabel_position(30)

    # Grid style
    ax.grid(True, linestyle="--", alpha=0.3)

    # Title and legend
    title = "CLP Compliance Score (1-5)"
    if font_prop:
        ax.set_title(title, fontproperties=font_prop, fontsize=14, pad=20)
    else:
        ax.set_title(title, fontsize=14, pad=20)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Overall scores annotation
    b_overall = baseline_data["overall_average"]
    a_overall = aligned_data["overall_average"]
    delta = a_overall - b_overall
    sign = "+" if delta >= 0 else ""

    annotation = f"Overall: {b_overall:.2f} -> {a_overall:.2f} ({sign}{delta:.2f})"
    fig.text(0.5, 0.02, annotation, ha="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="N5 Results Visualization")
    parser.add_argument("--baseline", required=True, help="Baseline CLP results JSON")
    parser.add_argument("--aligned", required=True, help="Aligned CLP results JSON")
    parser.add_argument("--output", default="./data/n5/clp_radar.png", help="Output PNG path")

    args = parser.parse_args()

    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.aligned, encoding="utf-8") as f:
        aligned = json.load(f)

    print("=== CLP Radar Chart ===")
    plot_radar(baseline, aligned, args.output)


if __name__ == "__main__":
    main()
