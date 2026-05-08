"""Generate charts for the Gemma 4 MTP article.

Outputs to workspace/blog/images/gemma4-mtp/:
  - mtp-speedup-bar.png        : 4 models, baseline vs MTP, long-form
  - mtp-jcq-vs-long.png        : JCQ short (max_tokens=8) vs long-form (256)
  - mtp-lang-comparison.png    : ja vs en for E2B/E4B (acceptance rate + tps)
  - mtp-bandwidth-analysis.png : DGX Spark vs A100 vs H100 memory bandwidth
  - mtp-feasibility-matrix.png : DGX Spark Gemma 4 MTP feasibility table

Usage:
    python make-charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Use Noto Sans CJK JP for Japanese labels
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(font_path)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "g4-mtp"
IMG_DIR = Path(__file__).resolve().parent.parent.parent / "images" / "gemma4-mtp"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["e2b", "e4b", "26b-a4b", "31b"]
MODEL_LABELS = ["E2B (BF16)", "E4B (BF16)", "26B-A4B (NVFP4)", "31B (NVFP4)"]
COLOR_BASELINE = "#5b8def"
COLOR_MTP = "#2ecc71"
COLOR_BAD = "#e74c3c"


def load_long(label: str) -> dict:
    return json.loads((DATA_DIR / f"{label}.long.summary.json").read_text())


def load_jcq(label: str) -> dict:
    return json.loads((DATA_DIR / f"{label}.summary.json").read_text())


def chart_speedup_bar() -> None:
    """Long-form (max_tokens=256) baseline vs MTP per model."""
    baseline = []
    mtp = []
    speedup = []
    for size in MODELS:
        b = load_long(f"{size}-longform-baseline")["warm"]["mean_tps"]
        m = load_long(f"{size}-longform-mtp2")["warm"]["mean_tps"]
        baseline.append(b)
        mtp.append(m)
        speedup.append(m / b)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_b = ax.bar(x - width / 2, baseline, width, color=COLOR_BASELINE, label="baseline (target only)")
    bars_m = ax.bar(x + width / 2, mtp, width, color=COLOR_MTP, label="MTP num_spec=2")

    for bar, val in zip(bars_b, baseline):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}", ha="center", fontsize=9)
    for bar, val, sp in zip(bars_m, mtp, speedup):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}\n({sp:.2f}x)", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("warm tok/s")
    ax.set_title("Gemma 4 MTP の長文 speedup (max_tokens=256, DGX Spark)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(mtp) * 1.25)

    fig.tight_layout()
    out = IMG_DIR / "mtp-speedup-bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def chart_jcq_vs_long() -> None:
    """JCQ short (max_tokens=8) vs long-form (256) speedup comparison."""
    jcq_speedups = []
    long_speedups = []
    for size in MODELS:
        # JCQ: load both and compute speedup
        jb = load_jcq(f"{size}-baseline")["mean_tps"]
        jm = load_jcq(f"{size}-mtp2")["mean_tps"]
        jcq_speedups.append(jm / jb)
        lb = load_long(f"{size}-longform-baseline")["warm"]["mean_tps"]
        lm = load_long(f"{size}-longform-mtp2")["warm"]["mean_tps"]
        long_speedups.append(lm / lb)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_j = ax.bar(x - width / 2, jcq_speedups, width, color="#888", label="JCQ short (max_tokens=8)")
    bars_l = ax.bar(x + width / 2, long_speedups, width, color=COLOR_MTP, label="long-form (max_tokens=256)")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(3.4, 1.02, "speedup = 1.0", fontsize=8, color="gray")

    for bar, sp in zip(bars_j, jcq_speedups):
        color = COLOR_BAD if sp < 1.0 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2, sp + 0.03, f"{sp:.2f}x", ha="center", fontsize=9, color=color)
    for bar, sp in zip(bars_l, long_speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, sp + 0.03, f"{sp:.2f}x", ha="center", fontsize=9, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_ylabel("speedup（baseline = 1.0）")
    ax.set_title("MTP は decode 量で効きが変わる（短文 vs 長文）")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 2.5)

    fig.tight_layout()
    out = IMG_DIR / "mtp-jcq-vs-long.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def chart_lang_comparison() -> None:
    """Japanese vs English for E2B/E4B: acceptance rate + tok/s."""
    sizes = ["e2b", "e4b"]
    sizes_label = ["E2B (BF16)", "E4B (BF16)"]
    ja_tps, en_tps, ja_acc, en_acc = [], [], [], []
    for size in sizes:
        ja = load_long(f"{size}-longform-mtp2")
        en = load_long(f"{size}-longform-mtp2-en")
        ja_tps.append(ja["warm"]["mean_tps"])
        en_tps.append(en["warm"]["mean_tps"])
        ja_acc.append(ja["metrics_delta"]["acceptance_rate"])
        en_acc.append(en["metrics_delta"]["acceptance_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(sizes))
    width = 0.36

    # Left: tok/s
    bars_ja = ax1.bar(x - width / 2, ja_tps, width, color="#e91e63", label="日本語")
    bars_en = ax1.bar(x + width / 2, en_tps, width, color="#2196f3", label="English")
    for bar, val in zip(bars_ja, ja_tps):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}", ha="center", fontsize=9)
    for bar, val in zip(bars_en, en_tps):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.0, f"{val:.1f}", ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes_label)
    ax1.set_ylabel("warm tok/s")
    ax1.set_title("MTP tok/s（言語別）")
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(ja_tps + en_tps) * 1.25)

    # Right: acceptance rate
    bars_ja2 = ax2.bar(x - width / 2, [a * 100 for a in ja_acc], width, color="#e91e63", label="日本語")
    bars_en2 = ax2.bar(x + width / 2, [a * 100 for a in en_acc], width, color="#2196f3", label="English")
    for bar, val in zip(bars_ja2, ja_acc):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.6, f"{val * 100:.1f}%", ha="center", fontsize=9)
    for bar, val in zip(bars_en2, en_acc):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.6, f"{val * 100:.1f}%", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes_label)
    ax2.set_ylabel("採択率 [%]")
    ax2.set_title("draft トークン採択率（言語別）")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 60)

    fig.suptitle("日本語 vs 英語: 言語の影響は限定的（±3pt 以内）", fontsize=12)
    fig.tight_layout()
    out = IMG_DIR / "mtp-lang-comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def chart_bandwidth_analysis() -> None:
    """Memory bandwidth comparison: DGX Spark vs A100 vs H100."""
    hw = ["DGX Spark\n(GB10 LPDDR5X)", "A100 SXM4\n(HBM2e)", "H100 SXM5\n(HBM3)"]
    bw = [273, 2039, 3350]  # GB/s
    colors = ["#5b8def", "#888", "#444"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(hw, bw, color=colors)
    for bar, v in zip(bars, bw):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v:,} GB/s",
                ha="center", fontsize=11, fontweight="bold")

    # Annotate ratios
    ratio_a100 = bw[1] / bw[0]
    ratio_h100 = bw[2] / bw[0]
    ax.text(1, 1700, f"× {ratio_a100:.1f}", ha="center", fontsize=10, color="white", fontweight="bold")
    ax.text(2, 2900, f"× {ratio_h100:.1f}", ha="center", fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("memory bandwidth [GB/s]")
    ax.set_title("メモリ帯域比較: DGX Spark vs データセンター GPU")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 3700)

    # Add annotation about MTP speedup ceiling
    ax.text(0.5, -0.18,
            "speculative decoding は memory bandwidth を活用する最適化。\n"
            "DGX Spark は H100 比 1/12 の帯域なので、speedup 上限が約 2x に張り付く構造。",
            ha="center", va="top", transform=ax.transAxes, fontsize=10, style="italic", color="#444")

    fig.tight_layout()
    out = IMG_DIR / "mtp-bandwidth-analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def chart_feasibility_matrix() -> None:
    """Feasibility table: model × backend × quantization."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    headers = ["モデル", "量子化", "vLLM 起動", "long speedup", "採択率", "品質劣化"]
    rows = []
    for size, label in zip(MODELS, MODEL_LABELS):
        b = load_long(f"{size}-longform-baseline")["warm"]["mean_tps"]
        m = load_long(f"{size}-longform-mtp2")["warm"]["mean_tps"]
        accept = load_long(f"{size}-longform-mtp2")["metrics_delta"]["acceptance_rate"]
        sp = m / b
        # 量子化
        if "BF16" in label:
            quant = "BF16"
        else:
            quant = "NVFP4"
        rows.append([
            label.split(" ")[0],
            quant,
            "✓ 動作",
            f"{sp:.2f}x ({m:.1f} tok/s)",
            f"{accept * 100:.1f}%",
            "なし (±0.5pt)",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Color header
    for i, _ in enumerate(headers):
        cell = table[(0, i)]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")

    # Color first column (model name)
    for i in range(1, len(rows) + 1):
        cell = table[(i, 0)]
        cell.set_facecolor("#ecf0f1")
        cell.set_text_props(fontweight="bold")
        # Speedup column color
        sp_cell = table[(i, 3)]
        sp_cell.set_facecolor("#e8f5e9")

    ax.set_title("DGX Spark での Gemma 4 MTP 可否マトリクス（vLLM 0.20.2rc1.dev99 / 2026-05-08）",
                 fontsize=12, pad=20)

    fig.tight_layout()
    out = IMG_DIR / "mtp-feasibility-matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print(f"output dir: {IMG_DIR}")
    chart_speedup_bar()
    chart_jcq_vs_long()
    chart_lang_comparison()
    chart_bandwidth_analysis()
    chart_feasibility_matrix()
    print("done")


if __name__ == "__main__":
    main()
