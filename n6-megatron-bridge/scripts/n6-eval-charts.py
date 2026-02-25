"""Generate evaluation charts for N6 Megatron-Bridge article."""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# Style settings
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK JP",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

OUTPUT_DIR = Path(__file__).parent.parent / "images"

# Color palette (matching article's blue theme)
C_BASELINE = "#9E9E9E"  # gray
C_HF_PEFT = "#42A5F5"  # blue
C_MEGATRON = "#EF5350"  # red-ish (to highlight the surprise)
C_ANSWERED = "#66BB6A"  # green (positive signal)


def chart1_eval_results():
    """F1 Score and Rejection Count comparison (2-panel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    models = ["Baseline\n（FT なし）", "HF PEFT\n（53% LoRA）", "Megatron-Bridge\n（100% LoRA）"]
    colors = [C_BASELINE, C_HF_PEFT, C_MEGATRON]
    x = np.arange(len(models))
    width = 0.55

    # --- Left: F1 Score ---
    f1_scores = [0.5646, 0.6536, 0.4884]
    bars1 = ax1.bar(x, f1_scores, width, color=colors, edgecolor="white", linewidth=1.5)

    ax1.set_ylabel("F1 スコア", fontsize=14)
    ax1.set_title("RAFT ドメイン F1", fontsize=15, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.set_ylim(0, 0.85)
    ax1.axhline(y=f1_scores[0], color=C_BASELINE, linestyle=":", alpha=0.5, linewidth=1)

    for bar, val in zip(bars1, f1_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # Annotate the drop
    ax1.annotate(
        "ベースラインを\n下回る",
        xy=(2, 0.4884),
        xytext=(2.45, 0.62),
        fontsize=11,
        color=C_MEGATRON,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_MEGATRON, lw=1.5),
        ha="center",
    )

    # --- Right: Rejection Count (FP) ---
    rejections = [47, 3, 69]
    bars2 = ax2.bar(x, rejections, width, color=colors, edgecolor="white", linewidth=1.5)

    ax2.set_ylabel("回答拒否数（FP）", fontsize=14)
    ax2.set_title("誤った回答拒否", fontsize=15, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.set_ylim(0, 90)

    for bar, val in zip(bars2, rejections):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val} 件",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # Annotate the increase
    ax2.annotate(
        "47→3 件に改善した\n前回から逆戻り",
        xy=(2, 69),
        xytext=(1.2, 80),
        fontsize=11,
        color=C_MEGATRON,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_MEGATRON, lw=1.5),
        ha="center",
    )

    fig.suptitle(
        "評価結果：100% LoRA にしたのに精度が悪化",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out = OUTPUT_DIR / "n6-eval-results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


def chart2_f1_paradox():
    """Overall F1 vs Answered-only F1 (the paradox)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    models = ["HF PEFT（53% LoRA）", "Megatron-Bridge（100% LoRA）"]
    overall_f1 = [0.6536, 0.4884]
    answered_f1 = [0.6636, 0.7457]

    x = np.arange(len(models))
    width = 0.32

    bars1 = ax.bar(
        x - width / 2,
        overall_f1,
        width,
        label="全体 F1",
        color=C_HF_PEFT,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        answered_f1,
        width,
        label="回答時 F1（回答したサンプルのみ）",
        color=C_ANSWERED,
        edgecolor="white",
        linewidth=1.5,
    )

    # Value labels
    for bar, val in zip(bars1, overall_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=C_HF_PEFT,
        )
    for bar, val in zip(bars2, answered_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#388E3C",
        )

    # Highlight the paradox with annotation
    ax.annotate(
        "全体 F1 は悪化\nしかし回答時 F1 は向上",
        xy=(1 + width / 2, 0.7457),
        xytext=(1.55, 0.58),
        fontsize=12,
        fontweight="bold",
        color="#E65100",
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.5),
        ha="center",
    )

    ax.set_ylabel("F1 スコア", fontsize=14)
    ax.set_title(
        "回答の質は向上、しかし拒否しすぎて全体 F1 は悪化",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["HF PEFT（53% LoRA）\n回答 197/200 件", "Megatron-Bridge（100% LoRA）\n回答 131/200 件"],
        fontsize=12,
    )
    ax.set_ylim(0, 0.88)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    fig.tight_layout()

    out = OUTPUT_DIR / "n6-f1-paradox.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    chart1_eval_results()
    chart2_f1_paradox()
    print("Done.")
