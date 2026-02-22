"""N3 article: Plot training loss and token accuracy curves."""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Matplotlib Japanese font support
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAexGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    metrics_path = Path(__file__).parent / "data/n3/ngc-adapter/training_metrics.json"
    output_path = Path(__file__).parent / "data/n3/n3-training-loss.png"

    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Filter out the final summary entry (has train_runtime key)
    steps_data = [m for m in metrics if "train_runtime" not in m]

    steps = [m["step"] for m in steps_data]
    losses = [m["loss"] for m in steps_data]
    accuracies = [m["mean_token_accuracy"] * 100 for m in steps_data]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Loss (left axis)
    color_loss = "#1976d2"
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Loss", color=color_loss, fontsize=12)
    (line1,) = ax1.plot(
        steps, losses, color=color_loss, linewidth=2, marker="o", markersize=5, label="Loss"
    )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(5.5, 11.5)

    # Token Accuracy (right axis)
    ax2 = ax1.twinx()
    color_acc = "#388e3c"
    ax2.set_ylabel("Mean Token Accuracy (%)", color=color_acc, fontsize=12)
    (line2,) = ax2.plot(
        steps,
        accuracies,
        color=color_acc,
        linewidth=2,
        marker="s",
        markersize=5,
        label="Token Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(65, 82)

    # Annotations for key points
    ax1.annotate(
        f"Loss: {losses[0]:.1f}",
        xy=(steps[0], losses[0]),
        xytext=(steps[0] + 10, losses[0] + 0.3),
        fontsize=9,
        color=color_loss,
    )
    min_loss_idx = np.argmin(losses)
    ax1.annotate(
        f"Loss: {losses[min_loss_idx]:.2f}\n(Step {steps[min_loss_idx]})",
        xy=(steps[min_loss_idx], losses[min_loss_idx]),
        xytext=(steps[min_loss_idx] - 30, losses[min_loss_idx] - 0.6),
        fontsize=9,
        color=color_loss,
        arrowprops=dict(arrowstyle="->", color=color_loss, lw=1),
    )

    # Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=10)

    # Title and grid
    ax1.set_title(
        "BF16 LoRA Training (1,100 samples × 1 epoch)",
        fontsize=14,
        pad=15,
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, max(steps) + 10, 20))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
