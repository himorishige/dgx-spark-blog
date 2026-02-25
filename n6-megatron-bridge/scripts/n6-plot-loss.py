#!/usr/bin/env python3
"""N6 Megatron-Bridge Training Loss Plot

Extract iteration / lm loss from training.txt and generate a loss curve chart.
Style matches N3's n3-training-loss.png (single y-axis, Loss only).

Usage:
    python n6-plot-loss.py
    python n6-plot-loss.py --log /path/to/training.txt --output /path/to/output.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

DEFAULT_LOG = Path("/home/morishige/n6-logs/logs/training.txt")
DEFAULT_OUTPUT = Path(__file__).parent.parent / "images" / "n6-training-loss.png"


def extract_loss_data(log_path: Path) -> tuple[list[int], list[float]]:
    """Extract (iteration, lm_loss) pairs from Megatron training log."""
    pattern = re.compile(
        r"iteration\s+(\d+)/\s*\d+\s.*?lm loss:\s*([\d.]+E[+-]?\d+)"
    )
    iterations = []
    losses = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                it = int(m.group(1))
                loss = float(m.group(2))
                iterations.append(it)
                losses.append(loss)

    return iterations, losses


def plot_loss(
    iterations: list[int],
    losses: list[float],
    output_path: Path,
) -> None:
    """Generate training loss curve chart matching N3 style."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Loss line (blue, matching N3 style)
    color_loss = "#1976D2"
    ax.plot(iterations, losses, color=color_loss, linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("lm loss", fontsize=12, color=color_loss)
    ax.tick_params(axis="y", labelcolor=color_loss)
    ax.set_xlim(0, max(iterations) + 10)
    ax.set_ylim(0, max(losses[:10]) * 1.1)  # Scale based on early (high) loss

    # Annotate key points
    # First iteration
    ax.annotate(
        f"Loss: {losses[0]:.3f}",
        xy=(iterations[0], losses[0]),
        xytext=(30, 10),
        textcoords="offset points",
        fontsize=9,
        color=color_loss,
        arrowprops=dict(arrowstyle="->", color=color_loss, lw=0.8),
    )

    # Final iteration
    ax.annotate(
        f"Loss: {losses[-1]:.3f}\n(Iter {iterations[-1]})",
        xy=(iterations[-1], losses[-1]),
        xytext=(-80, 30),
        textcoords="offset points",
        fontsize=9,
        color=color_loss,
        arrowprops=dict(arrowstyle="->", color=color_loss, lw=0.8),
    )

    # Minimum loss
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    min_iter = iterations[min_idx]
    if min_iter != iterations[-1]:
        ax.annotate(
            f"Min: {min_loss:.3f}\n(Iter {min_iter})",
            xy=(min_iter, min_loss),
            xytext=(-80, -30),
            textcoords="offset points",
            fontsize=9,
            color=color_loss,
            arrowprops=dict(arrowstyle="->", color=color_loss, lw=0.8),
        )

    ax.set_title(
        "Megatron-Bridge 100% LoRA Training (1,100 samples × 500 iter)",
        fontsize=14,
        pad=15,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="N6 Training Loss Plot")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Error: {args.log} not found")
        return

    iterations, losses = extract_loss_data(args.log)
    print(f"Extracted {len(iterations)} data points from {args.log}")
    print(f"  Iterations: {iterations[0]} - {iterations[-1]}")
    print(f"  Loss range: {min(losses):.4f} - {max(losses):.4f}")

    plot_loss(iterations, losses, args.output)


if __name__ == "__main__":
    main()
