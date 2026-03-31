#!/usr/bin/env python3
"""Visualize SAM3 detection results by drawing bounding boxes on images.

Usage:
    uv run sam3-demo.py image.jpg --output result.json
    uv run sam3-visualize.py image.jpg result.json --output annotated.jpg

    # Or pipe from demo:
    uv run sam3-demo.py image.jpg | uv run sam3-visualize.py image.jpg - --output annotated.jpg
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Color palette for different object types
COLORS = [
    "#FF6B6B",  # red
    "#4ECDC4",  # teal
    "#45B7D1",  # sky blue
    "#96CEB4",  # sage
    "#FFEAA7",  # yellow
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F7DC6F",  # gold
    "#BB8FCE",  # purple
    "#85C1E9",  # light blue
]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def draw_detections(image_path: str, detections: list[dict], output_path: str) -> None:
    """Draw bounding boxes and labels on an image."""
    img = Image.open(image_path).convert("RGB")
    # Resize to match SAM3 processing (1024px width)
    w, h = img.size
    new_w = 1024
    new_h = int(1024 * h / w)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    draw = ImageDraw.Draw(img_resized)

    # Try to load a reasonable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    for i, det in enumerate(detections):
        color_rgb = hex_to_rgb(COLORS[i % len(COLORS)])
        prompt = det["prompt"]
        boxes = det.get("boxes", [])
        max_score = det.get("max_score", 0)

        for box in boxes:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box

            # Draw box with thicker outline
            for offset in range(3):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=color_rgb,
                )

            # Draw label background
            label = f"{prompt} ({max_score:.2f})"
            bbox = draw.textbbox((x1, y1), label, font=font_small)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            draw.rectangle(
                [x1, y1 - label_h - 6, x1 + label_w + 8, y1],
                fill=color_rgb,
            )
            draw.text((x1 + 4, y1 - label_h - 4), label, fill="white", font=font_small)

    # Draw legend at bottom
    legend_y = new_h - 30
    legend_x = 10
    for i, det in enumerate(detections):
        color_rgb = hex_to_rgb(COLORS[i % len(COLORS)])
        label = f"{det['prompt']} ({det['count']})"
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color_rgb)
        draw.text((legend_x + 16, legend_y - 2), label, fill="white", font=font_small)
        bbox = draw.textbbox((legend_x + 16, legend_y), label, font=font_small)
        legend_x = bbox[2] + 20

    img_resized.save(output_path, quality=95)
    print(f"Saved annotated image: {output_path} ({new_w}x{new_h})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3 detection results")
    parser.add_argument("image", help="Path to original image")
    parser.add_argument(
        "result",
        help="Path to detection result JSON (or '-' for stdin)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="annotated.jpg",
        help="Output image path (default: %(default)s)",
    )
    args = parser.parse_args()

    # Load detection results
    if args.result == "-":
        data = json.load(sys.stdin)
    else:
        data = json.loads(Path(args.result).read_text())

    detections = data.get("detections", [])
    if not detections:
        print("No detections found in result", file=sys.stderr)
        sys.exit(0)

    draw_detections(args.image, detections, args.output)


if __name__ == "__main__":
    main()
