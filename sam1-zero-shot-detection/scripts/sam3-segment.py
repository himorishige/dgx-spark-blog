#!/usr/bin/env python3
"""SAM3 segmentation mask visualization.

Calls /segment endpoint and renders color-coded masks overlaid on the image.

Usage:
    uv run sam3-segment.py image.jpg --prompts "hard hat,safety vest,person" -o segmented.jpg
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "pillow", "numpy"]
# ///

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_URL = "http://localhost:8105"

# Semi-transparent colors for mask overlay (RGBA)
MASK_COLORS = [
    (255, 107, 107, 100),  # red
    (78, 205, 196, 100),   # teal
    (69, 183, 209, 100),   # sky blue
    (150, 206, 180, 100),  # sage
    (255, 234, 167, 100),  # yellow
    (221, 160, 221, 100),  # plum
    (152, 216, 200, 100),  # mint
    (247, 220, 111, 100),  # gold
    (187, 143, 206, 100),  # purple
    (133, 193, 233, 100),  # light blue
]

BORDER_COLORS = [
    (255, 107, 107),
    (78, 205, 196),
    (69, 183, 209),
    (150, 206, 180),
    (255, 234, 167),
    (221, 160, 221),
    (152, 216, 200),
    (247, 220, 111),
    (187, 143, 206),
    (133, 193, 233),
]


def rle_to_mask(rle: dict) -> np.ndarray:
    """Decode RLE to binary mask."""
    h, w = rle["size"]
    counts = [int(c) for c in rle["counts"].split(",")]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:  # odd index = ones
            flat[pos : pos + c] = 1
        pos += c
    return flat.reshape(h, w)


def segment(image_path: str, prompts: list[str], threshold: float, base_url: str) -> dict:
    """Call /segment endpoint."""
    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = httpx.post(
        f"{base_url}/segment",
        json={
            "image_base64": image_b64,
            "prompts": prompts,
            "score_threshold": threshold,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def render_masks(image_path: str, result: dict, output_path: str) -> None:
    """Render color-coded segmentation masks on the image."""
    img = Image.open(image_path).convert("RGB")
    h, w = result["image_size"]
    img_resized = img.resize((w, h), Image.LANCZOS)

    # Create mask overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for i, det in enumerate(result["detections"]):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        border = BORDER_COLORS[i % len(BORDER_COLORS)]

        for rle in det["masks_rle"]:
            mask = rle_to_mask(rle)
            # Create colored mask
            mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_rgba[mask == 1] = color
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_img)

    # Composite
    base_rgba = img_resized.convert("RGBA")
    composited = Image.alpha_composite(base_rgba, overlay)
    result_img = composited.convert("RGB")

    # Draw labels and legend
    draw = ImageDraw.Draw(result_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Draw boxes with labels
    for i, det in enumerate(result["detections"]):
        border = BORDER_COLORS[i % len(BORDER_COLORS)]
        for j, box in enumerate(det["boxes"]):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            score = det["scores"][j] if j < len(det["scores"]) else 0
            # Box outline
            for offset in range(2):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=border)
            # Label
            label = f"{det['prompt']} ({score:.2f})"
            bbox = draw.textbbox((x1, y1), label, font=font_small)
            lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - lh - 6, x1 + lw + 8, y1], fill=border)
            draw.text((x1 + 4, y1 - lh - 4), label, fill="white", font=font_small)

    # Legend bar at bottom
    legend_y = h - 28
    legend_x = 10
    for i, det in enumerate(result["detections"]):
        color = BORDER_COLORS[i % len(BORDER_COLORS)]
        label = f"{det['prompt']} ({det['count']})"
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=color)
        draw.text((legend_x + 16, legend_y - 2), label, fill="white", font=font_small)
        bbox = draw.textbbox((legend_x + 16, legend_y), label, font=font_small)
        legend_x = bbox[2] + 20

    result_img.save(output_path, quality=95)
    print(f"Saved segmented image: {output_path} ({w}x{h})", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="SAM3 segmentation mask visualization")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--prompts", default="person,car,laptop", help="Comma-separated prompts")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--url", default=DEFAULT_URL, help="SAM3 API base URL")
    parser.add_argument("-o", "--output", default="segmented.jpg", help="Output image path")
    parser.add_argument("--save-json", help="Also save raw JSON result")
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",")]
    print(f"Prompts: {prompts}", file=sys.stderr)

    result = segment(args.image, prompts, args.threshold, args.url)

    print(f"Detected {len(result['tags'])} types in {result['elapsed_ms']}ms", file=sys.stderr)
    for det in result["detections"]:
        print(f"  - {det['prompt']}: {det['count']} instance(s)", file=sys.stderr)

    if args.save_json:
        # Save without mask data (too large)
        slim = {**result, "detections": [
            {k: v for k, v in d.items() if k != "masks_rle"} for d in result["detections"]
        ]}
        Path(args.save_json).write_text(json.dumps(slim, indent=2, ensure_ascii=False))

    render_masks(args.image, result, args.output)


if __name__ == "__main__":
    main()
