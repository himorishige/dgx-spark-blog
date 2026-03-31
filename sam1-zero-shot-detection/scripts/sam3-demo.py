#!/usr/bin/env python3
"""SAM3 zero-shot detection demo script.

Usage:
    uv run sam3-demo.py image.jpg --prompts "hard hat,safety vest,person"
    uv run sam3-demo.py image.jpg --prompts "laptop,coffee cup" --threshold 0.3
    uv run sam3-demo.py image.jpg  # uses default prompts
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "pillow"]
# ///

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8105"
DEFAULT_PROMPTS = "person,hard hat,safety vest,safety glasses,laptop,coffee cup"


def detect(
    image_path: str,
    prompts: list[str],
    threshold: float,
    base_url: str,
) -> dict:
    """Send image to SAM3 API and return detection results."""
    image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    resp = httpx.post(
        f"{base_url}/detect",
        json={
            "image_base64": image_b64,
            "prompts": prompts,
            "score_threshold": threshold,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="SAM3 zero-shot detection demo")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help="Comma-separated detection prompts (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="SAM3 API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file (default: stdout)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: {image_path} not found", file=sys.stderr)
        sys.exit(1)

    prompts = [p.strip() for p in args.prompts.split(",")]
    print(f"Image: {image_path}", file=sys.stderr)
    print(f"Prompts: {prompts}", file=sys.stderr)
    print(f"Threshold: {args.threshold}", file=sys.stderr)

    result = detect(str(image_path), prompts, args.threshold, args.url)

    # Summary to stderr
    print(f"\nDetected {len(result['tags'])} object types in {result['elapsed_ms']}ms:", file=sys.stderr)
    for det in result["detections"]:
        print(
            f"  - {det['prompt']}: {det['count']} instance(s), max_score={det['max_score']}",
            file=sys.stderr,
        )

    # Full result to stdout or file
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
