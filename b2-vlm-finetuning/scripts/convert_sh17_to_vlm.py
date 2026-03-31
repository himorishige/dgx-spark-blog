"""Convert SH17 PPE dataset (YOLO format) to TRL SFT dataset for Cosmos-Reason2.

Reads YOLO annotations and images, generates VQA-style conversation pairs
for PPE compliance evaluation, and saves as HuggingFace Dataset.

Usage:
    uv run python convert_sh17_to_vlm.py --sh17-dir /home/morishige/datasets/sh17 --output-dir ./data/sh17_vlm
"""

import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage

# SH17 class mapping
SH17_CLASSES = {
    0: "person",
    1: "head",
    2: "face",
    3: "hand",
    4: "safety-helmet",
    5: "no-helmet",
    6: "safety-gloves",
    7: "no-gloves",
    8: "safety-boots",
    9: "no-boots",
    10: "safety-jacket",
    11: "no-ppe-jacket",
    12: "glasses",
    13: "no-glasses",
    14: "ears",
    15: "earmuffs",
    16: "no-earmuffs",
}

# PPE items: (wearing_class_id, not_wearing_class_id, japanese_name, english_name)
PPE_ITEMS = [
    (4, 5, "安全ヘルメット", "safety helmet"),
    (6, 7, "安全手袋", "safety gloves"),
    (8, 9, "安全靴", "safety boots"),
    (10, 11, "安全ジャケット", "safety jacket"),
    (12, 13, "保護メガネ", "safety glasses"),
    (15, 16, "イヤーマフ", "earmuffs"),
]

# Question templates for variety
QUESTION_TEMPLATES = [
    "この画像の作業者のPPE（個人用保護具）装着状態を評価してください。",
    "この作業現場でPPEコンプライアンス違反はありますか？具体的に説明してください。",
    "画像に写っている作業者の安全装備の装着状況を確認してください。",
    "この画像で安全上の問題点を指摘してください。保護具の観点から分析してください。",
    "作業者が適切な保護具を装着しているか評価し、不足があれば指摘してください。",
]


def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a YOLO format label file, skipping malformed lines."""
    annotations = []
    if not label_path.exists():
        return annotations

    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        # Skip lines with [xN] prefix (some SH17 files have this)
        line = line.strip()
        if line.startswith("["):
            # Extract after the bracket notation
            bracket_end = line.find("]")
            if bracket_end >= 0:
                line = line[bracket_end + 1 :].strip()

        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append(
                {
                    "class_id": class_id,
                    "class_name": SH17_CLASSES.get(class_id, f"unknown-{class_id}"),
                    "bbox": [x_center, y_center, width, height],
                }
            )
        except (ValueError, IndexError):
            continue

    return annotations


def analyze_ppe_status(annotations: list[dict]) -> dict:
    """Analyze PPE compliance from annotations."""
    person_count = sum(1 for a in annotations if a["class_id"] == 0)

    wearing = {}
    not_wearing = {}

    for wear_id, no_wear_id, ja_name, en_name in PPE_ITEMS:
        wear_count = sum(1 for a in annotations if a["class_id"] == wear_id)
        no_wear_count = sum(1 for a in annotations if a["class_id"] == no_wear_id)
        if wear_count > 0:
            wearing[ja_name] = wear_count
        if no_wear_count > 0:
            not_wearing[ja_name] = no_wear_count

    violations = []
    for item_name, count in not_wearing.items():
        violations.append(f"{item_name}未装着: {count}件")

    compliant_items = []
    for item_name, count in wearing.items():
        compliant_items.append(f"{item_name}装着: {count}件")

    return {
        "person_count": person_count,
        "wearing": wearing,
        "not_wearing": not_wearing,
        "violations": violations,
        "compliant_items": compliant_items,
        "is_compliant": len(not_wearing) == 0 and len(wearing) > 0,
    }


def generate_think_answer(ppe_status: dict) -> str:
    """Generate <think>/<answer> format response."""
    person_count = ppe_status["person_count"]
    violations = ppe_status["violations"]
    compliant_items = ppe_status["compliant_items"]

    # Think section: detailed analysis
    think_parts = []
    if person_count > 0:
        think_parts.append(f"画像には{person_count}名の作業者が確認できます。")
    else:
        think_parts.append("画像から作業者を確認します。")

    if compliant_items:
        think_parts.append("装着確認済み: " + "、".join(compliant_items) + "。")

    if violations:
        think_parts.append("違反検出: " + "、".join(violations) + "。")
    else:
        if compliant_items:
            think_parts.append("確認できる範囲でPPE違反は検出されませんでした。")

    think = " ".join(think_parts)

    # Answer section: concise summary
    if violations:
        violation_summary = "、".join(violations)
        answer = f"PPEコンプライアンス違反があります。{violation_summary}。"
    elif compliant_items:
        answer = "確認できる範囲では、作業者は適切にPPEを装着しています。"
    else:
        answer = "この画像からはPPE装着状態を判定するための十分な情報が得られません。"

    return f"<think>\n{think}\n</think>\n{answer}"


def resize_image(image: PILImage.Image, max_size: int = 1280) -> PILImage.Image:
    """Resize image so the longest side is at most max_size."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), PILImage.LANCZOS)


def convert_split(
    sh17_dir: Path, split: str, max_samples: int | None = None
) -> list[dict]:
    """Convert one split of SH17 to VLM format."""
    images_dir = sh17_dir / "images" / split
    labels_dir = sh17_dir / "labels" / split

    samples = []
    image_files = sorted(images_dir.iterdir())

    if max_samples:
        image_files = image_files[:max_samples]

    for img_path in image_files:
        # Find corresponding label
        label_path = labels_dir / (img_path.stem + ".txt")
        annotations = parse_yolo_label(label_path)

        # Skip images with no annotations
        if not annotations:
            continue

        # Check if there are PPE-related annotations (not just person/head/face/hand/ears)
        ppe_class_ids = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16}
        has_ppe = any(a["class_id"] in ppe_class_ids for a in annotations)
        if not has_ppe:
            continue

        ppe_status = analyze_ppe_status(annotations)
        response = generate_think_answer(ppe_status)
        question = random.choice(QUESTION_TEMPLATES)

        # Load and resize image (VLM input optimization)
        try:
            image = PILImage.open(img_path).convert("RGB")
            image = resize_image(image, max_size=1280)
        except Exception:
            continue

        # TRL SFT format: messages with image
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            ],
            "images": [image],
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="Convert SH17 to VLM SFT dataset")
    parser.add_argument(
        "--sh17-dir",
        type=Path,
        default=Path("/home/morishige/datasets/sh17"),
        help="Path to SH17 dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/sh17_vlm"),
        help="Output directory for HF dataset",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Max training samples (None for all)",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Max validation samples (None for all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"SH17 directory: {args.sh17_dir}")
    print(f"Output directory: {args.output_dir}")

    # Convert each split
    print("\n=== Converting train split ===")
    train_samples = convert_split(args.sh17_dir, "train", args.max_train)
    print(f"Train samples with PPE annotations: {len(train_samples)}")

    print("\n=== Converting val split ===")
    val_samples = convert_split(args.sh17_dir, "val", args.max_val)
    print(f"Val samples with PPE annotations: {len(val_samples)}")

    print("\n=== Converting test split ===")
    test_samples = convert_split(args.sh17_dir, "test", args.max_val)
    print(f"Test samples with PPE annotations: {len(test_samples)}")

    # Create HuggingFace Dataset
    def samples_to_dataset(samples):
        if not samples:
            return Dataset.from_dict({"messages": [], "images": []})
        return Dataset.from_dict(
            {
                "messages": [s["messages"] for s in samples],
                "images": [s["images"] for s in samples],
            }
        )

    dataset = DatasetDict(
        {
            "train": samples_to_dataset(train_samples),
            "validation": samples_to_dataset(val_samples),
            "test": samples_to_dataset(test_samples),
        }
    )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"\nDataset saved to {args.output_dir}")

    # Print sample
    if train_samples:
        sample = train_samples[0]
        print("\n=== Sample entry ===")
        print(f"Question: {sample['messages'][0]['content'][1]['text']}")
        print(f"Answer: {sample['messages'][1]['content'][0]['text'][:200]}...")
        print(f"Image size: {sample['images'][0].size}")

    # Stats
    total = len(train_samples) + len(val_samples) + len(test_samples)
    compliant = sum(
        1 for s in train_samples
        if "違反" not in s["messages"][1]["content"][0]["text"]
        or "ありません" in s["messages"][1]["content"][0]["text"]
    )
    print(f"\n=== Statistics ===")
    print(f"Total samples: {total}")
    print(f"Train compliant (no violations): {compliant}/{len(train_samples)}")
    print(
        f"Train with violations: {len(train_samples) - compliant}/{len(train_samples)}"
    )


if __name__ == "__main__":
    main()
