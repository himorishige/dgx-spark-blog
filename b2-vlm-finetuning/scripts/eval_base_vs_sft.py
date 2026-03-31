"""Evaluate Base vs LoRA SFT Cosmos-Reason2-8B on SH17 PPE test set.

Compares base model and fine-tuned model on the same test images,
measuring PPE detection accuracy and response quality.

Usage:
    uv run python eval_base_vs_sft.py
    uv run python eval_base_vs_sft.py --num-samples 20
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_base_model(model_name: str):
    """Load base model and processor."""
    print(f"Loading base model: {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def load_sft_model(model_name: str, adapter_path: str):
    """Load base model with LoRA adapter."""
    print(f"Loading SFT model: {model_name} + {adapter_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def run_inference(model, processor, image, question: str, max_new_tokens: int = 512) -> tuple[str, float]:
    """Run inference and return response text and time."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.time() - start

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text, elapsed


def extract_answer(response: str) -> str:
    """Extract the answer part from <think>...</think>\\nanswer format."""
    if "</think>" in response:
        return response.split("</think>")[-1].strip()
    return response.strip()


def check_violation_detection(response: str, ground_truth: str) -> dict:
    """Check if the model correctly identifies PPE violations."""
    gt_has_violation = "違反があります" in ground_truth
    resp_answer = extract_answer(response)

    # Check if response mentions violations
    violation_keywords = ["違反", "未装着", "装着していない", "着用していない", "not wearing", "missing", "without"]
    compliant_keywords = ["適切に", "正しく装着", "問題ありません", "compliant", "properly"]

    resp_has_violation = any(k in response for k in violation_keywords)
    resp_is_compliant = any(k in response for k in compliant_keywords)

    # Determine prediction
    if resp_has_violation and not resp_is_compliant:
        pred_violation = True
    elif resp_is_compliant and not resp_has_violation:
        pred_violation = False
    else:
        pred_violation = resp_has_violation  # default to violation detection

    correct = (pred_violation == gt_has_violation)

    return {
        "gt_has_violation": gt_has_violation,
        "pred_has_violation": pred_violation,
        "correct": correct,
        "has_think_tag": "<think>" in response and "</think>" in response,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--adapter-path", default="./outputs/Cosmos-Reason2-8B-ppe-lora")
    parser.add_argument("--dataset-dir", default="./data/sh17_vlm")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--output-file", default="./outputs/eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load test dataset
    print("Loading test dataset...")
    dataset = load_from_disk(args.dataset_dir)
    test_dataset = dataset["test"]
    print(f"Test samples available: {len(test_dataset)}")

    # Sample indices
    import random
    random.seed(args.seed)
    indices = random.sample(range(len(test_dataset)), min(args.num_samples, len(test_dataset)))
    print(f"Evaluating {len(indices)} samples")

    # Standard question for all evaluations
    eval_question = "この画像の作業者のPPE（個人用保護具）装着状態を評価してください。違反があれば具体的に指摘してください。"

    results = []

    # === Phase A: Base model ===
    print("\n" + "=" * 60)
    print("Phase A: Base Model Evaluation")
    print("=" * 60)
    base_model, base_processor = load_base_model(args.model_name)

    if torch.cuda.is_available():
        print(f"GPU memory after base model load: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")

    base_responses = []
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample["images"][0]
        gt_text = sample["messages"][1]["content"][0]["text"]

        response, elapsed = run_inference(base_model, base_processor, image, eval_question)
        check = check_violation_detection(response, gt_text)

        base_responses.append({
            "idx": idx,
            "response": response,
            "time": elapsed,
            **check,
        })
        status = "✓" if check["correct"] else "✗"
        print(f"  [{i+1}/{len(indices)}] {status} {elapsed:.1f}s | think={check['has_think_tag']} | GT={'violation' if check['gt_has_violation'] else 'ok'} Pred={'violation' if check['pred_has_violation'] else 'ok'}")

    # Free base model
    del base_model, base_processor
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # === Phase B: SFT model ===
    print("\n" + "=" * 60)
    print("Phase B: SFT Model Evaluation")
    print("=" * 60)
    sft_model, sft_processor = load_sft_model(args.model_name, args.adapter_path)

    if torch.cuda.is_available():
        print(f"GPU memory after SFT model load: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")

    sft_responses = []
    for i, idx in enumerate(indices):
        sample = test_dataset[idx]
        image = sample["images"][0]
        gt_text = sample["messages"][1]["content"][0]["text"]

        response, elapsed = run_inference(sft_model, sft_processor, image, eval_question)
        check = check_violation_detection(response, gt_text)

        sft_responses.append({
            "idx": idx,
            "response": response,
            "time": elapsed,
            **check,
        })
        status = "✓" if check["correct"] else "✗"
        print(f"  [{i+1}/{len(indices)}] {status} {elapsed:.1f}s | think={check['has_think_tag']} | GT={'violation' if check['gt_has_violation'] else 'ok'} Pred={'violation' if check['pred_has_violation'] else 'ok'}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    base_correct = sum(1 for r in base_responses if r["correct"])
    sft_correct = sum(1 for r in sft_responses if r["correct"])
    base_think = sum(1 for r in base_responses if r["has_think_tag"])
    sft_think = sum(1 for r in sft_responses if r["has_think_tag"])
    base_avg_time = sum(r["time"] for r in base_responses) / len(base_responses)
    sft_avg_time = sum(r["time"] for r in sft_responses) / len(sft_responses)
    n = len(indices)

    print(f"\n{'Metric':<30} {'Base':>10} {'SFT':>10} {'Delta':>10}")
    print("-" * 62)
    print(f"{'Violation Detection Acc':<30} {base_correct/n*100:>9.1f}% {sft_correct/n*100:>9.1f}% {(sft_correct-base_correct)/n*100:>+9.1f}%")
    print(f"{'<think> Tag Usage':<30} {base_think/n*100:>9.1f}% {sft_think/n*100:>9.1f}% {(sft_think-base_think)/n*100:>+9.1f}%")
    print(f"{'Avg Inference Time':<30} {base_avg_time:>9.1f}s {sft_avg_time:>9.1f}s {sft_avg_time-base_avg_time:>+9.1f}s")
    print(f"{'Samples':<30} {n:>10} {n:>10}")

    # Show a few example comparisons
    print("\n" + "=" * 60)
    print("EXAMPLE COMPARISONS (first 3)")
    print("=" * 60)
    for i in range(min(3, n)):
        print(f"\n--- Sample {i+1} (idx={indices[i]}) ---")
        print(f"Ground Truth: {test_dataset[indices[i]]['messages'][1]['content'][0]['text'][:200]}")
        print(f"\nBase ({base_responses[i]['time']:.1f}s):")
        print(f"  {base_responses[i]['response'][:300]}")
        print(f"\nSFT ({sft_responses[i]['time']:.1f}s):")
        print(f"  {sft_responses[i]['response'][:300]}")

    # Save results
    output = {
        "config": {
            "model": args.model_name,
            "adapter": args.adapter_path,
            "num_samples": n,
            "eval_question": eval_question,
        },
        "summary": {
            "base_accuracy": base_correct / n,
            "sft_accuracy": sft_correct / n,
            "base_think_rate": base_think / n,
            "sft_think_rate": sft_think / n,
            "base_avg_time": base_avg_time,
            "sft_avg_time": sft_avg_time,
        },
        "base_responses": base_responses,
        "sft_responses": sft_responses,
    }

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
