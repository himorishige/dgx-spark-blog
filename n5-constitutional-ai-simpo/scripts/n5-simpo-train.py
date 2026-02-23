#!/usr/bin/env python3
"""N5 SimPO Training Script

Train Nemotron 9B-v2-Japanese with SimPO (Simple Preference Optimization)
using Constitutional AI-generated preference pairs.

SimPO uses CPOTrainer with loss_type="simpo" from TRL.
No reference model needed (memory efficient for Mamba-2 Hybrid).

Fallback chain: SimPO -> DPO -> ORPO -> SFT

Prerequisites:
  - NGC NeMo container (nvcr.io/nvidia/nemo:25.11.01 or newer)
    OR Python environment with: trl, peft, transformers, torch
  - DGX Spark with 128GB unified memory
  - Training data: data/n5/train.jsonl (chosen/rejected pairs)

Usage (inside NGC container or with deps installed):
  # Phase 0: Verify CPOTrainer works with Nemotron-H
  python n5-simpo-train.py verify --data-file ./data/n5/train.jsonl

  # SimPO training
  python n5-simpo-train.py train --data-file ./data/n5/train.jsonl

  # DPO fallback
  python n5-simpo-train.py train --data-file ./data/n5/train.jsonl --method dpo

  # ORPO fallback
  python n5-simpo-train.py train --data-file ./data/n5/train.jsonl --method orpo

  # SFT fallback (chosen only)
  python n5-simpo-train.py train --data-file ./data/n5/train.jsonl --method sft

Reference:
  - SimPO: arXiv:2405.14734
  - DPO: arXiv:2305.18290
  - TRL CPOTrainer: https://huggingface.co/docs/trl/cpo_trainer
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"

# LoRA target modules (Attention + FFN only, 53% coverage)
# Mamba-2 layers (in_proj, out_proj) are NOT supported by HF PEFT
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # FFN (MLP)
]


def load_preference_dataset(data_file: str, max_samples: int | None = None):
    """Load preference dataset from JSONL file.

    Expected format per line:
    {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}

    Returns HuggingFace Dataset with columns: prompt, chosen, rejected
    """
    from datasets import Dataset

    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

    if max_samples:
        records = records[:max_samples]

    return Dataset.from_list(records)


def load_sft_dataset(data_file: str, max_samples: int | None = None):
    """Load SFT dataset from preference data (chosen only).

    Converts preference pairs to SFT format using only chosen responses.
    """
    from datasets import Dataset

    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append({
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]},
                ],
            })

    if max_samples:
        records = records[:max_samples]

    return Dataset.from_list(records)


def get_lora_config(lora_r: int = 16, lora_alpha: int = 32):
    """Get LoRA configuration for Nemotron-H."""
    from peft import LoraConfig

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer():
    """Load Nemotron 9B v2 Japanese model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_ID}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" can offload Mamba-2 layers to meta/CPU,
    # causing gradient errors in CPOTrainer backward pass.
    # DGX Spark's 128GB unified memory fits the full model on GPU.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded: {mem_gb:.1f} GB GPU memory")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Phase 0: Verification
# ---------------------------------------------------------------------------


def cmd_verify(args):
    """Phase 0: Verify CPOTrainer works with Nemotron-H.

    Runs a minimal training step (10 samples, 1 step) to check compatibility.
    Tests SimPO, DPO, ORPO in sequence; reports which methods work.
    """
    try:
        from trl.experimental.cpo import CPOConfig, CPOTrainer
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl import CPOConfig, CPOTrainer, ORPOConfig, ORPOTrainer
    from trl import DPOConfig, DPOTrainer

    data_file = args.data_file
    print("=" * 60)
    print("Phase 0: CPOTrainer Compatibility Verification")
    print("=" * 60)

    # Load minimal dataset
    dataset = load_preference_dataset(data_file, max_samples=10)
    print(f"  Test samples: {len(dataset)}")

    model, tokenizer = load_model_and_tokenizer()
    lora_config = get_lora_config()

    output_base = Path(args.output_dir) / "verify"
    results = {}

    # Test SimPO
    print("\n--- Testing SimPO (CPOTrainer + loss_type=simpo) ---")
    try:
        config = CPOConfig(
            output_dir=str(output_base / "simpo"),
            loss_type="simpo",
            cpo_alpha=0.5,
            simpo_gamma=0.5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=5e-5,
            bf16=True,
            max_length=512,
            max_completion_length=256,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = CPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()
        results["simpo"] = "OK"
        print("  SimPO: OK")

        # Clean up PEFT adapter for next test
        model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    except Exception as e:
        results["simpo"] = f"FAILED: {e}"
        print(f"  SimPO: FAILED - {e}")

    # Reload model for DPO test
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_model_and_tokenizer()

    # Test DPO
    print("\n--- Testing DPO (DPOTrainer) ---")
    try:
        config = DPOConfig(
            output_dir=str(output_base / "dpo"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=5e-6,
            beta=0.1,
            bf16=True,
            max_length=512,
            max_prompt_length=256,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = DPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()
        results["dpo"] = "OK"
        print("  DPO: OK")
    except Exception as e:
        results["dpo"] = f"FAILED: {e}"
        print(f"  DPO: FAILED - {e}")

    # Reload model for ORPO test
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_model_and_tokenizer()

    # Test ORPO
    print("\n--- Testing ORPO (ORPOTrainer) ---")
    try:
        config = ORPOConfig(
            output_dir=str(output_base / "orpo"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,
            learning_rate=5e-5,
            beta=0.1,
            bf16=True,
            max_length=512,
            max_completion_length=256,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = ORPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()
        results["orpo"] = "OK"
        print("  ORPO: OK")
    except Exception as e:
        results["orpo"] = f"FAILED: {e}"
        print(f"  ORPO: FAILED - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 Results:")
    print("=" * 60)
    for method, status in results.items():
        emoji = "GO" if status == "OK" else "NO-GO"
        print(f"  [{emoji}] {method}: {status}")

    # Recommend best method
    if results.get("simpo") == "OK":
        print(f"\n  Recommendation: Use SimPO (best memory efficiency)")
    elif results.get("dpo") == "OK":
        print(f"\n  Recommendation: Use DPO (fallback #1)")
    elif results.get("orpo") == "OK":
        print(f"\n  Recommendation: Use ORPO (fallback #2)")
    else:
        print(f"\n  Recommendation: Use SFT only (all preference methods failed)")

    # Save results
    results_path = output_base / "verify_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {results_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_simpo(model, tokenizer, dataset, lora_config, args):
    """Train with SimPO (CPOTrainer + loss_type=simpo)."""
    try:
        from trl.experimental.cpo import CPOConfig, CPOTrainer
    except ImportError:
        from trl import CPOConfig, CPOTrainer

    config = CPOConfig(
        output_dir=args.output_dir,
        loss_type="simpo",
        cpo_alpha=args.cpo_alpha,
        simpo_gamma=args.simpo_gamma,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        bf16=True,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = CPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n=== SimPO Training ===")
    print(f"  Samples: {len(dataset)}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  SimPO gamma: {args.simpo_gamma}")
    print(f"  CPO alpha: {args.cpo_alpha}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n  Training complete: {elapsed / 60:.1f} min")

    # Save adapter
    trainer.save_model(args.output_dir)
    print(f"  Adapter saved: {args.output_dir}")

    return trainer


def train_dpo(model, tokenizer, dataset, lora_config, args):
    """Train with DPO (fallback #1)."""
    from trl import DPOConfig, DPOTrainer

    config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=0.1,
        num_train_epochs=args.num_epochs,
        bf16=True,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_completion_length,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n=== DPO Training (fallback) ===")
    print(f"  Samples: {len(dataset)}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n  Training complete: {elapsed / 60:.1f} min")
    trainer.save_model(args.output_dir)
    print(f"  Adapter saved: {args.output_dir}")

    return trainer


def train_orpo(model, tokenizer, dataset, lora_config, args):
    """Train with ORPO (fallback #2)."""
    try:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl import ORPOConfig, ORPOTrainer

    config = ORPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=0.1,
        num_train_epochs=args.num_epochs,
        bf16=True,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n=== ORPO Training (fallback) ===")
    print(f"  Samples: {len(dataset)}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n  Training complete: {elapsed / 60:.1f} min")
    trainer.save_model(args.output_dir)
    print(f"  Adapter saved: {args.output_dir}")

    return trainer


def train_sft(model, tokenizer, data_file, lora_config, args):
    """Train with SFT only (last resort fallback)."""
    from trl import SFTConfig, SFTTrainer

    dataset = load_sft_dataset(data_file, max_samples=None)

    config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate * 10,  # SFT uses higher LR
        num_train_epochs=args.num_epochs,
        bf16=True,
        max_seq_length=args.max_length,
        optim="adamw_torch",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n=== SFT Training (last resort) ===")
    print(f"  Samples: {len(dataset)}")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    print(f"\n  Training complete: {elapsed / 60:.1f} min")
    trainer.save_model(args.output_dir)
    print(f"  Adapter saved: {args.output_dir}")

    return trainer


def cmd_train(args):
    """Run training with specified method."""
    data_file = args.data_file
    method = args.method

    print("=" * 60)
    print(f"N5 Training: {method.upper()}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    lora_config = get_lora_config(args.lora_r, args.lora_alpha)

    print(f"  LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")

    if method == "sft":
        train_sft(model, tokenizer, data_file, lora_config, args)
    else:
        dataset = load_preference_dataset(data_file)
        if method == "simpo":
            train_simpo(model, tokenizer, dataset, lora_config, args)
        elif method == "dpo":
            train_dpo(model, tokenizer, dataset, lora_config, args)
        elif method == "orpo":
            train_orpo(model, tokenizer, dataset, lora_config, args)

    # Report memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n  Peak GPU memory: {peak_mem:.1f} GB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="N5 SimPO/DPO Training for Constitutional AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify (Phase 0)
    verify = subparsers.add_parser("verify", help="Phase 0: compatibility check")
    verify.add_argument("--data-file", required=True, help="Training data JSONL")
    verify.add_argument("--output-dir", default="./data/n5/outputs", help="Output dir")
    verify.set_defaults(func=cmd_verify)

    # train
    train = subparsers.add_parser("train", help="Run training")
    train.add_argument("--data-file", required=True, help="Training data JSONL")
    train.add_argument("--output-dir", default="./data/n5/adapter", help="Output dir")
    train.add_argument(
        "--method",
        default="simpo",
        choices=["simpo", "dpo", "orpo", "sft"],
        help="Training method (default: simpo)",
    )
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--num-epochs", type=int, default=1)
    train.add_argument("--max-length", type=int, default=2048)
    train.add_argument("--max-completion-length", type=int, default=1024)
    train.add_argument("--lora-r", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--simpo-gamma", type=float, default=0.5)
    train.add_argument("--cpo-alpha", type=float, default=0.5)
    train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
