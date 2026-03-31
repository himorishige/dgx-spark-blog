"""LoRA SFT for Cosmos-Reason2-8B on SH17 PPE dataset (DGX Spark).

Based on the official trl_sft.py example, modified for:
- BF16 LoRA (no quantization, no bitsandbytes dependency)
- Cosmos-Reason2-8B (not 2B)
- Local SH17 PPE dataset
- DGX Spark (GB10, 128GB unified memory, sm_12.1)

Usage:
    uv run python train_lora_sft.py
    uv run python train_lora_sft.py --max-steps 10  # smoke test
    uv run python train_lora_sft.py --num-epochs 1  # full training
"""

import argparse
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for Cosmos-Reason2-8B")
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/Cosmos-Reason2-8B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./data/sh17_vlm",
        help="Path to converted SH17 VLM dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/Cosmos-Reason2-8B-ppe-lora",
        help="Output directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max training steps (-1 for full epochs)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (ignored if max-steps > 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to save memory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Cosmos-Reason2-8B LoRA SFT on SH17 PPE")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"LR: {args.lr}")
    print(f"Max steps: {args.max_steps if args.max_steps > 0 else 'full epochs'}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    train_dataset = dataset["train"]
    print(f"Train samples: {len(train_dataset)}")

    # Load model in BF16 (no quantization)
    print("Loading model in BF16...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # Use SDPA instead of flash-attn (no aarch64 wheel)
    )

    # Show memory after model load
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_reserved() / 1e9
        print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # LoRA config (LLM layers only, vision encoder frozen)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        learning_rate=args.lr,
        optim="adamw_torch",  # Standard AdamW (no 8bit, avoid bitsandbytes)
        bf16=True,
        max_length=None,  # Don't truncate (image tokens)
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        report_to="tensorboard",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,  # Unified memory: pinning is unnecessary
        remove_unused_columns=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # Memory stats before training
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        start_mem = torch.cuda.max_memory_reserved() / 1e9
        total_mem = props.total_memory / 1e9
        print(f"\nGPU: {props.name}")
        print(f"Total memory: {total_mem:.1f} GB")
        print(f"Reserved before training: {start_mem:.1f} GB")

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    # Memory stats after training
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nPeak GPU memory: {peak_mem:.1f} GB")
        print(f"Training memory: {peak_mem - start_mem:.1f} GB")
        print(f"Memory utilization: {peak_mem / total_mem * 100:.1f}%")

    # Print training stats
    print(f"\nTraining time: {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime'] / 60:.1f} min)")
    print(f"Train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    # Save model
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
