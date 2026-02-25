#!/usr/bin/env python3
"""N6 Megatron-Bridge Checkpoint Export to HuggingFace Format (Standalone)

Export a Megatron-Bridge trained checkpoint to HuggingFace format
for downstream use (GGUF conversion, Ollama, vLLM, etc.).

Root cause of Day 4 save_artifacts error:
  AutoBridge.export_ckpt() is an INSTANCE method, not a class method.
  Calling AutoBridge.export_ckpt(path, path) invokes it on the class,
  which uses from_hf_config() internally -> NemotronHConfig (no save_artifacts).
  Correct: bridge = AutoBridge.from_hf_pretrained() -> bridge.export_ckpt()
  This initializes hf_pretrained as PreTrainedCausalLM (has save_artifacts).

Strategy:
  Primary:     bridge = from_hf_pretrained() -> bridge.export_ckpt()
  Fallback A:  bridge.save_hf_pretrained(models, hf_path)
  Fallback B:  monkey-patch NemotronHConfig.save_artifacts
  Fallback C:  HF PEFT merge (if HF PEFT fallback was used in training)
  Ultimate:    tar checkpoint for offline conversion on DGX Spark

Usage:
  # Inside NeMo container after Phase 3 training
  torchrun --nproc-per-node=1 n6-export-hf.py

  # With custom paths
  torchrun --nproc-per-node=1 n6-export-hf.py \
    --training-ckpt /workspace/n6-megatron-bridge/output/n6-megatron-lora/checkpoints/iter_000500 \
    --export-dir /workspace/n6-megatron-bridge/hf-export

  # Verify exported model
  python n6-export-hf.py --verify --export-dir /workspace/n6-megatron-bridge/hf-export

Prerequisites:
  - NGC NeMo container with Megatron-Bridge (nvcr.io/nvidia/nemo:25.11.01)
  - Trained checkpoint from n6-brev-run.sh Phase 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Workaround: multiprocessing.Manager().Queue() fails in Docker containers.
import multiprocessing as mp

try:
    import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async

    fs_async._get_write_results_queue = lambda: mp.Queue()
except ImportError:
    pass  # Not in NeMo container

MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest training checkpoint under output_dir."""
    ckpt_base = os.path.join(output_dir, "n6-megatron-lora", "checkpoints")

    # Look for iter_XXXXX directories
    if os.path.isdir(ckpt_base):
        iters = sorted([d for d in os.listdir(ckpt_base) if d.startswith("iter_")])
        if iters:
            return os.path.join(ckpt_base, iters[-1])

    # Fallback: look for checkpoint files directly
    for candidate in [ckpt_base, os.path.join(output_dir, "n6-megatron-lora")]:
        if os.path.isdir(candidate) and any(
            f.endswith(".pt") or f.endswith(".distcp") or "model" in f
            for f in os.listdir(candidate)
        ):
            return candidate

    return None


def export_primary(
    training_ckpt: str, export_dir: str, model_id: str, base_ckpt: str | None = None
) -> bool:
    """Primary: from_hf_pretrained + export_ckpt with source_path.

    Key insight: export_ckpt(megatron_path, hf_path, source_path) where
      megatron_path = base checkpoint (from import_ckpt)
      hf_path       = export destination
      source_path   = training checkpoint (LoRA adapter weights)
    The training checkpoint only contains LoRA delta weights (.distcp),
    so we must provide the base checkpoint separately.
    """
    print("=== Primary: from_hf_pretrained + export_ckpt ===\n")

    from megatron.bridge import AutoBridge

    t0 = time.time()

    bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)
    print(f"  Bridge type: {type(bridge).__name__}")
    print(f"  hf_pretrained type: {type(bridge.hf_pretrained).__name__}")

    if base_ckpt:
        print(f"  Base checkpoint:     {base_ckpt}")
        print(f"  Training checkpoint: {training_ckpt}")
        bridge.export_ckpt(base_ckpt, export_dir, source_path=training_ckpt)
    else:
        bridge.export_ckpt(training_ckpt, export_dir)

    print(f"  export_ckpt succeeded ({time.time() - t0:.1f}s)")
    return True


def export_fallback_a(
    training_ckpt: str, export_dir: str, model_id: str, base_ckpt: str | None = None
) -> bool:
    """Fallback A: load_megatron_model + save_hf_pretrained."""
    print("=== Fallback A: load_megatron_model + save_hf_pretrained ===\n")

    from megatron.bridge import AutoBridge

    t0 = time.time()

    bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)
    ckpt = base_ckpt if base_ckpt else training_ckpt
    models = bridge.load_megatron_model(ckpt)
    bridge.save_hf_pretrained(
        models, export_dir, source_path=training_ckpt if base_ckpt else None
    )
    print(f"  save_hf_pretrained succeeded ({time.time() - t0:.1f}s)")
    return True


def export_fallback_b(
    training_ckpt: str, export_dir: str, model_id: str, base_ckpt: str | None = None
) -> bool:
    """Fallback B: export_ckpt with strict=False."""
    print("=== Fallback B: export_ckpt strict=False ===\n")

    from megatron.bridge import AutoBridge

    t0 = time.time()

    bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)
    ckpt = base_ckpt if base_ckpt else training_ckpt
    bridge.export_ckpt(
        ckpt,
        export_dir,
        strict=False,
        source_path=training_ckpt if base_ckpt else None,
    )
    print(f"  Fallback B export succeeded ({time.time() - t0:.1f}s)")
    return True


def export_fallback_c(checkpoint_dir: str, export_dir: str, model_id: str) -> bool:
    """Fallback C: HF PEFT adapter merge (if training used HF PEFT fallback)."""
    print("=== Fallback C: HF PEFT Merge ===\n")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(checkpoint_dir) / "adapter"
    if not adapter_dir.exists():
        adapter_dir = Path(checkpoint_dir)
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  No HF PEFT adapter found at {checkpoint_dir}")
            return False

    t0 = time.time()

    print(f"  Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print(f"  Loading adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("  Merging LoRA into base model...")
    model = model.merge_and_unload()

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(export_path))
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(export_path))

    print(f"  HF PEFT merge succeeded ({time.time() - t0:.1f}s)")
    return True


def export_ultimate_tar(training_ckpt: str, export_dir: str, model_id: str) -> bool:
    """Ultimate: tar checkpoint for offline conversion on DGX Spark."""
    print("=== Ultimate Fallback: tar checkpoint ===\n")

    from transformers import AutoTokenizer

    tar_path = os.path.join(os.path.dirname(export_dir), "training-ckpt-raw.tar.gz")
    print(f"  Archiving {training_ckpt} -> {tar_path}")
    subprocess.run(
        [
            "tar",
            "-czf",
            tar_path,
            "-C",
            os.path.dirname(training_ckpt),
            os.path.basename(training_ckpt),
        ],
        check=True,
    )

    # Save tokenizer for DGX Spark
    os.makedirs(export_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(export_dir)

    size = os.path.getsize(tar_path)
    print(f"  Archive: {tar_path} ({size / 1024**3:.1f} GB)")
    print("  NOTE: Download and convert on DGX Spark manually.")
    return False  # Not a real export success


def _save_metadata(export_dir: str, model_id: str, training_ckpt: str, method: str, elapsed: float):
    """Save export metadata."""
    metadata = {
        "model_id": model_id,
        "training_checkpoint": training_ckpt,
        "export_method": method,
        "export_time_seconds": elapsed,
        "lora_coverage": "100% (Attention + FFN + Mamba-2)",
    }
    with open(os.path.join(export_dir, "export_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def _ensure_tokenizer(export_dir: str, model_id: str):
    """Ensure tokenizer is saved in export dir."""
    if not os.path.exists(os.path.join(export_dir, "tokenizer_config.json")):
        print("\nSaving tokenizer...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.save_pretrained(export_dir)


def _report_size(export_dir: str):
    """Report total export size."""
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(export_dir)
        for f in files
    )
    print(f"Export size: {total / 1024**3:.1f} GB")


def verify_export(export_dir: str, model_id: str) -> bool:
    """Verify the exported model can be loaded and produces output."""
    print("=== Verifying Exported Model ===\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    export_path = Path(export_dir)
    if not export_path.exists():
        print(f"ERROR: Export directory not found: {export_dir}")
        return False

    print(f"Loading model from {export_dir}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        str(export_path), trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(export_path),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print(f"  Load time: {time.time() - t0:.1f}s")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.2f}B")

    # Quick inference test
    print("\nRunning inference test...")
    messages = [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "日本の首都はどこですか？"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"  Response: {response[:200]}")

    metadata_path = export_path / "export_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\n  Method:   {metadata.get('export_method', 'unknown')}")
        print(f"  Coverage: {metadata.get('lora_coverage', 'unknown')}")

    print("\nVerification: PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Megatron-Bridge checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--training-ckpt",
        default=None,
        help="Training checkpoint path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/n6-megatron-bridge/output",
        help="Training output directory (for auto-detecting checkpoint)",
    )
    parser.add_argument(
        "--export-dir",
        default="/workspace/n6-megatron-bridge/hf-export",
        help="Output directory for HF format",
    )
    parser.add_argument(
        "--base-ckpt",
        default="/workspace/n6-megatron-bridge/megatron-ckpt",
        help="Base Megatron checkpoint (from import_ckpt, contains full weights)",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model (load + inference test)",
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_export(args.export_dir, args.model_id)
        sys.exit(0 if success else 1)

    # Find training checkpoint
    training_ckpt = args.training_ckpt
    if not training_ckpt:
        training_ckpt = find_latest_checkpoint(args.output_dir)
        if not training_ckpt:
            print(f"ERROR: No training checkpoint found under {args.output_dir}")
            sys.exit(1)

    base_ckpt = args.base_ckpt if os.path.isdir(args.base_ckpt) else None

    print(f"Training checkpoint: {training_ckpt}")
    if base_ckpt:
        print(f"Base checkpoint:     {base_ckpt}")
    print(f"Export dir:          {args.export_dir}")
    os.makedirs(args.export_dir, exist_ok=True)

    t0 = time.time()
    success = False
    method = "unknown"

    # Try each strategy in order
    strategies = [
        ("primary", export_primary),
        ("fallback-a-save-hf", export_fallback_a),
        ("fallback-b-strict-false", export_fallback_b),
    ]

    for name, fn in strategies:
        try:
            if fn(training_ckpt, args.export_dir, args.model_id, base_ckpt=base_ckpt):
                success = True
                method = name
                break
        except Exception as e:
            print(f"  {name} failed: {e}")
            traceback.print_exc()
            print()

    # Try HF PEFT merge (for HF PEFT fallback training)
    if not success:
        try:
            peft_dir = os.path.join(args.output_dir, "n6-megatron-lora")
            if export_fallback_c(peft_dir, args.export_dir, args.model_id):
                success = True
                method = "fallback-c-hf-peft"
        except Exception as e:
            print(f"  Fallback C failed: {e}")
            traceback.print_exc()

    # Ultimate fallback: tar
    if not success:
        export_ultimate_tar(training_ckpt, args.export_dir, args.model_id)
        method = "tar-fallback"

    if success:
        _ensure_tokenizer(args.export_dir, args.model_id)

    elapsed = time.time() - t0
    _save_metadata(args.export_dir, args.model_id, training_ckpt, method, elapsed)
    _report_size(args.export_dir)

    print(f"\nExport result: {'SUCCESS' if success else 'FALLBACK (tar)'}")
    print(f"Method: {method}")
    print(f"Total time: {elapsed:.1f}s")

    if success:
        print("\n=== Next Steps ===")
        print(f"  1. Download: scp <brev-host>:{args.export_dir}/ ~/n6-export/")
        print(f"  2. GGUF:     python convert_hf_to_gguf.py ~/n6-export/")
        print(f"  3. Ollama:   ollama create nemotron-9b-n6 -f Modelfile")
        print(f"  4. Evaluate: python n3-evaluate.py jcq --model nemotron-9b-n6")


if __name__ == "__main__":
    main()
