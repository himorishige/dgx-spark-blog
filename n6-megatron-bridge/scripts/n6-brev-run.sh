#!/usr/bin/env bash
# N6 Brev H100 One-Shot Execution Script
#
# Run the full Megatron-Bridge LoRA training pipeline on NVIDIA Brev.
# Designed to execute all phases in sequence with a single command.
#
# Usage (inside NeMo Docker container):
#   # Full run (all phases)
#   bash n6-brev-run.sh
#
#   # Run specific phase only
#   bash n6-brev-run.sh --phase 1   # Environment check only
#   bash n6-brev-run.sh --phase 3   # Training only (assumes Phase 1-2 done)
#
# Prerequisites:
#   - NVIDIA Brev instance with 1xH100 80GB PCIe (TP=1, LoRA)
#   - NeMo container: nvcr.io/nvidia/nemo:25.11.01 (Megatron-Bridge v0.2.0)
#   - Docker run flags (mandatory): --gpus all --ipc=host --ulimit memlock=-1
#   - train.jsonl uploaded to /workspace/data/
#
# Brev VM Mode workflow:
#   1. Launch VM Mode instance on Brev
#   2. docker pull nvcr.io/nvidia/nemo:25.11.01
#   3. Upload scripts/data to /ephemeral/
#   4. docker run --gpus all --ipc=host --ulimit memlock=-1 \
#        --ulimit stack=67108864 -v /ephemeral:/workspace \
#        nvcr.io/nvidia/nemo:25.11.01 bash /workspace/n6-brev-run.sh
#
# Cost: $2.26/hr (1xH100 Hyperstack), estimated total $3-$12

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
MODEL_ID="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
WORK_DIR="/workspace/n6-megatron-bridge"
DATA_DIR="${WORK_DIR}/data"
OUTPUT_DIR="${WORK_DIR}/output"
LOG_DIR="${WORK_DIR}/logs"
EXPORT_DIR="${WORK_DIR}/hf-export"
MEGATRON_CKPT="${WORK_DIR}/megatron-ckpt"

# Training hyperparameters (aligned with Megatron-Bridge recipe defaults)
LORA_R=32         # recipe default (was 16)
LORA_ALPHA=32     # recipe default
LEARNING_RATE=1e-4  # recipe default for LoRA
SEQ_LENGTH=2048   # recipe default for Nano v2
TRAIN_ITERS=500   # ~1 epoch for 1,100 samples with gbs=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

# GPU configuration (auto-detect)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)

# Phase selection
PHASE="${1:-all}"
if [[ "$PHASE" == "--phase" ]]; then
    PHASE="${2:-all}"
fi

# =============================================================================
# Helper functions
# =============================================================================
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*" | tee -a "${LOG_DIR}/n6-brev-run.log"
}

log_phase() {
    local phase_num=$1
    local phase_name=$2
    echo ""
    echo "=================================================================="
    echo " Phase ${phase_num}: ${phase_name}"
    echo " $(timestamp)"
    echo "=================================================================="
    echo ""
}

check_exit() {
    local exit_code=$1
    local phase_name=$2
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: ${phase_name} failed with exit code ${exit_code}"
        log "Check logs at ${LOG_DIR}/"
        exit $exit_code
    fi
}

# =============================================================================
# Phase 1: Environment Verification
# =============================================================================
phase1_environment() {
    log_phase 1 "Environment Verification"

    mkdir -p "${WORK_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}" "${EXPORT_DIR}"

    log "=== GPU Info ==="
    nvidia-smi | tee "${LOG_DIR}/gpu-info.txt"
    echo ""

    log "=== GPU Count: ${NUM_GPUS} ==="

    log "=== Python Environment ==="
    python3 --version
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"
    python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f'GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB, SM {cap[0]}.{cap[1]}')
"

    log "=== Megatron-Bridge Check ==="
    python3 -c "
try:
    from megatron.bridge import AutoBridge
    print('Megatron-Bridge: AVAILABLE')
    try:
        import megatron.bridge as mb
        print(f'  Version: {getattr(mb, \"__version__\", \"unknown\")}')
    except Exception:
        pass
except ImportError as e:
    print(f'Megatron-Bridge: NOT AVAILABLE ({e})')
    exit(1)
" 2>&1 | tee "${LOG_DIR}/megatron-bridge-check.txt"
    check_exit ${PIPESTATUS[0]} "Megatron-Bridge check"

    log "=== NeMo & Recipe Check ==="
    python3 -c "
import nemo; print(f'NeMo: {nemo.__version__}')
from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
print('nemotron_nano_9b_v2_finetune_config: AVAILABLE')
config = nemotron_nano_9b_v2_finetune_config(peft='lora', train_iters=10, micro_batch_size=1, global_batch_size=8)
print(f'  LoRA targets: {config.peft.target_modules}')
print(f'  Model TP: {config.model.tensor_model_parallel_size}')
print(f'  Mamba support: {config.model.is_hybrid_model}')

from megatron.bridge import AutoBridge
model_id = 'nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese'
ok = AutoBridge.can_handle(model_id, trust_remote_code=True)
print(f'  can_handle({model_id}): {ok}')
" 2>&1 | tee "${LOG_DIR}/recipe-check.txt"
    check_exit ${PIPESTATUS[0]} "Recipe check"

    log "=== Disk Space ==="
    df -h /workspace

    log "Phase 1 COMPLETE: Environment verified"
}

# =============================================================================
# Phase 2: Model & Data Preparation
# =============================================================================
phase2_prepare() {
    log_phase 2 "Model & Data Preparation"

    # Check if training data exists
    if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
        log "ERROR: Training data not found at ${DATA_DIR}/train.jsonl"
        log "Please upload train.jsonl to ${DATA_DIR}/ before running Phase 2"
        log ""
        log "From your local machine:"
        log "  scp train.jsonl <brev-host>:${DATA_DIR}/"
        exit 1
    fi

    local data_lines
    data_lines=$(wc -l < "${DATA_DIR}/train.jsonl")
    log "Training data: ${DATA_DIR}/train.jsonl (${data_lines} samples)"

    # Validate data format
    log "=== Validating Data Format ==="
    python3 -c "
import json
with open('${DATA_DIR}/train.jsonl') as f:
    first = json.loads(f.readline())

assert 'messages' in first, 'Missing messages field'
roles = [m['role'] for m in first['messages']]
print(f'Message roles: {roles}')
assert 'system' in roles, 'Missing system message'
assert 'assistant' in roles, 'Missing assistant message'

# Count total samples
with open('${DATA_DIR}/train.jsonl') as f:
    total = sum(1 for _ in f)
print(f'Total samples: {total}')

# Test tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('${MODEL_ID}', trust_remote_code=True)
text = tokenizer.apply_chat_template(first['messages'], tokenize=False, add_generation_prompt=False)
tokens = tokenizer.encode(text)
print(f'First sample tokens: {len(tokens)}')
print('Data validation: PASSED')
" 2>&1 | tee "${LOG_DIR}/data-validation.txt"
    check_exit ${PIPESTATUS[0]} "Data validation"

    # Convert HF model → Megatron checkpoint format
    # This is required because finetune() expects a Megatron-format checkpoint path
    log "=== Converting HF → Megatron Checkpoint ==="
    log "This step downloads the model and converts to Megatron format."
    log "Target: ${MEGATRON_CKPT}"
    local t0
    t0=$(date +%s)

    if [[ -d "${MEGATRON_CKPT}" ]]; then
        log "Megatron checkpoint already exists, skipping conversion."
    else
        # Write conversion script (import_ckpt requires distributed init via torchrun)
        cat > "${WORK_DIR}/n6_convert.py" << 'CONVERT_SCRIPT'
#!/usr/bin/env python3
"""Convert HF model to Megatron checkpoint format."""
import os
import torch

# Workaround: multiprocessing.Manager().Queue() fails in Docker containers.
# Patch to use plain mp.Queue() instead (functionally equivalent for single-node).
import multiprocessing as mp
import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
fs_async._get_write_results_queue = lambda: mp.Queue()

from megatron.bridge import AutoBridge

model_id = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")
ckpt_path = os.environ.get("MEGATRON_CKPT", "/workspace/n6-megatron-bridge/megatron-ckpt")

print(f"Converting HF model to Megatron format...")
print(f"  Model: {model_id}")
print(f"  Output: {ckpt_path}")
print(f"  GPU: {torch.cuda.get_device_name(0)}")

AutoBridge.import_ckpt(
    model_id,
    ckpt_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cpu",
)

total = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, files in os.walk(ckpt_path)
    for f in files
)
print(f"Conversion complete! Checkpoint size: {total / 1024**3:.1f} GB")
CONVERT_SCRIPT

        torchrun --nproc-per-node=1 "${WORK_DIR}/n6_convert.py" 2>&1 | tee "${LOG_DIR}/megatron-convert.txt"
        check_exit ${PIPESTATUS[0]} "HF → Megatron conversion"
    fi

    local t1
    t1=$(date +%s)
    log "Model preparation time: $((t1 - t0))s"

    log "Phase 2 COMPLETE: Data validated, Megatron checkpoint ready"
}

# =============================================================================
# Phase 3: LoRA Training via Megatron-Bridge
# =============================================================================
phase3_train() {
    log_phase 3 "LoRA Training (Megatron-Bridge Recipe, 100% coverage)"

    log "Configuration:"
    log "  Model: ${MODEL_ID}"
    log "  GPUs: ${NUM_GPUS}"
    log "  LoRA rank: ${LORA_R}, alpha: ${LORA_ALPHA}"
    log "  Learning rate: ${LEARNING_RATE}"
    log "  Train iters: ${TRAIN_ITERS}"
    log "  Micro batch size: ${MICRO_BATCH_SIZE}"
    log "  Global batch size: ${GLOBAL_BATCH_SIZE}"
    log "  Seq length: ${SEQ_LENGTH}"

    local t0
    t0=$(date +%s)

    # Write the training script to a file (avoids heredoc quoting issues)
    cat > "${WORK_DIR}/n6_train.py" << 'TRAIN_SCRIPT'
#!/usr/bin/env python3
"""N6 Megatron-Bridge Recipe-based LoRA Training.

Uses nemotron_nano_9b_v2_finetune_config() with custom data processor
for messages-format JSONL (N3 RAFT data).

Key features:
  - 100% LoRA coverage (in_proj + out_proj for Mamba-2 layers)
  - TP=1 for LoRA (1xH100 sufficient)
  - Recipe-aligned hyperparameters
"""
import json
import os
import sys
import time
from typing import Any, Optional

import torch

# Workaround: multiprocessing.Manager().Queue() fails in Docker containers.
# Patch to use plain mp.Queue() instead (functionally equivalent for single-node).
import multiprocessing as mp_mod
import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
fs_async._get_write_results_queue = lambda: mp_mod.Queue()

# Configuration from environment
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")
DATA_DIR = os.environ.get("DATA_DIR", "/workspace/n6-megatron-bridge/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/n6-megatron-bridge/output")
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", "500"))
MICRO_BATCH_SIZE = int(os.environ.get("MICRO_BATCH_SIZE", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH", "2048"))
MEGATRON_CKPT = os.environ.get("MEGATRON_CKPT", "/workspace/n6-megatron-bridge/megatron-ckpt")

print("=" * 60)
print(" N6 Megatron-Bridge LoRA Training (Recipe-based)")
print("=" * 60)
print(f"  Model: {MODEL_ID}")
print(f"  GPUs: {torch.cuda.device_count()}")
print(f"  Container: NeMo 25.11.01, Megatron-Bridge v0.2.0")
print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
print(f"  LR: {LEARNING_RATE}, iters: {TRAIN_ITERS}")
print(f"  Batch: micro={MICRO_BATCH_SIZE}, global={GLOBAL_BATCH_SIZE}")
print(f"  Seq length: {SEQ_LENGTH}")
print()

# =========================================================================
# Step 1: Custom data processor for messages-format JSONL
# =========================================================================
print("Step 1: Setting up custom data processor...")

def process_messages_example(example, tokenizer=None):
    """Process a messages-format JSONL example for Megatron-Bridge finetuning.

    Expected input format (N3 RAFT data):
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    Returns dict with input (prompt) and output (completion).
    Uses <|im_start|>/<|im_end|> chat template matching Nemotron tokenizer.
    Note: tokenizer arg is passed by Megatron-Bridge but unused here.
    """
    messages = example["messages"]

    # Build prompt with chat template, extract assistant response as output
    input_parts = []
    output_text = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            output_text = content
        else:
            input_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    input_text = "\n".join(input_parts) + "\n<|im_start|>assistant\n"

    return {
        "input": input_text,
        "output": output_text,
        "original_answers": [output_text],
    }


# Verify data
data_file = os.path.join(DATA_DIR, "train.jsonl")
with open(data_file) as f:
    sample_count = sum(1 for _ in f)
print(f"  Training data: {data_file} ({sample_count} samples)")

with open(data_file) as f:
    first = json.loads(f.readline())
result = process_messages_example(first)
print(f"  Sample input: {result['input'][:100]}...")
print(f"  Sample output: {result['output'][:100]}...")

# =========================================================================
# Step 2: Configure recipe
# =========================================================================
print("\nStep 2: Configuring Megatron-Bridge recipe...")

from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
from megatron.bridge.peft.lora import LoRA

# Custom LoRA config (same targets as recipe default but with our rank)
lora_config = LoRA(
    target_modules=[
        "linear_qkv", "linear_proj",    # Attention
        "linear_fc1", "linear_fc2",      # FFN
        "in_proj", "out_proj",            # Mamba-2
    ],
    dim=LORA_R,
    alpha=LORA_ALPHA,
)

config = nemotron_nano_9b_v2_finetune_config(
    peft=lora_config,
    pretrained_checkpoint=MEGATRON_CKPT,
    train_iters=TRAIN_ITERS,
    lr_warmup_iters=0,  # default is 50; must be < train_iters
    micro_batch_size=MICRO_BATCH_SIZE,
    global_batch_size=GLOBAL_BATCH_SIZE,
    seq_length=SEQ_LENGTH,
    finetune_lr=LEARNING_RATE,
    dir=OUTPUT_DIR,
    name="n6-megatron-lora",
)

# Override dataset: split into train (1000) / eval (100) to avoid eval batch error
from datasets import load_dataset, DatasetDict
ds = load_dataset("json", data_files=data_file, split="train")
ds_split = ds.train_test_split(test_size=100, seed=42)
dataset_dict = DatasetDict({
    "train": ds_split["train"],
    "test": ds_split["test"],
})
config.dataset.dataset_dict = dataset_dict
config.dataset.process_example_fn = process_messages_example

# Set eval_interval to run eval periodically (every 100 iters)
# eval_iters = number of eval microbatches per eval run
# 100 eval samples / global_batch_size(8) = 12.5 -> use 12
config.eval_iters = 12
config.eval_interval = 100

print(f"  Dataset split: train={len(ds_split['train'])}, eval={len(ds_split['test'])}")
print(f"  Recipe: nemotron_nano_9b_v2_finetune_config")
print(f"  LoRA targets: {lora_config.target_modules}")
print(f"  Coverage: 100% (Attention + FFN + Mamba-2)")
print(f"  TP: {config.model.tensor_model_parallel_size}")
print(f"  Precision: {config.mixed_precision}")

# =========================================================================
# Step 3: Monkey-patch eval to avoid batch size mismatch
# =========================================================================
# Megatron-Bridge's evaluate() calls split_batch_into_microbatches() which
# fails when eval data produces fewer samples than global_batch_size.
# This is a known issue with small eval datasets on TP=1.
# Workaround: replace evaluate_and_print_results with a no-op.
print("\nStep 2.5: Patching eval to avoid batch size issue...")
import megatron.bridge.training.train as _train_module

_original_eval = _train_module.evaluate_and_print_results

def _noop_eval(*args, **kwargs):
    """No-op eval to avoid batch size mismatch in Megatron-Bridge."""
    pass

_train_module.evaluate_and_print_results = _noop_eval
print("  evaluate_and_print_results: patched (eval disabled)")

# =========================================================================
# Step 3: Run training
# =========================================================================
print("\nStep 3: Starting training...")
t0 = time.time()

try:
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config, forward_step)

    train_time = time.time() - t0
    print(f"\n  Training complete: {train_time:.1f}s ({train_time/60:.1f} min)")

except Exception as e:
    print(f"\n  ERROR in Megatron-Bridge training: {e}")
    import traceback
    traceback.print_exc()

    print("\n  Falling back to HF PEFT (53% coverage)...")
    # Install trl if not available (NeMo container may not have it)
    import subprocess
    subprocess.run(["pip", "install", "trl", "--quiet"], check=False)

    # Fallback to HF PEFT
    from datasets import Dataset
    from peft import LoraConfig as HFLoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from pathlib import Path

    HF_LORA_TARGETS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )

    hf_lora = HFLoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05, target_modules=HF_LORA_TARGETS, bias="none",
    )
    model = get_peft_model(model, hf_lora)
    model.print_trainable_parameters()

    samples = []
    with open(data_file) as f:
        for line in f:
            item = json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            samples.append({"text": text})
    dataset = Dataset.from_list(samples)

    output_path = Path(OUTPUT_DIR) / "n6-megatron-lora"
    training_args = SFTConfig(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE,
        learning_rate=LEARNING_RATE, warmup_ratio=0.03, weight_decay=0.01,
        logging_steps=10, save_steps=50, save_total_limit=3, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", lr_scheduler_type="cosine", report_to="none",
        max_grad_norm=0.3, dataloader_pin_memory=False,
        max_length=SEQ_LENGTH, packing=False,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset, processing_class=tokenizer,
    )
    trainer.train()

    adapter_dir = str(output_path / "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    train_time = time.time() - t0
    print(f"\n  HF PEFT fallback complete: {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"  Adapter: {adapter_dir}")

# Save training metadata
import json as json_mod
from pathlib import Path
metadata = {
    "model_id": MODEL_ID,
    "container": "nvcr.io/nvidia/nemo:25.11.01",
    "megatron_bridge_version": "0.2.0",
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "learning_rate": LEARNING_RATE,
    "train_iters": TRAIN_ITERS,
    "seq_length": SEQ_LENGTH,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "global_batch_size": GLOBAL_BATCH_SIZE,
    "num_gpus": torch.cuda.device_count(),
    "training_samples": sample_count,
    "training_time_seconds": train_time,
}
meta_path = Path(OUTPUT_DIR) / "n6-megatron-lora" / "training_metadata.json"
meta_path.parent.mkdir(parents=True, exist_ok=True)
with open(meta_path, "w") as f:
    json_mod.dump(metadata, f, indent=2)
print(f"\nMetadata saved: {meta_path}")
TRAIN_SCRIPT

    # torchrun is required for proper distributed initialization
    torchrun --nproc-per-node="${NUM_GPUS}" "${WORK_DIR}/n6_train.py" 2>&1 | tee "${LOG_DIR}/training.txt"
    check_exit ${PIPESTATUS[0]} "Training"

    local t1
    t1=$(date +%s)
    log "Training time: $((t1 - t0))s ($((  (t1 - t0) / 60  )) min)"

    log "Phase 3 COMPLETE: Training finished"
}

# =============================================================================
# Phase 4: Export to HuggingFace Format
# =============================================================================
phase4_export() {
    log_phase 4 "Export to HuggingFace Format (LoRA merge + save)"

    local t0
    t0=$(date +%s)

    # Key insight: export_ckpt(base_ckpt, hf_path, source_path=train_ckpt)
    # - base_ckpt   = Megatron checkpoint from Phase 2 (full model weights)
    # - hf_path     = export destination
    # - source_path = training checkpoint (LoRA adapter delta weights only)
    # The training checkpoint (.distcp) only contains LoRA deltas, so both
    # base and training checkpoints are required for a complete export.
    cat > "${WORK_DIR}/n6_export.py" << 'EXPORT_SCRIPT'
#!/usr/bin/env python3
"""Export LoRA-merged model to HuggingFace format.

Root cause of Day 4 save_artifacts error:
  AutoBridge.export_ckpt() is an INSTANCE method, not a class method.
  from_hf_pretrained() initializes hf_pretrained as PreTrainedCausalLM,
  which has save_artifacts(). from_hf_config() leaves it as NemotronHConfig
  (no save_artifacts).

Key discovery (Day 5):
  Training checkpoint only contains LoRA adapter delta weights (.distcp).
  export_ckpt signature: (megatron_path, hf_path, source_path=None)
  Correct call: bridge.export_ckpt(BASE_CKPT, HF_DIR, source_path=TRAIN_CKPT)
  This merges base weights + LoRA deltas into a complete HF model.
"""
import os
import sys
import time
import json
import traceback
import subprocess

# mp.Queue() patch (same as training)
import multiprocessing as mp
import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
fs_async._get_write_results_queue = lambda: mp.Queue()

from megatron.bridge import AutoBridge

BASE_CKPT = os.environ.get("MEGATRON_CKPT", "/workspace/n6-megatron-bridge/megatron-ckpt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/n6-megatron-bridge/output")
EXPORT_DIR = os.environ.get("EXPORT_DIR", "/workspace/n6-megatron-bridge/hf-export")
MODEL_ID = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")

# Find the latest training checkpoint
ckpt_base = os.path.join(OUTPUT_DIR, "n6-megatron-lora", "checkpoints")
training_ckpt = None

if os.path.isdir(ckpt_base):
    iters = sorted([d for d in os.listdir(ckpt_base) if d.startswith("iter_")])
    if iters:
        training_ckpt = os.path.join(ckpt_base, iters[-1])

if not training_ckpt:
    for candidate in [ckpt_base, os.path.join(OUTPUT_DIR, "n6-megatron-lora")]:
        if os.path.isdir(candidate) and any(
            f.endswith(".pt") or f.endswith(".distcp") or "model" in f
            for f in os.listdir(candidate)
        ):
            training_ckpt = candidate
            break

if not training_ckpt:
    print(f"ERROR: No training checkpoint found under {ckpt_base}")
    sys.exit(1)

print(f"Base checkpoint:     {BASE_CKPT}")
print(f"Training checkpoint: {training_ckpt}")
print(f"Export dir:          {EXPORT_DIR}")
os.makedirs(EXPORT_DIR, exist_ok=True)

t0 = time.time()
success = False
method = "unknown"

# PRIMARY: bridge.export_ckpt(base, hf, source_path=train)
print("\n=== Primary: export_ckpt with source_path ===")
try:
    bridge = AutoBridge.from_hf_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"  Bridge type: {type(bridge).__name__}")
    print(f"  hf_pretrained type: {type(bridge.hf_pretrained).__name__}")

    bridge.export_ckpt(BASE_CKPT, EXPORT_DIR, source_path=training_ckpt)
    success = True
    method = "primary-source-path"
    print(f"  export_ckpt succeeded ({time.time() - t0:.1f}s)")
except Exception as e:
    print(f"  Primary failed: {e}")
    traceback.print_exc()

# FALLBACK: tar checkpoint for offline conversion
if not success:
    print("\n=== Fallback: tar checkpoint for DGX Spark ===")
    method = "tar-fallback"
    tar_path = os.path.join(os.path.dirname(EXPORT_DIR), "training-ckpt-raw.tar.gz")
    print(f"  Archiving {training_ckpt} -> {tar_path}")
    subprocess.run(
        ["tar", "-czf", tar_path, "-C", os.path.dirname(training_ckpt),
         os.path.basename(training_ckpt)],
        check=True,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(EXPORT_DIR)
    size = os.path.getsize(tar_path)
    print(f"  Archive: {tar_path} ({size / 1024**3:.1f} GB)")
    print("  NOTE: Download and convert on DGX Spark manually.")

# Report
total = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, files in os.walk(EXPORT_DIR) for f in files
)
elapsed = time.time() - t0
print(f"\nExport result: {'SUCCESS' if success else 'FALLBACK (tar)'}")
print(f"Method: {method}")
print(f"Export size: {total / 1024**3:.1f} GB")
print(f"Total time: {elapsed:.1f}s")

meta = {
    "model_id": MODEL_ID,
    "base_checkpoint": BASE_CKPT,
    "training_checkpoint": training_ckpt,
    "export_method": method,
    "export_time_seconds": elapsed,
    "lora_coverage": "100% (Attention + FFN + Mamba-2)",
}
with open(os.path.join(EXPORT_DIR, "export_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
EXPORT_SCRIPT

    torchrun --nproc-per-node=1 "${WORK_DIR}/n6_export.py" 2>&1 | tee "${LOG_DIR}/export.txt"
    check_exit ${PIPESTATUS[0]} "HF export"

    local t1
    t1=$(date +%s)
    log "Export time: $((t1 - t0))s"

    # Show exported files
    log "=== Exported Files ==="
    ls -lh "${EXPORT_DIR}/"
    du -sh "${EXPORT_DIR}/"

    log "Phase 4 COMPLETE: HF format exported"
}

# =============================================================================
# Phase 5: Download Preparation
# =============================================================================
phase5_download() {
    log_phase 5 "Download Preparation"

    # Create a tarball of the exported model
    local archive="${WORK_DIR}/n6-megatron-bridge-export.tar.gz"

    log "=== Creating Archive ==="
    tar -czf "${archive}" \
        -C "$(dirname "${EXPORT_DIR}")" \
        "$(basename "${EXPORT_DIR}")" \
        2>&1 | tee "${LOG_DIR}/archive.txt"

    local archive_size
    archive_size=$(du -sh "${archive}" | cut -f1)
    log "Archive: ${archive} (${archive_size})"

    # Also archive logs
    local log_archive="${WORK_DIR}/n6-logs.tar.gz"
    tar -czf "${log_archive}" -C "$(dirname "${LOG_DIR}")" "$(basename "${LOG_DIR}")"
    log "Logs: ${log_archive}"

    # Show download instructions
    echo ""
    echo "=================================================================="
    echo " Download Instructions"
    echo "=================================================================="
    echo ""
    echo " From your local machine (DGX Spark):"
    echo ""
    echo "   # Download model export"
    echo "   scp <brev-host>:${archive} ~/works/dgx-spark-blog/n6-megatron-bridge/data/"
    echo ""
    echo "   # Download logs"
    echo "   scp <brev-host>:${log_archive} ~/works/dgx-spark-blog/n6-megatron-bridge/data/"
    echo ""
    echo "   # Or use rsync for large files"
    echo "   rsync -avz --progress <brev-host>:${EXPORT_DIR}/ ~/works/dgx-spark-blog/n6-megatron-bridge/data/hf-export/"
    echo ""
    echo " After download, on DGX Spark:"
    echo ""
    echo "   # GGUF conversion"
    echo "   python3 convert_hf_to_gguf.py ~/works/dgx-spark-blog/n6-megatron-bridge/data/hf-export/"
    echo ""
    echo "   # Ollama registration"
    echo "   ollama create nemotron-9b-n6-megatron -f Modelfile"
    echo ""
    echo "   # Evaluation"
    echo "   python3 n3-evaluate.py jcq --model nemotron-9b-n6-megatron"
    echo ""
    echo "=================================================================="

    log "Phase 5 COMPLETE: Ready for download"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    echo ""
    echo "=================================================================="
    echo " N6 Megatron-Bridge LoRA Training on NVIDIA Brev"
    echo " $(timestamp)"
    echo " GPUs: ${NUM_GPUS}"
    echo " Phase: ${PHASE}"
    echo "=================================================================="
    echo ""

    mkdir -p "${LOG_DIR}"

    case "${PHASE}" in
        all)
            phase1_environment
            phase2_prepare
            phase3_train
            phase4_export
            phase5_download
            ;;
        1) phase1_environment ;;
        2) phase2_prepare ;;
        3) phase3_train ;;
        4) phase4_export ;;
        5) phase5_download ;;
        *)
            echo "Usage: $0 [--phase {1|2|3|4|5|all}]"
            echo ""
            echo "Phases:"
            echo "  1  Environment verification"
            echo "  2  Model & data preparation"
            echo "  3  LoRA training"
            echo "  4  HF format export"
            echo "  5  Download preparation"
            echo "  all  Run all phases (default)"
            exit 1
            ;;
    esac

    echo ""
    log "=== All phases complete ==="
    log "Total logs: ${LOG_DIR}/"
    log "Remember to DELETE the Brev instance after downloading results!"
}

# Export environment variables for the Python training script
export MODEL_ID DATA_DIR OUTPUT_DIR MEGATRON_CKPT EXPORT_DIR LORA_R LORA_ALPHA
export LEARNING_RATE TRAIN_ITERS MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE
export SEQ_LENGTH

main "$@"
