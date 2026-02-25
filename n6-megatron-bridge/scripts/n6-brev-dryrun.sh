#!/usr/bin/env bash
# N6 Brev Day 4 Dry Run: Phase 1-2 only
#
# Minimal script to verify:
#   1. Environment (Megatron-Bridge, recipe, GPU)
#   2. HF → Megatron checkpoint conversion (import_ckpt)
#
# Usage (on Brev instance):
#   bash n6-brev-dryrun.sh
#
# Expected time: ~10-15 min (mostly model download + conversion)
# Expected cost: ~$0.50 (at $2.26/hr)
#
# Docker run (VM Mode):
#   docker run --gpus all --ipc=host --ulimit memlock=-1 \
#     --ulimit stack=67108864 -v /ephemeral:/workspace \
#     nvcr.io/nvidia/nemo:25.11.01 bash /workspace/n6-brev-dryrun.sh
#
# After verification, DELETE the instance immediately.

set -euo pipefail

MODEL_ID="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese"
WORK_DIR="/workspace/n6-dryrun"
MEGATRON_CKPT="${WORK_DIR}/megatron-ckpt"
DATA_DIR="${WORK_DIR}/data"
LOG_DIR="${WORK_DIR}/logs"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# =============================================================================
# Phase 1: Environment Verification
# =============================================================================
log "=========================================="
log " Phase 1: Environment Verification"
log "=========================================="

nvidia-smi

python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f'GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB, SM {cap[0]}.{cap[1]}')
"

python3 -c "
from megatron.bridge import AutoBridge
import megatron.bridge as mb
print(f'Megatron-Bridge: {getattr(mb, \"__version__\", \"unknown\")}')

from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
print('Recipe: nemotron_nano_9b_v2_finetune_config AVAILABLE')

ok = AutoBridge.can_handle('nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese', trust_remote_code=True)
print(f'can_handle: {ok}')

config = nemotron_nano_9b_v2_finetune_config(peft='lora', train_iters=3, micro_batch_size=1, global_batch_size=1)
print(f'LoRA targets: {config.peft.target_modules}')
print(f'TP: {config.model.tensor_model_parallel_size}')
"

log "Phase 1: PASSED"
echo ""

# =============================================================================
# Phase 2: Data Validation + HF → Megatron Conversion
# =============================================================================
log "=========================================="
log " Phase 2: Data + Checkpoint Conversion"
log "=========================================="

# Check data
if [[ -f "${DATA_DIR}/train.jsonl" ]]; then
    DATA_LINES=$(wc -l < "${DATA_DIR}/train.jsonl")
    log "Training data: ${DATA_LINES} samples"
else
    log "WARNING: train.jsonl not found at ${DATA_DIR}/train.jsonl"
    log "Skipping data validation (conversion test only)"
fi

# HF → Megatron conversion via torchrun
log "Converting HF → Megatron checkpoint..."
log "Model: ${MODEL_ID}"
log "Output: ${MEGATRON_CKPT}"

T0=$(date +%s)

cat > "${WORK_DIR}/convert.py" << 'PYEOF'
import os
import time
import torch

# Workaround: multiprocessing.Manager().Queue() fails in Docker containers.
import multiprocessing as mp
import megatron.core.dist_checkpointing.strategies.filesystem_async as fs_async
fs_async._get_write_results_queue = lambda: mp.Queue()

from megatron.bridge import AutoBridge

model_id = os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese")
ckpt_path = os.environ.get("MEGATRON_CKPT", "/workspace/n6-dryrun/megatron-ckpt")

print(f"Model: {model_id}")
print(f"Output: {ckpt_path}")
print(f"GPU: {torch.cuda.get_device_name(0)}, SM {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")

t0 = time.time()
AutoBridge.import_ckpt(
    model_id,
    ckpt_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cpu",
)
elapsed = time.time() - t0
print(f"\nConversion complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

total = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, files in os.walk(ckpt_path) for f in files
)
print(f"Checkpoint size: {total / 1024**3:.1f} GB")

# List top-level contents
for item in sorted(os.listdir(ckpt_path)):
    path = os.path.join(ckpt_path, item)
    if os.path.isdir(path):
        count = sum(1 for _, _, files in os.walk(path) for f in files)
        print(f"  {item}/ ({count} files)")
    else:
        size = os.path.getsize(path)
        print(f"  {item} ({size / 1024:.1f} KB)")
PYEOF

export MODEL_ID MEGATRON_CKPT
torchrun --nproc-per-node=1 "${WORK_DIR}/convert.py" 2>&1 | tee "${LOG_DIR}/convert.txt"

T1=$(date +%s)
log "Conversion time: $((T1 - T0))s"

# Verify checkpoint exists
if [[ -d "${MEGATRON_CKPT}" ]]; then
    CKPT_SIZE=$(du -sh "${MEGATRON_CKPT}" | cut -f1)
    log "Checkpoint verified: ${MEGATRON_CKPT} (${CKPT_SIZE})"
    log "Phase 2: PASSED"
else
    log "ERROR: Checkpoint not created!"
    log "Phase 2: FAILED"
    exit 1
fi

echo ""

# =============================================================================
# Quick Training Config Validation
# =============================================================================
log "=========================================="
log " Bonus: Training Config Validation"
log "=========================================="

python3 -c "
from megatron.bridge.recipes.nemotronh import nemotron_nano_9b_v2_finetune_config
from megatron.bridge.peft.lora import LoRA

lora = LoRA(
    target_modules=['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2', 'in_proj', 'out_proj'],
    dim=32, alpha=32,
)
config = nemotron_nano_9b_v2_finetune_config(
    peft=lora,
    pretrained_checkpoint='${MEGATRON_CKPT}',
    train_iters=3, micro_batch_size=1, global_batch_size=1,
    seq_length=512, finetune_lr=1e-4,
)
config.validate()
print('Config validation: PASSED')
print(f'  pretrained_checkpoint: {config.checkpoint.pretrained_checkpoint}')
print(f'  LoRA targets: {config.peft.target_modules}')
print(f'  TP: {config.model.tensor_model_parallel_size}')
"

log "Config validation: PASSED"

echo ""
log "=========================================="
log " DRY RUN COMPLETE"
log "=========================================="
echo ""
echo "All checks passed! The pipeline is ready for full training."
echo ""
echo "Next steps:"
echo "  1. DELETE this Brev instance (stop billing)"
echo "  2. For full training, launch new instance and run:"
echo "     bash n6-brev-run.sh"
echo ""
echo "Logs saved to: ${LOG_DIR}/"
echo "Checkpoint at: ${MEGATRON_CKPT}"
