#!/usr/bin/env bash
# Launch Gemma 4 26B-A4B NVFP4 on DGX Spark (MoE, Active 3.8B).
# Model: nvidia/Gemma-4-26B-A4B-NVFP4 (~17GB weights, NVIDIA Model Optimizer NVFP4)
# 31B Dense is ~8x slower than Omni MoE so we use the A4B variant for fair
# throughput comparison. The 31B Dense numbers are kept as a latency footnote.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="/home/morishige/works/private/workspace-dgx/workspace/scratchpad/projects/chocolate-factory-poc/infra/vllm"

if [ -f "${VLLM_DIR}/../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${VLLM_DIR}/../.env"
  set +a
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export VLLM_CACHE_ROOT="${HOME}/.cache/vllm-local"
mkdir -p "${VLLM_CACHE_ROOT}"

source "${VLLM_DIR}/.venv/bin/activate"

# Gemma 4 26B-A4B is MoE with GELU activation; FLASHINFER_CUTLASS does not
# support GELU, so let vLLM auto-pick a compatible backend (TRTLLM / CUTEDSL /
# MARLIN are tried in order).
exec vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name gemma4-a4b \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager \
  --media-io-kwargs '{"image":{"max_pixels":2097152}}' \
  --limit-mm-per-prompt '{"image":4}' \
  --trust-remote-code
