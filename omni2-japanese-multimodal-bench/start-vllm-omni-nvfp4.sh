#!/usr/bin/env bash
# Launch Nemotron 3 Nano Omni NVFP4 on DGX Spark for Omni2 Japanese multimodal bench.
# Based on chocolate-factory-poc/infra/vllm/run.sh with --media-io-kwargs added
# for Heron-Bench (image-only) workload.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="/home/morishige/works/private/workspace-dgx/workspace/scratchpad/projects/chocolate-factory-poc/infra/vllm"

# Pull HF_TOKEN from chocolate-factory-poc .env
if [ -f "${VLLM_DIR}/../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${VLLM_DIR}/../.env"
  set +a
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# Avoid root-owned /root/.cache/vllm pitfall (Omni1 章 8 hammari)
export VLLM_CACHE_ROOT="${HOME}/.cache/vllm-local"
mkdir -p "${VLLM_CACHE_ROOT}"

source "${VLLM_DIR}/.venv/bin/activate"

# Memory budget: same as chocolate-factory-poc/infra/vllm/run.sh (NVFP4, 0.4 util,
# enforce-eager) — 21GB weights + KV headroom on 128GB UMA.
exec vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name nemotron-omni \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.4 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --moe-backend flashinfer_cutlass \
  --media-io-kwargs '{"image":{"max_pixels":2097152}}' \
  --limit-mm-per-prompt '{"image":4}' \
  --trust-remote-code
