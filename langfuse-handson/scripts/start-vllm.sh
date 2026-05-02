#!/usr/bin/env bash
# Start a Nemotron 3 Nano 30B-A3B NVFP4 vLLM server on DGX Spark for the
# Langfuse handson chapter 13 (local LLM tracing).
#
# Reuses the existing chocolate-factory-poc vLLM venv to avoid an extra build.

set -euo pipefail

VENV=${VENV:-/home/morishige/works/private/workspace-dgx/workspace/scratchpad/projects/chocolate-factory-poc/infra/vllm/.venv}
MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}
SERVED_NAME=${SERVED_NAME:-nemotron-3-nano-nvfp4-local}
PORT=${PORT:-8001}

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm-local}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torch_inductor}"
# Mamba/SSM kernels JIT-compile through ninja; the venv ships ninja but
# its bin/ is not on PATH by default, so subprocess lookup fails.
export PATH="$VENV/bin:$PATH"

mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --moe-backend flashinfer_cutlass \
  --trust-remote-code
