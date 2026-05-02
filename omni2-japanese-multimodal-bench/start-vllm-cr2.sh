#!/usr/bin/env bash
# Launch Cosmos-Reason2-8B on DGX Spark.
# Model: nvidia/Cosmos-Reason2-8B (~16GB weights, BF16, dense Transformer VLM)
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

# CR2 is 8B BF16 (smallest of the three). Set util a bit lower since model is
# small enough that overhead matters less; same image limits as Omni/Gemma.
exec vllm serve nvidia/Cosmos-Reason2-8B \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name cosmos-reason2 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.3 \
  --enforce-eager \
  --media-io-kwargs '{"image":{"max_pixels":2097152}}' \
  --limit-mm-per-prompt '{"image":4}' \
  --trust-remote-code
