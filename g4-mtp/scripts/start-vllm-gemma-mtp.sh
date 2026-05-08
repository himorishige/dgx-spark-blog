#!/usr/bin/env bash
# Launch Gemma 4 with optional MTP (Multi-Token Prediction) drafter on DGX Spark.
#
# vLLM nightly required (PR #41745, merged 2026-05-06, SHA 27e0057).
# Tested env: vllm 0.20.2rc1.dev99 + torch 2.11.0+cu130 + aarch64 / SM121 / 128GB UMA.
#
# Drafter model_type "gemma4_assistant" is auto-rewritten to "gemma4_mtp" inside vLLM,
# which bypasses _raise_if_multimodal() by zeroing image/audio at draft time.
#
# Usage:
#   ./start-vllm-gemma-mtp.sh <size> [baseline | mtp:<N>]
#
# Examples:
#   ./start-vllm-gemma-mtp.sh e2b              # target only, baseline
#   ./start-vllm-gemma-mtp.sh e2b mtp:4        # target + drafter, num_speculative_tokens=4
#   ./start-vllm-gemma-mtp.sh e4b mtp:1
#   ./start-vllm-gemma-mtp.sh 26b-a4b mtp:4
#   ./start-vllm-gemma-mtp.sh 31b mtp:4
#
# Phase 0 (5/7) focus: e2b の動作確認 → e4b → 26b-a4b → 31b の順で smoke test
set -euo pipefail

SIZE="${1:?model size required: e2b | e4b | 26b-a4b | 31b}"
MTP="${2:-baseline}"

case "$SIZE" in
  e2b)
    TARGET="google/gemma-4-E2B-it"
    DRAFTER="google/gemma-4-E2B-it-assistant"
    SERVED="gemma4-e2b"
    ;;
  e4b)
    TARGET="google/gemma-4-E4B-it"
    DRAFTER="google/gemma-4-E4B-it-assistant"
    SERVED="gemma4-e4b"
    ;;
  26b-a4b)
    # NVFP4 quantized weights (already in HF cache from Omni2 verification).
    TARGET="nvidia/Gemma-4-26B-A4B-NVFP4"
    DRAFTER="google/gemma-4-26B-A4B-it-assistant"
    SERVED="gemma4-a4b"
    ;;
  31b)
    # NVFP4 quantized weights (BF16 65GB exceeds DGX Spark UMA budget).
    TARGET="nvidia/Gemma-4-31B-IT-NVFP4"
    DRAFTER="google/gemma-4-31B-it-assistant"
    SERVED="gemma4-31b"
    ;;
  *)
    echo "Unknown size: $SIZE (expected: e2b | e4b | 26b-a4b | 31b)" >&2
    exit 1
    ;;
esac

VENV="${HOME}/works/gemma4-mtp/.venv"
export VLLM_CACHE_ROOT="${HOME}/.cache/vllm-local"
mkdir -p "${VLLM_CACHE_ROOT}"

if [ ! -d "${VENV}" ]; then
  echo "venv not found at ${VENV}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

ARGS=(
  --host 0.0.0.0
  --port 8001
  --served-model-name "${SERVED}"
  --max-model-len 8192
  --max-num-seqs 4
  --gpu-memory-utilization 0.5
  --enforce-eager
  # Gemma 4 multimodal token size is 2496; default 2048 fails with
  # "max_tokens_per_mm_item > max_num_batched_tokens" on 26B-A4B / 31B.
  --max-num-batched-tokens 4096
)

if [[ "${MTP}" == "baseline" ]]; then
  echo ">>> baseline (target only) for ${SERVED}"
else
  N="${MTP#mtp:}"
  if ! [[ "${N}" =~ ^[0-9]+$ ]]; then
    echo "Invalid MTP arg: ${MTP} (expected baseline | mtp:<int>)" >&2
    exit 1
  fi
  echo ">>> MTP enabled: drafter=${DRAFTER}, num_speculative_tokens=${N}"
  # PR #41745 (merged 2026-05-06): vLLM's "mtp" method now accepts Gemma 4 drafters.
  # The drafter's model_type "gemma4_assistant" is auto-converted to "gemma4_mtp".
  ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${N},\"model\":\"${DRAFTER}\"}")
fi

echo ">>> vllm serve ${TARGET} ${ARGS[*]}"
exec vllm serve "${TARGET}" "${ARGS[@]}"
