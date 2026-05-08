#!/usr/bin/env bash
# Long-form (max_tokens=256) bench cell for Phase 1 Day 2.
# Same shape as run-phase1-cell.sh but runs bench-longform.py with --grab-metrics.
#
# Usage:
#   ./run-longform-cell.sh <size> <mtp_arg> <label> [lang]
# Example:
#   ./run-longform-cell.sh e2b baseline e2b-longform-baseline
#   ./run-longform-cell.sh e2b mtp:2 e2b-longform-mtp2
#   ./run-longform-cell.sh e2b mtp:2 e2b-longform-mtp2-en en
#
# Side effects:
#   - vLLM serve runs on port 8001 during the bench, then is killed.
#   - Logs go to ~/works/gemma4-mtp/logs/vllm-{label}.log
#   - Long-form results go to data/g4-mtp/{label}.long.jsonl + .summary.json

set -euo pipefail

SIZE="${1:?size required}"
MTP="${2:?mtp arg required (baseline | mtp:N)}"
LABEL="${3:?label required}"
LANG="${4:-ja}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${HOME}/works/gemma4-mtp/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/vllm-${LABEL}.log"

case "$SIZE" in
  e2b)        SERVED="gemma4-e2b" ;;
  e4b)        SERVED="gemma4-e4b" ;;
  26b-a4b)    SERVED="gemma4-a4b" ;;
  31b)        SERVED="gemma4-31b" ;;
  *) echo "unknown size $SIZE" >&2; exit 1 ;;
esac

GRAB_METRICS=()
if [[ "${MTP}" != "baseline" ]]; then
  GRAB_METRICS+=(--grab-metrics)
fi

echo "=== [${LABEL}] start vLLM ${SIZE} ${MTP} → ${LOG}"
nohup bash "${SCRIPT_DIR}/start-vllm-gemma-mtp.sh" "${SIZE}" "${MTP}" > "${LOG}" 2>&1 &
SERVER_PID=$!
echo "  server pid: ${SERVER_PID}"

echo "=== [${LABEL}] waiting for port 8001 ..."
for _ in $(seq 1 240); do
  if ss -tln 2>/dev/null | grep -qE ':8001\s'; then
    break
  fi
  if grep -qE "Traceback|RuntimeError|NotImplementedError|ValueError|Killed|CUDA out of memory" "${LOG}" 2>/dev/null; then
    echo "  ERROR detected in log:"
    tail -15 "${LOG}"
    exit 2
  fi
  sleep 5
done

if ! ss -tln 2>/dev/null | grep -qE ':8001\s'; then
  echo "  TIMEOUT waiting for port 8001"
  tail -25 "${LOG}"
  pkill -TERM -f "vllm serve" 2>/dev/null || true
  exit 3
fi

echo "=== [${LABEL}] running long-form bench (lang=${LANG})"
source "${HOME}/works/gemma4-mtp/.venv/bin/activate"
python "${SCRIPT_DIR}/bench-longform.py" --model "${SERVED}" --label "${LABEL}" --rounds 12 --max-tokens 256 --lang "${LANG}" "${GRAB_METRICS[@]}"

echo "=== [${LABEL}] stopping vLLM"
pkill -TERM -f "vllm serve" 2>/dev/null || true
sleep 8
if ss -tln 2>/dev/null | grep -qE ':8001\s'; then
  echo "  port still listening, sending KILL"
  pkill -KILL -f "vllm serve" 2>/dev/null || true
  sleep 5
fi

echo "=== [${LABEL}] done"
