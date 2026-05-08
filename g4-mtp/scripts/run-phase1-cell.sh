#!/usr/bin/env bash
# Run a single Phase 1 cell: start vLLM, wait for ready, run JCQ bench, stop.
#
# Usage:
#   ./run-phase1-cell.sh <size> <mtp_arg> <label>
# Example:
#   ./run-phase1-cell.sh e2b baseline e2b-baseline
#   ./run-phase1-cell.sh e2b mtp:2 e2b-mtp2
#   ./run-phase1-cell.sh e4b mtp:1 e4b-mtp1
#   ./run-phase1-cell.sh 26b-a4b mtp:2 26b-a4b-mtp2
#
# Side effects:
#   - vLLM serve runs on port 8001 during the bench, then is killed.
#   - Logs go to ~/works/gemma4-mtp/logs/vllm-{label}.log
#   - JCQ results go to data/g4-mtp/{label}.jsonl + summary.

set -euo pipefail

SIZE="${1:?size required}"
MTP="${2:?mtp arg required (baseline | mtp:N)}"
LABEL="${3:?label required}"

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

echo "=== [${LABEL}] running JCQ bench"
source "${HOME}/works/gemma4-mtp/.venv/bin/activate"
python "${SCRIPT_DIR}/bench-jcq.py" --model "${SERVED}" --label "${LABEL}"

echo "=== [${LABEL}] stopping vLLM"
pkill -TERM -f "vllm serve" 2>/dev/null || true
sleep 8
if ss -tln 2>/dev/null | grep -qE ':8001\s'; then
  echo "  port still listening, sending KILL"
  pkill -KILL -f "vllm serve" 2>/dev/null || true
  sleep 5
fi

echo "=== [${LABEL}] done"
