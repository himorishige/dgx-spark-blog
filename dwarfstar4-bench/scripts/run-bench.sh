#!/usr/bin/env bash
# Phase 1: ds4-bench frontier sweep on DGX Spark + resource log.
# Usage: run-bench.sh [ctx_max] [step_incr] [gen_tokens]
set -euo pipefail
DS4_DIR="${DS4_DIR:-$HOME/works/dwarfstar4/ds4}"
R="${R:-$HOME/works/dwarfstar4/results}"
mkdir -p "$R"
CTX_MAX="${1:-65536}"
STEP="${2:-2048}"
GEN="${3:-128}"
PROMPT="${PROMPT:-$DS4_DIR/speed-bench/promessi_sposi.txt}"
CSV="$R/dgx-spark-q2-sweep.csv"

cd "$DS4_DIR"
( while true; do
    echo "$(date +%s) $(free -m | awk '/^Mem:/{print "used="$3" buffcache="$6" avail="$7}') gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\n')"
    sleep 3
  done ) > "$R/phase1-mem.log" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

echo "ds4-bench: ctx 2048..$CTX_MAX step $STEP gen $GEN -> $CSV"
time ./ds4-bench -m ds4flash.gguf --prompt-file "$PROMPT" \
  --ctx-start 2048 --ctx-max "$CTX_MAX" --step-incr "$STEP" --gen-tokens "$GEN" \
  --csv "$CSV"
echo "done. rows: $(($(wc -l < "$CSV") - 1))"
