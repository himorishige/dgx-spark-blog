#!/usr/bin/env bash
# Sample system resources during a DwarfStar 4 run.
# Usage: monitor-sys.sh <label> <out_dir> [interval_sec]
# Writes <out_dir>/<label>-{free,gpu,iostat}.log until killed.
set -euo pipefail
label="${1:?label required}"
out_dir="${2:?out dir required}"
interval="${3:-2}"
mkdir -p "$out_dir"

nvme_dev="$(lsblk -ndo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')"
echo "monitoring nvme device: ${nvme_dev:-<none>}" >&2

( while true; do echo "=== $(date +%s) ==="; free -m; sleep "$interval"; done ) > "$out_dir/${label}-free.log" 2>&1 &
free_pid=$!
( nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l "$interval" ) > "$out_dir/${label}-gpu.log" 2>&1 &
gpu_pid=$!
( iostat -x "$interval" ) > "$out_dir/${label}-iostat.log" 2>&1 &
io_pid=$!

trap 'kill $free_pid $gpu_pid $io_pid 2>/dev/null || true' EXIT INT TERM
echo "monitor pids: free=$free_pid gpu=$gpu_pid iostat=$io_pid (Ctrl-C to stop)" >&2
wait
