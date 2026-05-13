#!/usr/bin/env bash
# Phase 2: long-context prefill + disk KV cache cold-vs-warm experiment.
#
# DwarfStar 4's disk KV cache persists a "cold" prefix checkpoint so a client
# that re-sends a longer version of the same prompt skips re-prefilling token 0.
# Here we measure: (1) prefill time/throughput at growing context sizes, while
# logging memory + NVMe I/O; (2) cold prefill of an ~Ntok prompt vs warm
# re-prefill of the same prompt after a server restart (cache hit).
#
# Requires the ds4 binaries built in $DS4_DIR.
set -euo pipefail
DS4_DIR="${DS4_DIR:-$HOME/works/dwarfstar4/ds4}"
R="${R:-$HOME/works/dwarfstar4/results}"
KVDIR="${KVDIR:-/tmp/ds4-kv}"
PROMPT="${PROMPT:-$DS4_DIR/speed-bench/promessi_sposi.txt}"
mkdir -p "$R"
cd "$DS4_DIR"

NVME="$(lsblk -ndo NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1; exit}')"
echo "nvme device: ${NVME:-<none>}, kv dir: $KVDIR"

start_mon() {  # $1 = label
  ( while true; do
      echo "$(date +%s) $(free -m | awk '/^Mem:/{print "used="$3" buffcache="$6" avail="$7}')"
      sleep 2
    done ) > "$R/$1-mem.log" 2>&1 &
  echo $!
  ( iostat -x 2 "$NVME" ) > "$R/$1-iostat.log" 2>&1 &
  echo $!
}

# --- Part A: prefill at growing context sizes (no generation, big -n=1) ---
# Use ds4 one-shot with --prompt-file truncated by ds4-bench-style frontier is
# simpler with ds4-bench --ctx-max; we already have the 65k sweep. Here we push
# to longer contexts with a coarse sweep.
echo "=== Part A: long-context coarse sweep (ctx 65536 -> 262144, step-mul 2) ==="
PIDS=$(start_mon partA)
time ./ds4-bench -m ds4flash.gguf --prompt-file "$PROMPT" \
  --ctx-start 32768 --ctx-max 262144 --step-mul 2 --gen-tokens 64 \
  --csv "$R/dgx-spark-q2-longctx.csv" || echo "(longctx sweep ended early)"
kill $PIDS 2>/dev/null || true
echo "longctx csv:"; cat "$R/dgx-spark-q2-longctx.csv" 2>/dev/null || true

# --- Part B: disk KV cache cold vs warm ---
echo "=== Part B: disk KV cache cold vs warm (~30k token prompt) ==="
rm -rf "$KVDIR"; mkdir -p "$KVDIR"
# Build a ~30k-token prompt slice (head of promessi_sposi); ds4 will tokenize.
head -c 120000 "$PROMPT" > "$R/kv-prompt.txt"

echo "--- cold run (empty cache) ---"
PIDS=$(start_mon partB-cold)
( time ./ds4-server --ctx 100000 --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --port 8011 --trace "$R/partB-cold-trace.txt" >/dev/null 2>&1 & echo $! > "$R/srv.pid"; sleep 18
  curl -s -m 600 http://127.0.0.1:8011/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()+"\n\nIn one sentence, what is this text about?"))' "$R/kv-prompt.txt")}],\"max_tokens\":16,\"thinking\":{\"type\":\"disabled\"}}" > "$R/partB-cold-resp.json"
  kill "$(cat "$R/srv.pid")" 2>/dev/null || true ) 2>&1 | tee "$R/partB-cold.log"
sleep 4; kill $PIDS 2>/dev/null || true
ls -la "$KVDIR"

echo "--- warm run (cache populated, server restarted) ---"
PIDS=$(start_mon partB-warm)
( time ./ds4-server --ctx 100000 --kv-disk-dir "$KVDIR" --kv-disk-space-mb 16384 \
    --port 8012 --trace "$R/partB-warm-trace.txt" >/dev/null 2>&1 & echo $! > "$R/srv.pid"; sleep 18
  curl -s -m 600 http://127.0.0.1:8012/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c 'import json,sys;print(json.dumps(open(sys.argv[1]).read()+"\n\nIn one sentence, what is this text about?"))' "$R/kv-prompt.txt")}],\"max_tokens\":16,\"thinking\":{\"type\":\"disabled\"}}" > "$R/partB-warm-resp.json"
  kill "$(cat "$R/srv.pid")" 2>/dev/null || true ) 2>&1 | tee "$R/partB-warm.log"
sleep 4; kill $PIDS 2>/dev/null || true

echo "=== traces: grep cache decisions ==="
grep -i -E 'cache|prefill|cold|warm|reuse|hit|prefix' "$R/partB-cold-trace.txt" 2>/dev/null | head -20 || true
echo "---"
grep -i -E 'cache|prefill|cold|warm|reuse|hit|prefix' "$R/partB-warm-trace.txt" 2>/dev/null | head -20 || true
