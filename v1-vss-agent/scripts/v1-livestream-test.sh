#!/bin/bash
# v1-livestream-test.sh - VSS LIVE STREAM SUMMARIZATION test script
# Streams a local video file as RTSP and connects it to VSS for real-time analysis.
#
# Usage:
#   bash scripts/v1-livestream-test.sh start [video_path]  # Start MediaMTX + FFmpeg + register stream
#   bash scripts/v1-livestream-test.sh summarize            # Start live summarization
#   bash scripts/v1-livestream-test.sh query "question"     # Q&A on live stream
#   bash scripts/v1-livestream-test.sh status               # Check status
#   bash scripts/v1-livestream-test.sh stop                 # Cleanup

set -euo pipefail

VSS_API="http://localhost:8100"
RTSP_HOST="host.docker.internal"
RTSP_PORT=8554
STREAM_NAME="warehouse"
DEFAULT_VIDEO="$HOME/videos/vss-test/warehouse.mp4"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[VSS-LIVE]${NC} $*"; }
warn() { echo -e "${YELLOW}[VSS-LIVE]${NC} $*"; }
err() { echo -e "${RED}[VSS-LIVE]${NC} $*" >&2; }

cmd_start() {
    local video="${1:-$DEFAULT_VIDEO}"

    if [ ! -f "$video" ]; then
        err "Video file not found: $video"
        exit 1
    fi

    log "Starting MediaMTX RTSP server..."
    if docker ps --format '{{.Names}}' | grep -q '^mediamtx$'; then
        warn "MediaMTX is already running"
    else
        docker run --rm -d --name mediamtx --network host bluenviron/mediamtx:latest
        sleep 2
    fi

    if ! ss -tlnp | grep -q ":${RTSP_PORT}"; then
        err "MediaMTX failed to start on port ${RTSP_PORT}"
        exit 1
    fi
    log "MediaMTX listening on port ${RTSP_PORT}"

    log "Starting FFmpeg RTSP stream: $video -> rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}"
    if pgrep -f "ffmpeg.*rtsp.*${STREAM_NAME}" > /dev/null 2>&1; then
        warn "FFmpeg stream already running"
    else
        ffmpeg -re -stream_loop -1 \
            -i "$video" \
            -c copy \
            -f rtsp "rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}" \
            > /tmp/ffmpeg-rtsp.log 2>&1 &
        echo $! > /tmp/ffmpeg-rtsp.pid
        sleep 3
    fi

    if ! ffprobe -v quiet -show_format "rtsp://localhost:${RTSP_PORT}/${STREAM_NAME}" > /dev/null 2>&1; then
        err "RTSP stream verification failed"
        exit 1
    fi
    log "RTSP stream verified"

    log "Registering live stream with VSS..."
    local response
    response=$(curl -s -X POST "${VSS_API}/live-stream" \
        -H "Content-Type: application/json" \
        -d "{
            \"liveStreamUrl\": \"rtsp://${RTSP_HOST}:${RTSP_PORT}/${STREAM_NAME}\",
            \"description\": \"Live Stream Test - $(basename "$video")\",
            \"camera_id\": \"camera_1\"
        }")

    local stream_id
    stream_id=$(echo "$response" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

    if [ -z "$stream_id" ]; then
        err "Failed to register stream: $response"
        exit 1
    fi

    echo "$stream_id" > /tmp/vss-stream-id.txt
    log "Stream registered: $stream_id"
    log "Done. Run 'bash $0 summarize' to start live summarization."
}

cmd_summarize() {
    local stream_id
    stream_id=$(cat /tmp/vss-stream-id.txt 2>/dev/null)
    if [ -z "$stream_id" ]; then
        err "No stream ID found. Run 'start' first."
        exit 1
    fi

    log "Starting live summarization (chunk=10s, summary=30s)..."
    curl -s -N -X POST "${VSS_API}/summarize" \
        -H "Content-Type: application/json" \
        -d "{
            \"id\": \"${stream_id}\",
            \"model\": \"Cosmos-Reason2-8B\",
            \"prompt\": \"Write a concise and clear dense caption for the provided video, focusing on irregular or hazardous events. Start and end each sentence with a time stamp.\",
            \"caption_summarization_prompt\": \"Summarize the following events in the format start_time:end_time:caption. Note any irregular activities in detail. Output as bullet points.\",
            \"summary_aggregation_prompt\": \"Aggregate the following captions. If events are similar, merge them. Cluster output into relevant categories.\",
            \"chunk_duration\": 10,
            \"summary_duration\": 30,
            \"enable_chat\": true,
            \"summarize\": true,
            \"stream\": true,
            \"max_tokens\": 2048
        }" &
    local curl_pid=$!
    echo "$curl_pid" > /tmp/vss-summarize.pid
    log "Summarization started (PID: $curl_pid). Press Ctrl+C to stop watching."
    log "Q&A is now enabled. Run: bash $0 query \"your question\""
    wait "$curl_pid" 2>/dev/null || true
}

cmd_query() {
    local question="${1:?Usage: $0 query \"your question\"}"
    local stream_id
    stream_id=$(cat /tmp/vss-stream-id.txt 2>/dev/null)
    if [ -z "$stream_id" ]; then
        err "No stream ID found. Run 'start' first."
        exit 1
    fi

    log "Q&A: $question"
    local start_time=$SECONDS
    local response
    response=$(curl -s -X POST "${VSS_API}/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"id\": \"${stream_id}\",
            \"model\": \"Cosmos-Reason2-8B\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${question}\"}],
            \"stream\": false,
            \"max_tokens\": 512
        }")

    local elapsed=$((SECONDS - start_time))
    echo "$response" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'choices' in data:
    for c in data['choices']:
        print(c['message']['content'])
    usage = data.get('usage', {})
    print(f'--- chunks: {usage.get(\"total_chunks_processed\", \"N/A\")}, latency: ${elapsed}s ---')
else:
    print(json.dumps(data, indent=2))
"
}

cmd_status() {
    log "=== Live Stream Status ==="
    echo ""

    echo "MediaMTX:"
    if docker ps --format '{{.Names}}' | grep -q '^mediamtx$'; then
        echo "  Running"
    else
        echo "  Not running"
    fi

    echo ""
    echo "FFmpeg RTSP:"
    if pgrep -f "ffmpeg.*rtsp.*${STREAM_NAME}" > /dev/null 2>&1; then
        echo "  Streaming (PID: $(cat /tmp/ffmpeg-rtsp.pid 2>/dev/null || echo 'unknown'))"
    else
        echo "  Not streaming"
    fi

    echo ""
    echo "VSS Live Streams:"
    curl -s "${VSS_API}/live-stream" | python3 -m json.tool 2>/dev/null || echo "  API not available"

    echo ""
    echo "VIA Server Logs (last 5 lines):"
    docker logs --tail 5 local_deployment_single_gpu-via-server-1 2>&1 | grep -v "HTTP/1.1"
}

cmd_stop() {
    log "Stopping live stream test..."

    local stream_id
    stream_id=$(cat /tmp/vss-stream-id.txt 2>/dev/null)
    if [ -n "$stream_id" ]; then
        log "Removing stream from VSS: $stream_id"
        curl -s -X DELETE "${VSS_API}/live-stream/${stream_id}" || true
        rm -f /tmp/vss-stream-id.txt
    fi

    if [ -f /tmp/vss-summarize.pid ]; then
        kill "$(cat /tmp/vss-summarize.pid)" 2>/dev/null || true
        rm -f /tmp/vss-summarize.pid
    fi

    log "Stopping FFmpeg..."
    pkill -f "ffmpeg.*rtsp.*${STREAM_NAME}" 2>/dev/null || true
    rm -f /tmp/ffmpeg-rtsp.pid /tmp/ffmpeg-rtsp.log

    log "Stopping MediaMTX..."
    docker stop mediamtx 2>/dev/null || true

    log "Cleanup complete."
}

case "${1:-help}" in
    start)     cmd_start "${2:-}" ;;
    summarize) cmd_summarize ;;
    query)     cmd_query "${2:-}" ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    *)
        echo "Usage: $0 {start|summarize|query|status|stop}"
        echo ""
        echo "  start [video]  - Start RTSP server + stream + register with VSS"
        echo "  summarize      - Start live summarization (enables Q&A)"
        echo "  query \"text\"   - Ask a question about the live stream"
        echo "  status         - Show current status"
        echo "  stop           - Stop everything and cleanup"
        ;;
esac
