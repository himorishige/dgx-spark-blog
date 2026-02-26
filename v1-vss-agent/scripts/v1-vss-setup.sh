#!/usr/bin/env bash
# v1-vss-setup.sh — VSS Agent on DGX Spark セットアップスクリプト
#
# DGX Spark (ARM64/Blackwell) で VSS Agent を Single GPU 構成でデプロイする手順を自動化します。
# 事前に NGC API Key と HuggingFace Token を環境変数で設定してください。
#
# Usage:
#   export NGC_API_KEY="your-key"
#   export HF_TOKEN="your-token"
#   bash v1-vss-setup.sh [step]
#
# Steps:
#   check   - 環境チェック（デフォルト）
#   login   - NGC Container Registry ログイン
#   nim     - LLM / Embedding / Reranker NIM 起動
#   vss     - VSS 本体デプロイ（docker compose up）
#   status  - 全コンポーネントの状態確認
#   stop    - 全コンテナ停止
#   nemotron - Nemotron 9B-v2-Japanese に LLM 差し替え

set -euo pipefail

VSS_REPO="${VSS_REPO:-$HOME/video-search-and-summarization}"
VSS_DEPLOY_DIR="${VSS_REPO}/deploy/docker/local_deployment_single_gpu"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-/tmp/nim-cache}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─────────────────────────────────────────────
# Step: check — 環境チェック
# ─────────────────────────────────────────────
step_check() {
  info "=== DGX Spark 環境チェック ==="

  # DGX OS
  if [ -f /etc/dgx-release ]; then
    local os_ver
    os_ver=$(grep DGX_OTA_VERSION /etc/dgx-release | tail -1 | cut -d= -f2 | tr -d '"')
    info "DGX OS: ${os_ver}"
  else
    warn "DGX OS release file not found"
  fi

  # GPU Driver
  local driver_ver
  driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  info "GPU Driver: ${driver_ver}"

  # CUDA
  local cuda_ver
  cuda_ver=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
  info "CUDA: ${cuda_ver}"

  # Docker
  if command -v docker &>/dev/null; then
    info "Docker: $(docker --version | awk '{print $3}' | tr -d ',')"
  else
    error "Docker not found"
    return 1
  fi

  # NGC API Key
  if [ -n "${NGC_API_KEY:-}" ]; then
    info "NGC_API_KEY: set (${#NGC_API_KEY} chars)"
  else
    warn "NGC_API_KEY: not set"
  fi

  # HuggingFace Token
  if [ -n "${HF_TOKEN:-}" ]; then
    info "HF_TOKEN: set"
  else
    warn "HF_TOKEN: not set"
  fi

  # Storage
  local avail
  avail=$(df -h /tmp | tail -1 | awk '{print $4}')
  info "Storage available (/tmp): ${avail}"

  # Nemotron in Ollama
  if command -v ollama &>/dev/null; then
    local nemotron_count
    nemotron_count=$(ollama list 2>/dev/null | grep -ci nemotron || true)
    info "Nemotron models in Ollama: ${nemotron_count}"
  fi

  # VSS Repo
  if [ -d "${VSS_REPO}" ]; then
    info "VSS repo: ${VSS_REPO}"
  else
    warn "VSS repo not found at ${VSS_REPO}. Run: git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git"
  fi

  echo ""
  info "Environment check complete."
}

# ─────────────────────────────────────────────
# Step: login — NGC ログイン
# ─────────────────────────────────────────────
step_login() {
  if [ -z "${NGC_API_KEY:-}" ]; then
    error "NGC_API_KEY is not set"
    return 1
  fi
  info "Logging in to NGC Container Registry..."
  echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
  info "NGC login successful"
}

# ─────────────────────────────────────────────
# Step: nim — LLM / Embedding / Reranker NIM 起動
# ─────────────────────────────────────────────
step_nim() {
  if [ -z "${NGC_API_KEY:-}" ]; then
    error "NGC_API_KEY is not set"
    return 1
  fi

  mkdir -p "${LOCAL_NIM_CACHE}"

  info "=== Starting NIM containers ==="

  # LLM NIM (Llama 3.1 8B, DGX Spark optimized)
  info "Starting LLM NIM (Llama 3.1 8B) on port 8007..."
  docker run -d --name vss-llm-nim \
    -u "$(id -u)" \
    --gpus '"device=0"' \
    --shm-size=16GB \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
    -p 8007:8000 \
    -e NIM_GPU_MEMORY_FRACTION=0.2 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:1.0 \
    2>&1 || warn "LLM NIM container may already exist"

  # Embedding NIM
  info "Starting Embedding NIM on port 8006..."
  docker run -d --name vss-embedding-nim \
    -u "$(id -u)" \
    --gpus '"device=0"' \
    --shm-size=16GB \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
    -p 8006:8000 \
    nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.9.0 \
    2>&1 || warn "Embedding NIM container may already exist"

  # Reranker NIM
  info "Starting Reranker NIM on port 8005..."
  docker run -d --name vss-reranker-nim \
    -u "$(id -u)" \
    --gpus '"device=0"' \
    --shm-size=16GB \
    -e NGC_API_KEY="${NGC_API_KEY}" \
    -v "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
    -p 8005:8000 \
    nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:1.7.0 \
    2>&1 || warn "Reranker NIM container may already exist"

  info "NIM containers started. Use 'docker logs -f <name>' to monitor startup."
  info "Wait for all NIMs to report 'ready' before proceeding to VSS deploy."
}

# ─────────────────────────────────────────────
# Step: vss — VSS 本体デプロイ
# ─────────────────────────────────────────────
step_vss() {
  if [ ! -d "${VSS_DEPLOY_DIR}" ]; then
    error "VSS deploy dir not found: ${VSS_DEPLOY_DIR}"
    return 1
  fi

  info "=== Deploying VSS (Standard mode, Single GPU) ==="
  info "Deploy dir: ${VSS_DEPLOY_DIR}"

  cd "${VSS_DEPLOY_DIR}"

  # Source .env
  set -a
  source .env
  set +a

  # ARM64 flag for DGX Spark
  export IS_SBSA=1

  # Override with actual keys
  export NGC_API_KEY="${NGC_API_KEY}"
  export HF_TOKEN="${HF_TOKEN}"

  info "IS_SBSA=1 (ARM64 mode)"
  info "VLM: ${VLM_MODEL_TO_USE:-cosmos-reason2}"
  info "VLM_BATCH_SIZE: ${VLM_BATCH_SIZE:-32}"
  info "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"

  docker compose up -d
  info "VSS deployed. Access Web UI at http://$(hostname -I | awk '{print $1}'):9100"
}

# ─────────────────────────────────────────────
# Step: status — 状態確認
# ─────────────────────────────────────────────
step_status() {
  info "=== Container Status ==="
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "vss|neo4j|milvus|arango|minio|elasticsearch|via" || true

  echo ""
  info "=== GPU Memory ==="
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

  echo ""
  info "=== Endpoint Health ==="
  for port in 8005 8006 8007; do
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/health/ready" 2>/dev/null || echo "down")
    info "  Port ${port}: ${status}"
  done

  for port in 8100 9100; do
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}" 2>/dev/null || echo "down")
    info "  Port ${port}: ${status}"
  done
}

# ─────────────────────────────────────────────
# Step: stop — 全停止
# ─────────────────────────────────────────────
step_stop() {
  info "=== Stopping all VSS containers ==="

  # Stop NIM containers
  for name in vss-llm-nim vss-embedding-nim vss-reranker-nim vss-nemotron-nim; do
    docker stop "${name}" 2>/dev/null && docker rm "${name}" 2>/dev/null && info "Stopped ${name}" || true
  done

  # Stop VSS compose
  if [ -d "${VSS_DEPLOY_DIR}" ]; then
    cd "${VSS_DEPLOY_DIR}"
    set -a; source .env; set +a
    export IS_SBSA=1
    docker compose down 2>/dev/null && info "VSS compose stopped" || true
  fi

  info "All VSS containers stopped."
}

# ─────────────────────────────────────────────
# Step: nemotron — Nemotron 9B-v2-Japanese に差し替え
# ─────────────────────────────────────────────
step_nemotron() {
  info "=== Switching LLM to Nemotron 9B-v2-Japanese ==="

  # Stop Llama NIM
  info "Stopping Llama 3.1 8B NIM..."
  docker stop vss-llm-nim 2>/dev/null && docker rm vss-llm-nim 2>/dev/null || true

  # Ensure Ollama is running
  if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    info "Starting Ollama..."
    ollama serve &
    sleep 3
  fi

  # Verify Nemotron model
  if ollama list 2>/dev/null | grep -q "nemotron-9b-jp-nothink"; then
    info "Nemotron 9B-v2-Japanese (nothink) is available in Ollama"
  else
    warn "nemotron-9b-jp-nothink not found. Available nemotron models:"
    ollama list 2>/dev/null | grep nemotron || true
  fi

  # Update config.yaml
  local config="${VSS_DEPLOY_DIR}/config.yaml"
  if [ -f "${config}" ]; then
    info "Updating config.yaml for Nemotron..."
    cp "${config}" "${config}.llama-backup"

    # Apply Nemotron config
    local nemotron_config
    nemotron_config="$(dirname "$0")/../config/config-nemotron.yaml"
    if [ -f "${nemotron_config}" ]; then
      cp "${nemotron_config}" "${config}"
      info "Applied config-nemotron.yaml"
    else
      warn "config-nemotron.yaml not found, manual config update needed"
      info "Update chat_llm/summarization_llm/notification_llm in ${config}:"
      info "  model: nemotron-9b-jp-nothink"
      info "  base_url: http://host.docker.internal:11434/v1"
    fi
  fi

  info "Nemotron switch complete."
  info "Next: Re-ingest videos with Japanese VLM prompt and re-run queries."
  info ""
  info "To enable Japanese VLM captions, set before docker compose up:"
  info "  export VLM_SYSTEM_PROMPT=\"あなたは映像解析の専門家です。映像の内容を日本語で詳細に説明してください。\""
}

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
main() {
  local step="${1:-check}"

  case "${step}" in
    check)    step_check ;;
    login)    step_login ;;
    nim)      step_nim ;;
    vss)      step_vss ;;
    status)   step_status ;;
    stop)     step_stop ;;
    nemotron) step_nemotron ;;
    *)
      echo "Usage: $0 {check|login|nim|vss|status|stop|nemotron}"
      exit 1
      ;;
  esac
}

main "$@"
