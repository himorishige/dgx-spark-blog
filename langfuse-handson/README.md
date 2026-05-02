# Langfuse Self-host Handson

DevelopersIO 記事「Langfuse を DGX Spark で Self-host して LLM アプリの観測と評価をひととおり試してみた」の検証スクリプト一式です。

- 記事 URL: 公開後追記
- 検証環境: NVIDIA DGX Spark (Grace Blackwell GB10 / 128GB unified memory) / ARM64
- バージョン: Langfuse server v3.172.1 / Python SDK v4.5.1

## ディレクトリ構成

```
langfuse-handson/
├── README.md
├── .env.example
├── .gitignore
└── scripts/
    ├── _general_qa.py             # 30 件 QA データ（共通モジュール）
    ├── 01_hello_smoke.py          # Self-host への疎通
    ├── 02_instrument_langchain.py # LangChain CallbackHandler
    ├── 03_otel_openinference.py   # OTel / OpenInference 経由
    ├── 04_sessions_demo.py        # マルチターン会話 + session_id
    ├── 05_upload_dataset.py       # Dataset 30 件投入
    ├── 06_run_dataset.py          # Dataset Run + 自動評価
    ├── 09_prompts_versioned.py    # Prompt v1/v2 + production label
    ├── 10_annotation_queue.py     # Annotation Queue
    ├── 11_vllm_local.py           # DGX Spark vLLM (Nemotron NVFP4) 接続
    └── start-vllm.sh              # vLLM 起動スクリプト
```

`07` / `08`（LLM-as-a-Judge）は Langfuse の Web UI から設定する手順を記事中で扱っているので、SDK スクリプトとしては用意していません。記事側を参照してください。

## セットアップ

### 1. Langfuse Self-host を起動

公式 `docker-compose.yml` を取得してから、本リポの `.env.example` をベースに `.env` を作成します。

```bash
cd langfuse-handson
curl -fsSL -o compose.yaml \
  https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml
cp .env.example .env
# .env の <CHANGEME> を全て書き換える
#   openssl rand -hex 32         # NEXTAUTH_SECRET / SALT / ENCRYPTION_KEY 用
#   openssl rand -base64 24 ...  # POSTGRES / CLICKHOUSE / REDIS / MINIO 用
docker compose up -d
```

`http://localhost:3000` を開いて UI が立ち上がっていれば OK です。詳しい初回ユーザ・API キーの自動初期化（`LANGFUSE_INIT_*`）は記事を参照してください。

### 2. Python 検証環境

```bash
uv venv --python 3.12
uv pip install \
  "langfuse>=4.5" \
  "langchain>=0.3" "langchain-anthropic" "langchain-openai" \
  "openinference-instrumentation-langchain" \
  "openinference-instrumentation-anthropic" \
  "arize-phoenix-otel"
```

API キー類は `.env` に記載されているので、`uv run --env-file=.env scripts/01_hello_smoke.py` のように実行してください。

### 3. 章 13 用：DGX Spark vLLM ローカル LLM

`11_vllm_local.py` を動かす場合は別途 vLLM を起動します。

```bash
./scripts/start-vllm.sh   # Nemotron 3 Nano 30B-A3B-NVFP4 を port 8001 で起動
```

詳細は `scripts/start-vllm.sh` のコメントと記事の章 13 を参照してください。

## ライセンス

スクリプト: MIT License
