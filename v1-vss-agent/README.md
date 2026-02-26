# V1: VSS Agent on DGX Spark

DGX Spark で NVIDIA VSS (Video Search and Summarization) Agent を動かした記事の関連スクリプト。

## 記事

- [DGX Spark で映像検索 AI エージェントを動かしてみた（VSS Agent）](https://dev.classmethod.jp/articles/dgx-spark-vss-agent/) _(公開後にリンク更新)_

## 構成

```
v1-vss-agent/
├── README.md
├── scripts/
│   ├── v1-vss-setup.sh          # セットアップ自動化スクリプト
│   └── v1-livestream-test.sh    # LIVE STREAM テスト（擬似 RTSP 配信）
└── config/
    └── config-nemotron.yaml     # Nemotron 9B-v2-Japanese 差し替え用 config
```

## Quick Start

```bash
# 1. 環境チェック
export NGC_API_KEY="your-key"
export HF_TOKEN="your-token"
bash scripts/v1-vss-setup.sh check

# 2. NGC ログイン
bash scripts/v1-vss-setup.sh login

# 3. NIM コンテナ起動（LLM / Embedding / Reranker）
bash scripts/v1-vss-setup.sh nim

# 4. VSS 本体デプロイ
bash scripts/v1-vss-setup.sh vss

# 5. 状態確認
bash scripts/v1-vss-setup.sh status

# Web UI: http://<DGX Spark IP>:9100
```

### Nemotron 差し替え（日本語検索）

```bash
# Llama 3.1 8B → Nemotron 9B-v2-Japanese に切り替え
bash scripts/v1-vss-setup.sh nemotron

# 日本語 VLM キャプション有効化（VSS 再起動前に設定）
export VLM_SYSTEM_PROMPT="あなたは映像解析の専門家です。映像の内容を日本語で詳細に説明してください。"

# VSS 再起動
bash scripts/v1-vss-setup.sh stop
bash scripts/v1-vss-setup.sh vss
```

## 環境

| 項目          | バージョン                       |
| ------------- | -------------------------------- |
| DGX Spark     | 128GB LPDDR5x, GB10             |
| DGX OS        | 7.2.3                           |
| GPU Driver    | 580.126.09                       |
| CUDA          | 13.0                            |
| VSS Agent     | 2.4.1                           |
| VLM           | Cosmos-Reason2-8B                |
| LLM (default) | Llama 3.1 8B (NIM DGX Spark)   |
| LLM (JP)      | Nemotron 9B-v2-Japanese (Ollama) |

### LIVE STREAM テスト（擬似 RTSP 配信）

カメラなしでライブストリーム機能を試すスクリプト。MediaMTX + FFmpeg で録画映像を RTSP 配信します。

```bash
# 1. RTSP サーバー起動 + ストリーム配信 + VSS 登録
bash scripts/v1-livestream-test.sh start ~/videos/vss-test/warehouse.mp4

# 2. ライブ要約開始（10 秒チャンク、30 秒ごとにサマリー集約）
bash scripts/v1-livestream-test.sh summarize

# 3. Q&A（別ターミナルで）
bash scripts/v1-livestream-test.sh query "What safety violations were detected?"

# 4. クリーンアップ
bash scripts/v1-livestream-test.sh stop
```

前提: VSS が稼働中、Docker と FFmpeg がインストール済み。

## DGX Spark Gotchas

- `IS_SBSA=1` 環境変数が ARM64 イメージ選択に必須
- NIM コンテナは DGX Spark 最適化版（`-dgx-spark` suffix）を使用
- `VLLM_GPU_MEMORY_UTILIZATION=0.4` が .env のデフォルト
- Nemotron 差し替え時は Ollama の OpenAI 互換 API（ポート 11434）を利用
