# DGX Spark Blog

DGX Spark 関連の技術ブログ記事で使用したスクリプト・データを公開するリポジトリです。

記事は [DevelopersIO](https://dev.classmethod.jp/) で公開しています。

## Articles

| Directory | Article |
|-----------|---------|
| `n3-raft-finetuning/` | [国税庁 FAQ × RAFT で Nemotron 9B-v2 の RAG 精度を上げてみた](https://dev.classmethod.jp/articles/nemotron-9b-raft-finetuning/) |
| `n5-constitutional-ai-simpo/` | [クラスメソッドのカルチャー（CLP）を Nemotron 9B-v2 に教え込んでみた](https://dev.classmethod.jp/articles/nemotron-9b-constitutional-ai-simpo) |
| `n6-megatron-bridge/` | [Nemotron 9B × Megatron-Bridge で Mamba-2 含む全層 LoRA を NVIDIA Brev H100 で学習させてみた](https://dev.classmethod.jp/articles/nemotron-9b-megatron-bridge-brev) |
| `r1-isaac-sim-so-arm101/` | [DGX Spark でロボットアームの強化学習を試してみた（Isaac Sim + Isaac Lab + SO-ARM101）](https://dev.classmethod.jp/articles/dgx-spark-isaac-sim-so-arm101) |
| `b2-vlm-finetuning/` | [Cosmos-Reason2-8B を DGX Spark で PPE 検出向けにファインチューニングしてみた](https://dev.classmethod.jp/articles/dgx-spark-cosmos-reason2-vlm-finetuning/) |
| `sam1-zero-shot-detection/` | [SAM 3.1 を DGX Spark で動かしてゼロショット物体検出を試してみた](https://dev.classmethod.jp/articles/dgx-spark-sam3-zero-shot-detection/) |
| `v1-vss-agent/` | [DGX Spark で映像検索 AI エージェントを動かしてみた（VSS Agent）](https://dev.classmethod.jp/articles/dgx-spark-vss-agent) |
| `langfuse-handson/` | [Langfuse を DGX Spark で Self-host して LLM アプリの観測と評価を試してみた](https://dev.classmethod.jp/articles/langfuse-self-host-llm-observability-handson/) |
| `g4-mtp/` | [Gemma 4 MTP を DGX Spark で動かして日本語生成の高速化を実測してみた](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/) |
| `omni2-japanese-multimodal-bench/` | [Nemotron 3 Nano Omni / Gemma 4 / Cosmos-Reason2 を日本語マルチモーダルベンチで比べてみた](https://dev.classmethod.jp/articles/dgx-spark-nemotron3-nano-omni-japanese-multimodal-bench/) |
| `dwarfstar4-bench/` | [DwarfStar 4 で DeepSeek V4 Flash 284B を DGX Spark に載せてみた](https://dev.classmethod.jp/articles/dgx-spark-dwarfstar4-deepseek-v4-flash-bench/) |
| `chronos2-plc-sim/` | [Chronos-2 で PLC 風時系列データを予測し、Nemotron で保全コメントを生成してみた](https://dev.classmethod.jp/articles/dgx-spark-chronos2-plc-sim-llm-maintenance/) |

## Environment

- **Hardware**: NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory)
- **Container**: NGC NeMo (`nvcr.io/nvidia/nemo:25.11.01`)
- **Inference**: vLLM (nightly) / Ollama / GGUF

## License

Scripts: MIT License
Data: See individual README files for dataset licenses.
