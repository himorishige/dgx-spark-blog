# DGX Spark Blog

DGX Spark 関連の技術ブログ記事で使用したスクリプト・データを公開するリポジトリです。

記事は [DevelopersIO](https://dev.classmethod.jp/) で公開しています。

## Articles

| Directory | Article |
|-----------|---------|
| `n3-raft-finetuning/` | [国税庁 FAQ × RAFT で Nemotron 9B-v2 の RAG 精度を上げてみた](https://dev.classmethod.jp/articles/nemotron-9b-raft-finetuning/) |
| `n5-constitutional-ai-simpo/` | [クラスメソッドのカルチャー（CLP）を Nemotron 9B-v2 に教え込んでみた](https://dev.classmethod.jp/articles/nemotron-9b-constitutional-ai-simpo) |
| `n6-megatron-bridge/` | [Nemotron 9B × Megatron-Bridge で Mamba-2 含む全層 LoRA を NVIDIA Brev H100 で学習させてみた](https://dev.classmethod.jp/articles/nemotron-9b-megatron-bridge-brev) |

## Environment

- **Hardware**: NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory)
- **Container**: NGC NeMo (`nvcr.io/nvidia/nemo:25.11.01`)
- **Inference**: Ollama + GGUF

## License

Scripts: MIT License
Data: See individual README files for dataset licenses.
