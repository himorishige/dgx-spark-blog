# DGX Spark Blog

DGX Spark 関連の技術ブログ記事で使用したスクリプト・データを公開するリポジトリです。

記事は [DevelopersIO](https://dev.classmethod.jp/) で公開しています。

## Articles

| Directory | Article | Status |
|-----------|---------|--------|
| `n3-raft-finetuning/` | Nemotron 9B-v2-Japanese を国税庁 FAQ でファインチューニングして RAG の精度を上げてみた | Draft |
| `n5-constitutional-ai-simpo/` | クラスメソッドの行動規範を Nemotron 9B-v2 に教え込んでみた | Draft |

## Environment

- **Hardware**: NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified memory)
- **Container**: NGC NeMo (`nvcr.io/nvidia/nemo:25.11.01`)
- **Inference**: Ollama + GGUF

## License

Scripts: MIT License
Data: See individual README files for dataset licenses.
