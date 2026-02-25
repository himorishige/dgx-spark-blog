# N6: Nemotron 9B × Megatron-Bridge で Mamba-2 含む全層 LoRA 学習

HuggingFace PEFT では Mamba-2 層に LoRA を適用できず 53% カバーに留まっていた Nemotron 9B-v2-Japanese に対し、NGC NeMo コンテナの Megatron-Bridge v0.2.0 を使って 100% LoRA カバーを実現し、NVIDIA Brev のクラウド H100 で学習を実行したスクリプトとデータです。

## Results

### JCommonsenseQA（一般知識退行テスト）

| Model | Accuracy | Delta |
|-------|----------|-------|
| Baseline（FT なし） | 91.96%（1029/1119） | - |
| RAFT FT（HF PEFT / 53% LoRA） | 91.51%（1024/1119） | -0.45pp |
| RAFT FT（Megatron-Bridge / 100% LoRA） | 92.14%（1031/1119） | **+0.18pp** |

### RAFT ドメイン F1 評価（国税庁 FAQ 200 件）

| Model | F1 | False Positive (回答拒否) |
|-------|-----|--------------------------|
| Baseline（FT なし） | 0.5646 | 47 |
| RAFT FT（HF PEFT / 53% LoRA） | 0.6536 | 3 |
| RAFT FT（Megatron-Bridge / 100% LoRA） | 0.4884 | 69 |

100% LoRA にしたにもかかわらず F1 はベースラインを下回り、回答拒否が急増するという negative result。ただし回答済みサンプルに限定した F1 は 0.7457 と HF PEFT（0.6636）を上回っており、「回答の質は上がったが拒否しすぎ」という結果。500 iter ≈ 3.6 epochs で「回答不可」パターンを過学習した可能性が高い。

## Environment

- **Training**: NVIDIA Brev 1xH100 PCIe 80GB ($1.90/hr)
- **Container**: NGC NeMo (`nvcr.io/nvidia/nemo:25.11.01`, Megatron-Bridge v0.2.0)
- **Inference**: DGX Spark + Ollama (GGUF Q4_K_M)
- **Training Time**: ~12 min (500 iter, LoRA rank=32, seq_len=2048)
- **Total Cost**: ~$3-12 (Brev instance)

## Scripts

| File | Description |
|------|-------------|
| `scripts/n6-brev-run.sh` | Full Megatron-Bridge LoRA training pipeline (5 phases, single command) |
| `scripts/n6-brev-dryrun.sh` | Dry run for Phase 1-2 only (environment check + checkpoint conversion) |
| `scripts/n6-export-hf.py` | Megatron checkpoint → HuggingFace format export (standalone, 4 fallback strategies) |
| `scripts/n6-plot-loss.py` | Training loss curve visualization |
| `scripts/n6-eval-charts.py` | Evaluation result charts (F1 comparison + F1 paradox) |

### n6-brev-run.sh Phases

| Phase | Description |
|-------|-------------|
| 1 | Environment validation (GPU, Megatron-Bridge, NeMo recipe) |
| 2 | Data validation + HF → Megatron checkpoint conversion (`import_ckpt`) |
| 3 | 100% LoRA training with `nemotron_nano_9b_v2_finetune_config` recipe |
| 4 | Megatron → HF export (`export_ckpt`) |
| 5 | Archive creation for download to DGX Spark |

Each phase can be run independently with `--phase N`.

## Data

| File | Description |
|------|-------------|
| `data/train.jsonl` | Training data (1,100 samples, RAFT format, same as N3) |

## Quick Start

```bash
# 1. Launch Brev VM Mode instance (1xH100 PCIe 80GB)
# 2. Pull NeMo container
docker pull nvcr.io/nvidia/nemo:25.11.01

# 3. Upload scripts and data to /ephemeral/n6-megatron-bridge/

# 4. Run full pipeline
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 -v /ephemeral:/workspace \
    nvcr.io/nvidia/nemo:25.11.01 \
    bash /workspace/n6-megatron-bridge/scripts/n6-brev-run.sh

# 5. Download HF export to DGX Spark and convert to GGUF
# (see article for GGUF conversion and Ollama registration steps)
```

## Known Issues

- `multiprocessing.Manager().Queue()` が Docker コンテナ内で EOFError を起こすため、`mp.Queue()` へのモンキーパッチで回避
- Megatron-Bridge の eval は `global_batch_size` 未満のデータでクラッシュするため、`evaluate_and_print_results` を no-op 化して回避
- `export_ckpt` は LoRA delta のみの学習チェックポイントを扱うため、`source_path=` でベース+学習チェックポイント両方の指定が必要

## Dataset License

- Training data (RAFT format): Derived from [JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k), CC BY 4.0
