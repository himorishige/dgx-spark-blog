# B2: Cosmos-Reason2-8B VLM LoRA Fine-Tuning for PPE Detection

SH17 PPE データセット（YOLO 形式）を VLM 学習データに変換し、Cosmos-Reason2-8B を LoRA SFT で PPE 検出に特化させたスクリプトです。

## Results

| Metric | Base | SFT (LoRA) | Delta |
|--------|------|------------|-------|
| PPE Violation Detection | 46.7% | **90.0%** | **+43.3%** |
| `<think>` Tag Usage | 0% | 100% | — |
| Avg Inference Time | 47.1s | 53.2s | +6.1s |

## Environment

- **Hardware**: NVIDIA DGX Spark (GB10, 128GB unified memory)
- **CUDA**: 13.0 (SM 12.1)
- **Framework**: PyTorch 2.9.0+cu130, TRL 0.26.1, PEFT 0.18.0
- **Model**: [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)
- **Training Time**: 14h 36m (300 steps, BF16 LoRA rank=32)
- **Peak GPU Memory**: 34.5 GB / 128 GB (27%)

## Scripts

| File | Description |
|------|-------------|
| `scripts/convert_sh17_to_vlm.py` | SH17 YOLO annotations → VQA conversation pairs (HF Datasets) |
| `scripts/train_lora_sft.py` | BF16 LoRA SFT with TRL SFTTrainer (DGX Spark optimized) |
| `scripts/eval_base_vs_sft.py` | Base vs SFT comparison on test set (30 samples) |

## Usage

```bash
# 1. Setup (inside cosmos-reason2 repo)
git clone https://github.com/nvidia-cosmos/cosmos-reason2
cd cosmos-reason2/examples/notebooks
# Edit pyproject.toml: cu128_torch28 → cu130_torch29
uv sync

# 2. Convert SH17 dataset
uv run python convert_sh17_to_vlm.py \
  --sh17-dir /path/to/sh17 \
  --output-dir ./data/sh17_vlm

# 3. Train (smoke test)
uv run python train_lora_sft.py --max-steps 10

# 4. Train (full)
uv run python train_lora_sft.py \
  --max-steps 300 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 2e-4 \
  --lora-rank 32

# 5. Evaluate
uv run python eval_base_vs_sft.py --num-samples 30
```

## Notes

- DGX Spark (ARM64) では flash-attn の公式ホイールがないため、`attn_implementation="sdpa"` を使用しています。[Jetson AI Lab の sbsa wheel](https://pypi.jetson-ai-lab.dev/sbsa/cu128) で flash-attn をインストールできることは確認済みですが、長時間学習での安定性は未検証です
- VLM は画像ごとにトークン数が異なるため、バッチサイズを上げてもパディングオーバーヘッドで逆に遅くなります。`batch_size=1` + `gradient_accumulation` が最速でした
- bitsandbytes (QLoRA) は ARM64 + CUDA 13.0 での動作が未保証のため、量子化なしの BF16 LoRA を採用しています

## Dataset

- [SH17 PPE Dataset](https://arxiv.org/abs/2407.04590) — 建設現場の PPE 装着状態を 17 クラスでアノテーション
- Train: 6,437 / Val: 804 / Test: 803 (PPE アノテーションを含む画像のみ)

## Article

[Cosmos-Reason2-8B を DGX Spark で PPE 検出向けにファインチューニングしてみた](https://dev.classmethod.jp/articles/dgx-spark-cosmos-reason2-vlm-finetuning/)
