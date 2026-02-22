# N3: Nemotron 9B-v2-Japanese RAFT Fine-Tuning

国税庁 FAQ（JaGovFaqs-22k）で Nemotron 9B-v2-Japanese を RAFT ファインチューニングし、RAG の精度を検証したスクリプトとデータです。

## Results

| Metric | Baseline | RAFT FT | Delta |
|--------|----------|---------|-------|
| F1 (test 200) | 0.565 | 0.654 | **+0.089** |
| False Positive (回答拒否) | 47 | 3 | **-93.6%** |
| JCQ (1,119) | 92.0% | 91.5% | -0.5pp |

## Environment

- **Training**: DGX Spark + NGC NeMo container (`nvcr.io/nvidia/nemo:25.11.01`)
- **Inference**: DGX Spark + Ollama (BF16 GGUF)
- **CoT Generation**: Claude Haiku (~$2)
- **Training Time**: 55 min (BF16 LoRA, 1,100 samples × 1 epoch)

## Scripts

| File | Description |
|------|-------------|
| `scripts/n3-raft-data-prep.py` | RAFT dataset preparation (multi-backend CoT generation) |
| `scripts/n3-nemo-train.py` | BF16 LoRA training with HF PEFT + TRL |
| `scripts/n3-evaluate.py` | Evaluation (EM/F1, unanswerable detection, JCQ) |
| `scripts/n3-gguf-convert.py` | LoRA → GGUF conversion + Ollama registration |
| `scripts/n3-ngc-inference-test.py` | NGC container inference test |
| `scripts/n3-plot-loss.py` | Training loss curve visualization |
| `scripts/requirements.txt` | Python dependencies |

## Data

| File | Description |
|------|-------------|
| `data/train.jsonl` | Training data (1,100 samples, RAFT format) |
| `data/test.jsonl` | Test data (200 samples, NTA) |
| `data/test_unseen.jsonl` | Unseen test data (100 samples, other ministries) |
| `data/nemotron-9b-raft-lora-ngc.gguf` | Trained LoRA adapter (GGUF, 35MB) |
| `data/Modelfile-ngc-nothink` | Ollama Modelfile (thinking OFF) |
| `data/results_*.json` | Evaluation results (baseline / RAFT FT) |
| `data/jcq_*.json` | JCommonsenseQA results |
| `data/training_metrics.json` | Training loss/accuracy per step |

## Quick Start

```bash
# 1. Prepare RAFT data (requires API key for CoT generation)
pip install -r scripts/requirements.txt
python scripts/n3-raft-data-prep.py all \
    --output-dir ./data \
    --copyright-filter "国税庁" \
    --backend anthropic

# 2. Train with NGC container
docker run --gpus all --rm --ipc=host \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/nemo:25.11.01 \
    bash -c 'pip install -q trl && python /workspace/scripts/n3-nemo-train.py \
        --data-file /workspace/data/train.jsonl \
        --output-dir /workspace/data/ngc-adapter'

# 3. Convert to GGUF and register with Ollama
python scripts/n3-gguf-convert.py all \
    --adapter-dir ./data/ngc-adapter/adapter \
    --llama-cpp-dir ~/llama.cpp \
    --base-model nemotron-9b-jp

# 4. Or skip training and use the pre-trained adapter directly
ollama create nemotron-9b-jp-raft -f data/Modelfile-ngc-nothink

# 5. Evaluate
python scripts/n3-evaluate.py rag --model nemotron-9b-jp-raft --data data/test.jsonl
python scripts/n3-evaluate.py jcq --model nemotron-9b-jp-raft
```

## Dataset License

- [JaGovFaqs-22k](https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k): CC BY 4.0
- Training data (RAFT format): Derived from JaGovFaqs-22k, CC BY 4.0
