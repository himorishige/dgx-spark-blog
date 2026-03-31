# SAM1: SAM 3.1 Zero-Shot Object Detection on DGX Spark

SAM 3.1 (Segment Anything with Concepts) を DGX Spark で動かし、テキストプロンプトによるゼロショット物体検出とセグメンテーションマスク可視化を試したスクリプトです。

## Results

5 ドメインでのゼロショット検出結果:

| Scene | Prompts | Detections | Elapsed |
|-------|---------|------------|---------|
| PPE (construction) | hard hat, person | 4 + 4 | 1,840ms |
| Warehouse | forklift, pallet, box | 8 | 773ms |
| Office | laptop, phone | 2 | 688ms |
| Street | person, car | 26 + 18 | 706ms |
| Supermarket | apple, banana | 39 + 17 | 702ms |

## Environment

- **Hardware**: NVIDIA DGX Spark (GB10, 128GB unified memory)
- **Container**: NGC PyTorch 26.03 (`nvcr.io/nvidia/pytorch:26.03-py3`)
- **Model**: [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (848M params)
- **VRAM**: ~3.3 GiB
- **flash-attn**: Jetson AI Lab sbsa wheel (`pypi.jetson-ai-lab.dev/sbsa/cu128`)

## Scripts

| File | Description |
|------|-------------|
| `Dockerfile` | SAM3 API server container (NGC PyTorch 26.03 + flash-attn sbsa) |
| `scripts/server.py` | FastAPI SAM3 API server (/detect, /segment endpoints) |
| `scripts/sam3-demo.py` | Zero-shot detection demo (text prompt → BBox + scores) |
| `scripts/sam3-visualize.py` | BBox visualization on images |
| `scripts/sam3-segment.py` | Segmentation mask visualization (color-coded overlay) |
| `scripts/sam3-video-bench.py` | SAM3 vs SAM3.1 video tracking benchmark |

## Usage

```bash
# 1. Build and run SAM3 API server
docker build -f Dockerfile -t sam3-api .
docker run --gpus all -p 8105:8105 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  sam3-api

# 2. Zero-shot detection
uv run scripts/sam3-demo.py image.jpg \
  --prompts "hard hat,safety vest,person" \
  --output result.json

# 3. BBox visualization
uv run scripts/sam3-visualize.py image.jpg result.json -o annotated.jpg

# 4. Segmentation mask visualization
uv run scripts/sam3-segment.py image.jpg \
  --prompts "hard hat,safety vest,person" \
  -o segmented.jpg

# 5. Video tracking benchmark (run inside container)
python scripts/sam3-video-bench.py video.mp4 \
  --prompts "person,hard hat" \
  --max-frames 30
```

## Notes

- SAM 3 のモデルは HuggingFace のゲート付きリポジトリです。`HF_TOKEN` の設定と、リポジトリの利用規約への同意が必要です
- DGX Spark (ARM64/SM121) では dtype 互換性パッチが必要です（`server.py` と `sam3-video-bench.py` に含まれています）
- SAM 3.1 の動画追跡（Multiplex Video Predictor）には flash-attn が必要です。画像検出のみなら flash-attn なしでも動作します
- ライセンスは Meta 独自の「SAM License」です。商用利用は可能ですが、軍事・核産業での使用は禁止されています

## Article

[SAM 3.1 を DGX Spark で動かしてゼロショット物体検出を試してみた](https://dev.classmethod.jp/articles/dgx-spark-sam3-zero-shot-detection/)
