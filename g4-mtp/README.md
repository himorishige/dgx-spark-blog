# Gemma 4 MTP × DGX Spark 検証スクリプト

DevelopersIO 記事「[Gemma 4 MTP を DGX Spark で動かして日本語生成の高速化を実測してみた](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/)」の検証スクリプト・結果データ一式です。

- 検証環境: NVIDIA DGX Spark (Grace Blackwell GB10 / 128GB unified memory) / ARM64
- vLLM: `0.20.2rc1.dev99+g9c0812ffd`（[PR #41745](https://github.com/vllm-project/vllm/pull/41745) 後の nightly、cu130 wheel）
- 主役: Gemma 4 全 4 モデル（E2B / E4B / 26B-A4B / 31B）+ それぞれの drafter（`*-it-assistant`）

## ディレクトリ構成

```
g4-mtp/
├── README.md
├── scripts/
│   ├── start-vllm-gemma-mtp.sh   # vLLM 起動（baseline / MTP 切替、4 モデル切替）
│   ├── smoke-test.py              # 軽量 chat completion で tok/s 計測（Phase 0 用）
│   ├── bench-jcq.py               # JCommonsenseQA 200 問で正答率・tok/s を収集
│   ├── bench-longform.py          # 長文生成（max_tokens=256、12 round）で tok/s と受け入れ率
│   ├── make-charts.py             # matplotlib で記事の画像 5 枚を生成
│   ├── run-phase1-cell.sh         # JCQ ベンチ用の自動化（起動 → bench → 停止）
│   ├── run-longform-cell.sh       # 長文ベンチ用の自動化
│   ├── PHASE0_RESULTS.md          # Phase 0 (E2B BF16 動作確認、1.94x speedup) の結果
│   ├── PHASE1_RESULTS.md          # Phase 1 Day 1 (JCQ 200 問 × 全 4 モデル × baseline/MTP) の結果
│   └── PHASE1_DAY2_RESULTS.md     # Phase 1 Day 2 (長文 + 言語比較) の結果
└── data/
    ├── *-baseline.jsonl           # JCQ ベースラインの問単位 raw データ
    ├── *-baseline.summary.json    # JCQ ベースラインの集計（accuracy、p50/p95、tok/s）
    ├── *-mtp2.jsonl               # JCQ MTP num_spec=2 の raw データ
    ├── *-mtp2.summary.json        # JCQ MTP num_spec=2 の集計
    ├── *-longform-baseline.long.{jsonl,summary.json}  # 長文ベースライン
    ├── *-longform-mtp2.long.{jsonl,summary.json}      # 長文 MTP num_spec=2
    └── *-longform-mtp2-en.long.{jsonl,summary.json}   # 長文 MTP（英語比較用）
```

## 使い方

### 環境準備

```bash
# uv で venv を作成（Python 3.12 推奨）
uv venv --python 3.12
source .venv/bin/activate

# vLLM nightly + cu130 backend
uv pip install --pre --upgrade vllm \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --torch-backend=cu130

# ベンチ用の周辺ライブラリ
uv pip install datasets accelerate
```

### サーバ起動

```bash
# E2B + drafter で MTP 有効化（num_spec=2）
bash scripts/start-vllm-gemma-mtp.sh e2b mtp:2

# 別ターミナルで smoke test
python scripts/smoke-test.py --model gemma4-e2b --rounds 5
```

引数の組み合わせ:

| 引数        | 値                              |
| ----------- | ------------------------------- |
| size        | `e2b` / `e4b` / `26b-a4b` / `31b` |
| mode        | `baseline` / `mtp:1` / `mtp:2` / `mtp:4` |

### Phase 1（JCQ 200 問）の自動実行

```bash
# baseline
bash scripts/run-phase1-cell.sh e2b baseline e2b-baseline

# MTP num_spec=2
bash scripts/run-phase1-cell.sh e2b mtp:2 e2b-mtp2
```

結果は `data/{label}.jsonl` + `data/{label}.summary.json` に保存されます。

### 長文ベンチ（max_tokens=256、12 round）の自動実行

```bash
bash scripts/run-longform-cell.sh e2b baseline e2b-longform-baseline
bash scripts/run-longform-cell.sh e2b mtp:2 e2b-longform-mtp2
```

### 記事用の画像生成

```bash
python scripts/make-charts.py --data-dir data --out-dir ../images/gemma4-mtp
```

## 主要結果

長文生成（max_tokens=256、warm rounds 2-12 平均、temperature=0、シングルクライアント）:

| モデル          | baseline tok/s | MTP tok/s | 高速化倍率 | 受け入れ率 |
| --------------- | -------------- | --------- | ---------- | ---------- |
| E2B (BF16)      | 36.5           | 69.1      | 1.89x      | 38.8%      |
| E4B (BF16)      | 18.5           | 38.7      | 2.10x      | 44.6%      |
| 26B-A4B (NVFP4) | 28.1           | 47.9      | 1.71x      | 54.9%      |
| 31B (NVFP4)     | 6.5            | 12.5      | 1.91x      | 54.8%      |

短文（JCQ、max_tokens=8）では全モデル ~1.0x 以下、26B-A4B は 0.81x で減速。**26B-A4B が短文 0.81x → 長文 1.71x に逆転** するのが記事の核です。詳細は [DevelopersIO 記事](https://dev.classmethod.jp/articles/dgx-spark-gemma4-mtp-multi-token-prediction-bench/) を参照してください。

## 関連リンク

- [vLLM PR #41745: Add Gemma4 MTP speculative decoding support](https://github.com/vllm-project/vllm/pull/41745)
- [Google: Multi-Token Prediction for Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [vLLM Recipes: Gemma 4 26B-A4B](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it)
- [HuggingFace: Gemma 4 collection](https://huggingface.co/collections/google/gemma-4-686b40235a0b0dcfe80e1e5a)
- [JCommonsenseQA v1.1](https://huggingface.co/datasets/leemeng/jcommonsenseqa-v1.1)
