# Omni2: 日本語マルチモーダル深掘り検証スクリプト

`workspace/blog/drafts/dgx-spark-nemotron3-nano-omni-japanese-multimodal-bench.md` の検証コード一式。

## モデル構成

| モデル | HF ID | 量子化 | 役割 |
| --- | --- | --- | --- |
| Nemotron 3 Nano Omni | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | NVFP4 (~21GB) | 速報 Omni1 の続編、3 つ巴の主役 |
| Gemma 4 31B IT | `nvidia/Gemma-4-31B-IT-NVFP4` | NVFP4 (~25GB) | 日本語強の Dense モデル |
| Cosmos-Reason2 | `nvidia/Cosmos-Reason2-8B` | BF16 (~16GB) | VLM 特化 8B、V3 / V4 / CR2 記事の実績 |

## ベンチマーク

| ベンチ | HF Dataset | 件数 | 評価方法 |
| --- | --- | --- | --- |
| Heron-Bench | `turing-motors/Japanese-Heron-Bench` | 102 | LLM-as-a-Judge (Claude Haiku 4.5、5 段階) |
| JMMMU | `JMMMU/JMMMU` | 1,090 | Exact Match (A/B/C/D 抽出) |

## venv 構成

| 用途 | venv |
| --- | --- |
| `vllm serve` 起動 | `~/works/private/workspace-dgx/workspace/scratchpad/projects/chocolate-factory-poc/infra/vllm/.venv` |
| `bench_*.py` / `make_charts.py` 実行 | `~/works/langfuse-handson/.venv`（langfuse 4.5.1 + anthropic 0.97.0 + matplotlib + datasets + pillow） |

## 実行手順

### 1. モデル + データセット DL（初回のみ）

```bash
hf download nvidia/Gemma-4-31B-IT-NVFP4
hf download nvidia/Cosmos-Reason2-8B
hf download turing-motors/Japanese-Heron-Bench --repo-type dataset
hf download JMMMU/JMMMU --repo-type dataset
```

### 2. Langfuse Dataset 登録（初回のみ）

```bash
cd workspace/blog/scripts/nemotron3-omni-omni2/
VENV_LF=~/works/langfuse-handson/.venv
$VENV_LF/bin/python bench_heron.py --upload
$VENV_LF/bin/python bench_jmmmu.py --upload  # 28 分野ロードに数分
```

### 3. モデルを 1 体ずつ起動 → ベンチ実行

DGX Spark UMA で 3 モデル同居不可。**順次** に切り替える。

```bash
# Day 1: Omni
./start-vllm-omni-nvfp4.sh   # 別ターミナルで起動、5 分待つ
$VENV_LF/bin/python bench_heron.py --model omni --limit 5     # smoke (judge コスト確認)
$VENV_LF/bin/python bench_heron.py --model omni
$VENV_LF/bin/python bench_jmmmu.py --model omni
# Ctrl+C で vllm 停止

# Day 2: Gemma 4
./start-vllm-gemma4.sh
$VENV_LF/bin/python bench_heron.py --model gemma4
$VENV_LF/bin/python bench_jmmmu.py --model gemma4

# Day 3: CR2
./start-vllm-cr2.sh
$VENV_LF/bin/python bench_heron.py --model cr2
$VENV_LF/bin/python bench_jmmmu.py --model cr2
```

### 4. グラフ生成

```bash
$VENV_LF/bin/python make_charts.py
# -> workspace/blog/images/nemotron3-omni-omni2/{heron-radar,jmmmu-bars-*,latency-comparison}.png
```

### 5. GitHub push（消失再発防止）

```bash
# 各バッチ後に
cd ~/works/private/workspace-dgx
git add workspace/blog/scripts/nemotron3-omni-omni2/
git commit -m "checkpoint: omni heron 50/102"

# 全完了後
gh repo clone himorishige/dgx-spark-blog /tmp/dgx-spark-blog
cp -r workspace/blog/scripts/nemotron3-omni-omni2 /tmp/dgx-spark-blog/
cd /tmp/dgx-spark-blog && git add . && git commit -m "feat: omni2 japanese multimodal bench" && git push
```

## ファイル一覧

| ファイル | 役割 |
| --- | --- |
| `start-vllm-omni-nvfp4.sh` | Omni NVFP4 起動 (port 8001、`--moe-backend flashinfer_cutlass`) |
| `start-vllm-gemma4.sh` | Gemma 4 31B IT NVFP4 起動 (Dense なので `--moe-backend` なし) |
| `start-vllm-cr2.sh` | Cosmos-Reason2-8B BF16 起動 |
| `lib_vllm_client.py` | OpenAI 互換クライアント + Langfuse `@observe` 統合 |
| `lib_judge.py` | Claude Haiku 4.5 judge (プロンプトキャッシング) |
| `bench_heron.py` | Heron-Bench 102 問ループ + LLM-as-Judge |
| `bench_jmmmu.py` | JMMMU 1,090 問ループ + Exact Match |
| `make_charts.py` | レーダー / 棒グラフ生成 |
| `results/{omni,gemma4,cr2}/{heron,jmmmu}/*.jsonl` | ローカルバックアップ (Langfuse outage 対策) |

## Langfuse の使い方

- **Dataset Run**: モデルごとに Run が立ち上がり、Experiments タブで side-by-side
- **Trace Cost**: Claude Haiku 4.5 の正規表現（`(?i)^claude-haiku-4-5(-\d+)?$`）が前回 Langfuse 記事で投入済み
- **Custom Evaluator (UI 設定)**: Heron は SDK 側でスコアを投入するため UI 設定は不要
- **Annotation Queue**: 章 7 の出力サンプル選定で活用

## 想定コスト

| 項目 | 金額 |
| --- | --- |
| Heron-Bench judge (Claude Haiku 4.5、102 問 × 3 モデル × プロンプトキャッシング) | ~$5 |
| JMMMU exact match (judge 不要) | $0 |
| **合計** | **~$5** |

## 既知のハマりポイント (DGX Spark + ARM64)

- `~/.cache/vllm/` が root 所有 → `VLLM_CACHE_ROOT=$HOME/.cache/vllm-local` で回避（NG2 知見）
- `--gpu-memory-utilization 0.9` は UMA で起動失敗 → 0.4 に下げる（chocolate-factory-poc 実績）
- vllm serve 並列起動不可 → 1 モデルずつ順次に
- Heron-Bench の列名は `text` または `question` で揺れ → `bench_heron.py` の `_detect_columns` で吸収
