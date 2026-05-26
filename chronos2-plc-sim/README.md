# Chronos-2 + PLC 風シミュレータ + Nemotron 保全コメント検証スクリプト

DevelopersIO 記事「[Chronos-2 で PLC 風時系列データを予測し、Nemotron で保全コメントを生成してみた](https://dev.classmethod.jp/articles/dgx-spark-chronos2-plc-sim-llm-maintenance/)」の検証コード一式。

DGX Spark（128GB UMA）1 台で「PLC 風データ生成 → Chronos-2 多変量予測 → 残差ベース異常スコア → ローカル LLM が日本語の保全コメント」をエンドツーエンドで動かす最小構成のリファレンス実装です。

## 構成

| 役割 | 採用モデル / バージョン |
| --- | --- |
| 時系列基盤モデル（28M） | `autogluon/chronos-2-small` |
| 時系列基盤モデル（120M） | `amazon/chronos-2` |
| 保全コメント生成 LLM | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`（Active 3B MoE / NVFP4 / 18GB） |
| シミュレータ | 自作 6 ブロック構成（FactoryState / LoadGenerator / EquipmentPhysics / SensorGenerator / FailureInjector / StreamPublisher） |

## venv 構成

| 用途 | venv |
| --- | --- |
| シミュレータ実行 + 図生成 + LLM クライアント | `./.venv`（`uv sync` で構築、numpy / pandas / matplotlib + 任意で httpx） |
| Chronos-2 推論 | 既存の Chronos-2 用 venv（`chronos` 2.x 系 + torch 2.12 + cu130）を直接呼ぶ |
| Nemotron 用 vLLM サーブ | 既存の vLLM 用 venv（vLLM 0.20+ / NVFP4 サポート）を直接呼ぶ |

GPU が必要なのは Chronos-2 推論と vLLM サーブの 2 か所だけです。シミュレータと図生成は CPU のみで動きます。

## 実行手順

### 1. シミュレータと目視プロット（GPU 不要）

```bash
uv sync
uv run python run_simulation.py --hours 72 --output data/sim_72h.csv --quiet
uv run python plot_timeline.py
```

シード固定なので Mac / Linux いずれでも同じ 259,200 行 CSV と 6 段タイムラインプロットが出ます。

### 2. Chronos-2 予測パイプライン（DGX Spark の GPU を使用）

`predict_pipeline.py` は Chronos-2 用 venv で直接呼ぶ前提です。SKAB ベンチで構築した同系の venv をそのまま流用できます。

```bash
PYTHONPATH=. <chronos2 venv>/bin/python predict_pipeline.py \
  --input data/sim_72h.csv \
  --model chronos2-28m \
  --positive-kinds spike
```

主なオプション:

- `--model chronos2-28m` / `chronos2-120m` — 28M small / 120M base
- `--positive-kinds spike` / `spike wear` — どのラベル種を陽性扱いするか
- `--max-windows 100` — smoke test
- `--aggregations mean max pca` — 集約戦略

出力:

- `data/predictions/{model}_predictions.npz` — X / Y / preds / labels / residual
- `data/predictions/{model}_summary.json` — AUC / F1 / latency / 閾値メトリクス

### 3. Nemotron 保全コメントパイプライン

別ターミナルで vLLM を起動します。langfuse 連載で整備した起動スクリプトと同じ系統です。

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --host 0.0.0.0 --port 8001 \
  --served-model-name nemotron-3-nano-nvfp4-local \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --moe-backend flashinfer_cutlass \
  --trust-remote-code
```

サーブが落ち着いたら、Chronos-2 推論結果から代表 6 ウィンドウを取って LLM にコメントを書かせます。

```bash
<chronos2 venv>/bin/pip install httpx  # 1 回だけ
<chronos2 venv>/bin/python comment_pipeline.py \
  --predictions data/predictions/chronos2-28m_predictions.npz \
  --top-k-positive 3 --top-k-negative 3 \
  --temperature 0.2 --repeats 1 --max-tokens 200
```

Nemotron 3 系は reasoning モードでは `</think>` まで内省を書く設計なので、短い保全コメントを返してほしいときは `--enable-thinking` を付けず（デフォルト OFF）に呼んでください。

### 4. 図の再生成

`make_charts.py` は `data/sim_72h.csv` と `data/predictions/chronos2-28m_predictions.npz` を読んで 3 枚の PNG を生成します。出力ディレクトリは記事の画像配置に合わせて調整してください。

## ファイル一覧

| ファイル | 役割 |
| --- | --- |
| `simulator.py` | 6 ブロック構成の PLC 風シミュレータ |
| `run_simulation.py` | シミュレータ CLI（CSV 出力） |
| `plot_timeline.py` | 6 段タイムラインプロット |
| `lib_chronos2_pipeline.py` | Chronos-2 multivariate 推論ラッパ |
| `lib_anomaly_score.py` | 残差スコア + 集約 + AUC / F1 メトリクス |
| `predict_pipeline.py` | 72h CSV → Chronos-2 → スコア → JSON / NPZ |
| `lib_llm_client.py` | vLLM `/v1/chat/completions` httpx クライアント |
| `comment_pipeline.py` | 代表 window 抽出 + LLM 呼び出し + JSONL 保存 |
| `make_charts.py` | 記事用の図生成（タイムライン / 予測 vs 実測 / 異常スコア） |

## 主要な発見（記事の山場）

- Chronos-2 28M は突発スパイクに非常に強い（spike-only AUC=0.999、MAR=0.000）
- 緩やかな drift は予測モデルが「予測の延長」として吸収してしまい、残差ベースでは検出困難（AUC≈0.51）
- より大きな 120M モデルが必ずしも有利ではない（28M F1=0.83 vs 120M F1=0.75 on spike）
- ローカル LLM の保全コメントは温度 0.2 + thinking OFF でかなり安定するが、z-score と物理単位の混同などは温度 0.7 で目立つ

## ライセンス

スクリプト類は MIT。Chronos-2 / Nemotron 3 Nano 30B-A3B-NVFP4 / vLLM はそれぞれのライセンスに従ってください。
