# Phase 1 結果（2026-05-07）

vLLM PR #41745（gemma4_mtp）後、Gemma 4 全 4 モデル × baseline/MTP num_spec=2 の JCQ 200 問ベンチマーク。

## 環境

- DGX Spark (GB10, SM121, aarch64, 128GB UMA)
- vLLM `0.20.2rc1.dev99+g9c0812ffd`
- torch 2.11.0+cu130 / transformers 5.8.0
- 共通 vLLM 引数: `--max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.5 --enforce-eager --max-num-batched-tokens 4096`
- JCQ (JCommonsenseQA v1.1) validation 1,119 問から seed=42 で先頭 200 問抽出（3-shot プロンプト、`temperature=0`、`max_tokens=8`）

## 全 8 セル結果

| ラベル | モデル | 量子化 | accuracy | parsed | p50 | p95 | mean tps | 200 問所要 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| e2b-baseline | google/gemma-4-E2B-it | BF16 | 0.875 | 1.000 | 82.8 ms | 84.2 ms | 24.09 | 16.6 s |
| e2b-mtp2 | + E2B-it-assistant | BF16 | 0.880 | 1.000 | 82.7 ms | 85.1 ms | 23.00 | 17.4 s |
| e4b-baseline | google/gemma-4-E4B-it | BF16 | 0.930 | 1.000 | 160.2 ms | 162.8 ms | 11.95 | 33.5 s |
| e4b-mtp2 | + E4B-it-assistant | BF16 | 0.930 | 1.000 | 154.7 ms | 157.3 ms | 12.11 | 33.0 s |
| 26b-a4b-baseline | nvidia/Gemma-4-26B-A4B-NVFP4 | NVFP4 | 0.975 | 1.000 | 127.7 ms | 131.7 ms | 15.23 | 26.3 s |
| 26b-a4b-mtp2 | + 26B-A4B-it-assistant | NVFP4 | 0.970 | 1.000 | 160.5 ms | 164.3 ms | 12.33 | 32.4 s |
| 31b-baseline | nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | 0.980 | 1.000 | 476.6 ms | 480.3 ms | 4.17 | 96.0 s |
| 31b-mtp2 | + 31B-it-assistant | NVFP4 | 0.980 | 1.000 | 519.2 ms | 526.6 ms | 3.84 | 104.2 s |

## モデル別 speedup（JCQ 短文、max_tokens=8）

| モデル | tps speedup | p50 speedup | accuracy delta | 評価 |
|---|---:|---:|---:|---|
| E2B | 0.955x | 1.001x | +0.005 | ほぼ同等（drafter overhead で軽微減速） |
| E4B | 1.013x | 1.036x | +0.000 | わずかに改善（noise 範囲） |
| 26B-A4B | **0.809x** | 0.795x | -0.005 | **MoE drafter b=1 で 19% 減速**（PR 警告通り） |
| 31B | 0.921x | 0.918x | +0.000 | 8% 減速 |

## キーストーリー

**JCQ (max_tokens=8) では MTP の旨味は出ない**。これは PR コメントの公式注意点とも整合し、論文の主張（speculative decoding は decode 量が多いほど効く）の実機再現になる。

特筆ポイント:

1. **品質劣化はすべて accuracy ±0.5pt 以内**（200 問で 1 問差） — Gemma 4 MTP の品質維持は実機確認できた
2. **26B-A4B（MoE）で MTP が逆効果** — Active 3.8B drafter を加えるオーバーヘッドが、短文生成では回収されない。PR の `b=1 で旨味出ない` 警告が DGX Spark 単一クライアント環境で再現
3. **Phase 0 の長文 (max_tokens=256) では E2B で 1.94x speedup** — JCQ と長文生成で真逆の結果が出るのが本記事の核心

## DGX Spark 固有の挙動

- **NVFP4 量子化**: 26B-A4B は **MARLIN MoE backend**、31B は **FlashInferCutlass NVFP4 GEMM kernel** を採用（モデル種別で path 分岐）
- DGX Spark は FP4 native compute 非対応 → Marlin が weight-only 圧縮で動作。`Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.` という警告
- TRITON_ATTN backend forced（Gemma 4 head_dim 不均質）
- multimodal target + text-only drafter のバイパスは全モデルで動作（draft 段階で image/audio=0）
- centroids masking で lm_head を 262K → 4K トークンに削減（PR の主要最適化）

## メモリ

| モデル | weight only | total (KV cache 含む) |
|---|---:|---:|
| E2B BF16 | 9.79 GiB | ~57 GiB |
| E2B BF16 + drafter | 9.96 GiB | ~58 GiB |
| E4B BF16 | 14.89 GiB | — |
| E4B BF16 + drafter | 15.34 GiB | ~67 GiB |
| 26B-A4B NVFP4 | 17.50 GiB | 18.12 GiB |
| 26B-A4B NVFP4 + drafter | 18.12 GiB | 18.9 GiB |
| 31B NVFP4 | 30.39 GiB | 31.18 GiB |
| 31B NVFP4 + drafter | 31.18 GiB | 32.06 GiB |

drafter 追加コスト: E2B/E4B で 0.18 GiB、26B-A4B で 0.81 GiB、31B で 0.87 GiB

## ハマりポイント（記事素材）

1. `max_num_batched_tokens=2048` がデフォルト → Gemma 4 multimodal token (2496) と衝突。`--max-num-batched-tokens 4096` に上げる（前例: Omni2 検証）
2. NVFP4 だが target HF レポは `nvidia/Gemma-4-{26B-A4B,31B-IT}-NVFP4` を指す必要（`google/gemma-4-31B-it` BF16 は 65 GB で 128 GB UMA を超える）
3. drafter は `google/gemma-4-{size}-it-assistant` を指定 — `gemma4_assistant` model_type が vLLM 内で `gemma4_mtp` に自動 rewrite

## 次のステップ

Phase 2（執筆）に向けて:
- 長文 (max_tokens=256) での speedup を取り直す（Phase 0 の E2B 1.94x のような数値を全モデルで）
- num_spec=1 vs 2 vs 4 比較（時間に余裕があれば）
- batch=1/4/8 比較（26B-A4B のみ、MoE drafter スケーリング検証）
- 採択率取得（vLLM ログから抽出 or vllm metrics endpoint）
- 画像 5 枚生成（matplotlib）
