# Phase 0 結果（2026-05-07）

vLLM PR #41745 マージ翌日（保留判断の翌日）のフィージビリティ判定。

## 環境

- DGX Spark (GB10, SM121, aarch64, 128GB UMA)
- vLLM `0.20.2rc1.dev99+g9c0812ffd`（PR #41745 SHA `27e0057` 含む）
- torch 2.11.0+cu130, transformers 5.8.0, accelerate 1.13.0
- venv: `~/works/gemma4-mtp/.venv`

## E2B BF16 (target) — Phase 0 単発検証

target = `google/gemma-4-E2B-it`、drafter = `google/gemma-4-E2B-it-assistant`。
`max_model_len=8192`、`max_num_seqs=4`、`gpu_memory_utilization=0.5`、`enforce-eager`、`temperature=0.0`、`max_tokens=256`、シングルクライアント。

| 設定 | mean tok/s | warm tok/s (rounds 2-5) | speedup (warm) | model load | init engine |
|---|---|---|---|---|---|
| baseline (target only) | 36.59 | 37.67 | 1.00x | 71.0s / 9.79 GiB | 3.85s |
| MTP `num_spec=2` | 57.78 | 73.10 | **1.94x** | 65.0s / 9.96 GiB | 41.80s |

ログ: `~/works/gemma4-mtp/logs/vllm-e2b-{baseline,mtp}-phase0.log`

## PR #41745 の実機での挙動確認

ログから明示的に確認できた事項:

1. **multimodal バイパス**: `[llm_base_proposer.py:1375] Draft model does not support multimodal inputs, falling back to text-only mode`
   - target は `Gemma4ForConditionalGeneration`（multimodal）のまま
   - drafter は `Gemma4MTPModel` で text-only に降格、`_raise_if_multimodal()` を迂回
2. **centroids masking 適用**: `[gemma4_mtp.py:536] Gemma4 MTP: centroids masking enabled (num_centroids=2048, top_k=32, active_tokens=4096/262144)`
   - lm_head 計算が 262K → 4K トークンに削減（PR の主要最適化）
3. **embedding 共有**: `Detected MTP model. Sharing target model embedding weights with the draft model`
4. **drafter 構造**: 4 層、3 層が target layer 13 (sliding_attention)、1 層が target layer 14 (full_attention) にバインド
5. **CUDA graph キャプチャ**: `captured centroids CUDA graphs for sizes [1, 2, 4, 8, 16, 32, 64]`
6. **MTP num_spec>1 警告**: `Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate` — Phase 1 で num_spec=1 vs 2 vs 4 を実機比較する材料に

## attention backend の固有事情

`Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.`

→ Gemma 4 は層によって head_dim が違うため、Flash/CUDA attention ではなく Triton attention 強制。記事のハマりポイント章ネタ。

## Phase 1 着手判定

✅ **Go**: speedup 1.94x、品質維持（応答内容も一貫）、安定起動。

## Phase 1 への TODO

1. E4B-it（HF cache 未取得）の取得 + smoke
2. 26B-A4B（NVFP4）+ drafter（BF16）+ smoke
3. 31B（NVFP4）+ drafter（BF16）+ smoke
4. JCQ 200 問ベンチマークスクリプト（`g4-benchmark.py` 流用）
5. num_spec=1/2/4 の比較（PR の警告検証）
6. batch_size=1/4/8（26B-A4B のみ、MoE drafter b=1 警告検証）
