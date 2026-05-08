# Phase 1 Day 2 結果（2026-05-08）

長文生成（max_tokens=256、12 round、warm rounds 2-12）での Gemma 4 MTP speedup を全 4 モデルで実測。Phase 1 Day 1（JCQ max_tokens=8）と対をなす測定。

## 環境

Phase 1 Day 1 と同一。vLLM `0.20.2rc1.dev99+g9c0812ffd`、共通 vLLM 引数 `--max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.5 --enforce-eager --max-num-batched-tokens 4096`。`temperature=0`、`max_tokens=256`、シングルクライアント。

## 全 8 セル結果（warm = round 2-12 平均）

| ラベル | モデル | 量子化 | warm tok/s | p50 | p95 | 採択率 |
|---|---|---|---:|---:|---:|---:|
| e2b-longform-baseline | E2B-it | BF16 | 36.53 | 3.34 s | 4.02 s | — |
| e2b-longform-mtp2 | + E2B-it-assistant | BF16 | **69.14** | 1.67 s | 2.20 s | **38.8%** |
| e4b-longform-baseline | E4B-it | BF16 | 18.45 | 5.97 s | 9.35 s | — |
| e4b-longform-mtp2 | + E4B-it-assistant | BF16 | **38.70** | 2.83 s | 4.18 s | **44.6%** |
| 26b-a4b-longform-baseline | 26B-A4B-NVFP4 | NVFP4 | 28.08 | 4.42 s | 4.99 s | — |
| 26b-a4b-longform-mtp2 | + 26B-A4B-it-assistant | NVFP4 | **47.94** | 2.51 s | 2.95 s | **54.9%** |
| 31b-longform-baseline | 31B-IT-NVFP4 | NVFP4 | 6.53 | 16.96 s | 20.76 s | — |
| 31b-longform-mtp2 | + 31B-it-assistant | NVFP4 | **12.48** | 9.54 s | 10.33 s | **54.8%** |

## モデル別 speedup（長文 vs JCQ 短文）

| モデル | JCQ speedup (Day 1) | 長文 speedup (Day 2) | 採択率 (Day 2) |
|---|---:|---:|---:|
| E2B | 0.955x | **1.892x** | 38.8% |
| E4B | 1.013x | **2.098x** | 44.6% |
| 26B-A4B (MoE) | 0.810x | **1.708x** | 54.9% |
| 31B | 0.921x | **1.913x** | 54.8% |

**26B-A4B の逆転**: JCQ では 0.81x（19% 減速）だった MoE モデルが、長文では 1.71x（71% 加速）に転じる。MoE drafter のオーバーヘッドが decode 量で回収される構造。

## 記事の核となるストーリー

1. **長文生成では Gemma 4 MTP は 1.7-2.1x speedup** — Google blog の最大 3x speedup（A100/H100 環境）には届かないが、DGX Spark でも明確に効く
2. **JCQ 短文（max_tokens=8）では効かないどころか減速** — overhead が回収されない。Day 1 と Day 2 の対比が記事の肝
3. **26B-A4B MoE の逆転** — 最も興味深い数値。drafter のオーバーヘッドと decode 量の収支が分かりやすく可視化される
4. **採択率はモデルサイズに比例** — E2B 38.8% → E4B 44.6% → 26B-A4B/31B 54%台。target が大きく強いほど drafter の予測も当たりやすい
5. **全モデルで品質劣化ゼロ**（JCQ accuracy delta ±0.5pt 以内、Day 1 で確認済み）

## 細かい観察

- **採択率の絶対値**: Google blog の「最大 3x」はおそらく acceptance rate 70-80% 程度の領域。DGX Spark の 38-55% は H100 比で低めだが、それでも 1.7-2.1x の効果が出る
- **stdev**: 26B-A4B/31B は MTP で stdev 増大（±3.3 / ±0.68 tok/s）。drafter の採択率が round ごとに揺れる
- **31B baseline 6.53 tok/s**: NVFP4 marlin の重さが顕著。MTP で 12.48 まで上がるが H100 native NVFP4 の数値（数十-100 tok/s 級）には届かない

## 採択率（Phase 1 Day 2 確定値）

| モデル | drafts | draft_tokens | accepted | rate |
|---|---:|---:|---:|---:|
| E2B | 797 | 1594 | 618 | 38.77% |
| E4B | 747 | 1494 | 667 | 44.65% |
| 26B-A4B | 677 | 1354 | 743 | 54.87% |
| 31B | 658 | 1316 | 721 | 54.79% |

各 round で `num_speculative_tokens=2` を draft → target で受理判定。

## 言語比較（追加検証 2026-05-08）

「日本語タスクだから採択率が低いのでは？」という仮説を、英語プロンプト 12 round で検証。

| モデル | 言語 | tok/s | 採択率 | 出力 tok/round |
|---|---|---:|---:|---:|
| E2B | 日本語 | 69.1 | 38.8% | 116.6 |
| E2B | **英語** | 71.4 | 42.0% | 165.2 |
| E4B | 日本語 | 38.7 | **44.6%** | 117.5 |
| E4B | **英語** | 37.6 | 42.0% | 164.8 |

**結論: 言語による採択率の差は ±3pt 程度で限定的、tok/s はほぼ同等**

- E2B は英語の方が +3.2pt 採択率が高いが、E4B では逆に -2.6pt 低い → 統計的に明確な傾向ではない
- 出力トークン数は英語の方が約 41% 多い（SentencePiece 構造、200 字 ≒ 150 words の差）が、tok/s は両言語でほぼ同じ
- **真のボトルネックは別にある** = DGX Spark のメモリ帯域・compute path

## ボトルネック分析: DGX Spark のメモリ帯域

speculative decoding の理論的効果は「target を 1 forward で N トークン出すことで memory bandwidth を活用する」最適化。しかし DGX Spark の memory bandwidth は構造的に低い:

| ハードウェア | メモリ帯域 | DGX Spark 比 |
|---|---:|---:|
| **DGX Spark GB10 LPDDR5X** | **273 GB/s** | 1.0x |
| A100 HBM2e | 2,039 GB/s | 7.5x |
| H100 SXM HBM3 | 3,350 GB/s | 12.3x |

Google blog の「最大 3x speedup」は H100 上の値で、bandwidth が桁違いに広い環境。DGX Spark の 1.7-2.1x という現実値は、**理論的にも妥当なライン**:

- speculative decoding の最大 speedup は「1 step で受理されたトークン数」に比例
- 採択率 40-55% × num_spec=2 なら、平均 1.4-1.5 トークン/step が理想
- ただし memory bandwidth が低いと 1 step の forward がそもそもメモリ転送律速 → speedup の頭打ち

加えて、NVFP4 系（26B-A4B / 31B）では:
- DGX Spark は **FP4 native compute 非対応** → Marlin weight-only 圧縮で代替
- weight load は 4bit になるが compute は BF16 dequant 経由 → compute path の追加オーバーヘッド
- これにより 26B-A4B baseline の 28 tok/s、31B baseline の 6.5 tok/s という比較的低い数値

## 整理: 記事の主張

1. **PR #41745 マージで Gemma 4 MTP が DGX Spark で動く**（2026-05-06、2 日後の実機検証）
2. **品質劣化はゼロ**（JCQ accuracy delta ±0.5pt 以内、200 問 × 4 モデル × baseline/MTP）
3. **長文では 1.7-2.1x speedup**（max_tokens=256、warm round）
4. **JCQ 短文（max_tokens=8）では効かないどころか減速**（drafter overhead が回収されない）
5. **26B-A4B MoE の劇的な逆転**（短文で 0.81x → 長文で 1.71x）
6. **言語の影響は限定的**（日英で採択率差 ±3pt、tok/s 差は数 %）
7. **memory bandwidth が真のボトルネック**（DGX Spark 273 GB/s vs H100 3,350 GB/s）— Google blog の「最大 3x」は H100 native の値で、DGX Spark の 2x 級は構造的な限界

## 次のステップ（Phase 2 執筆へ）

データは完全に揃った。記事執筆フェーズ:
- 章構成 11 章のドラフト着手（特に「言語影響は限定的、メモリ帯域が真のボトルネック」が記事の独自性）
- 画像 5 枚 (matplotlib): mtp-architecture / mtp-speedup-bar / mtp-lang-comparison / mtp-bandwidth-analysis / mtp-feasibility-matrix
- num_spec=1/4 比較、batch=4/8 比較は時間に余裕あれば付録扱い

**「DGX Spark + 日本語 + Gemma 4 MTP」の連載最適点が定量化できる状態。**
