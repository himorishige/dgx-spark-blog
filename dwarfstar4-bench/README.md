# DwarfStar 4 × DGX Spark 検証スクリプト

DevelopersIO 記事「[DwarfStar 4 で DeepSeek V4 Flash 284B を DGX Spark に載せてみた](https://dev.classmethod.jp/articles/dgx-spark-dwarfstar4-deepseek-v4-flash-bench/)」の検証スクリプトと生データ一式です。

- 推論エンジン: [antirez/ds4](https://github.com/antirez/ds4)（DwarfStar 4、commit `a97e7a3` = 2026-05-12）
- モデル: DeepSeek V4 Flash 284B、`q2-imatrix` GGUF（routed エキスパートだけ 2bit、約 81 GiB）
- 検証環境: NVIDIA DGX Spark（GB10 / sm_121 / 128 GB LPDDR5X 統合メモリ、~273 GB/s）、DGX OS Ubuntu 24.04 aarch64、CUDA 13.0、Driver 580.142

## ディレクトリ構成

```
dwarfstar4-bench/
├── README.md
├── scripts/
│   ├── README.md                # ローカル検証ディレクトリ向けの手順
│   ├── run-bench.sh             # ds4-bench フロンティアスイープラッパー（Phase 1）
│   ├── monitor-sys.sh           # free / nvidia-smi / iostat を並走サンプリング
│   ├── disk-kv-experiment.sh    # 長文スイープ + disk KV cold/warm 実験（Phase 2）
│   ├── bench-agent-task.py      # ds4-server に長文タスクを投げて prefill/decode 時間を計測（Phase 3）
│   └── make-charts.py           # 記事用 PNG 4 枚を生成（matplotlib + Noto Sans CJK JP）
└── data/
    ├── dgx-spark-q2-sweep.csv   # Phase 1: ctx 2K→65K の ds4-bench 出力（32 行）
    ├── dgx-spark-q2-longctx.csv # Phase 2a: ctx 65K→262K の長文スイープ（3 行）
    ├── disk-kv.json             # Phase 2b: cold 115.2s / warm 7.7s / 約 15 倍
    ├── agent-tasks.json         # Phase 3: コードレビュー 35k tok / 要約 56k tok の実測
    └── env.txt                  # commit hash / CUDA / Driver / モデルファイルサイズ
```

## 主要な実測値

| 指標                                | 値                                              |
| ----------------------------------- | ----------------------------------------------- |
| ビルド時間（`make` 一発、CUDA 13.0） | 約 16 秒                                        |
| モデルロード（80.76 GiB、cold）     | 26.4 秒                                         |
| 生成スループット（ctx 2K → 262K）   | 13.2 → 7.94 トークン/秒                         |
| prefill スループット（増分計測）    | 65 → 247 トークン/秒（ウォームアップ後）        |
| 圧縮 KV キャッシュサイズ            | 約 13.8 KB / token（100 万 tok 外挿 ≈ 12.8 GiB） |
| 動作中ピークメモリ（ctx 262K）      | 約 115 GiB（空き約 7 GiB）                      |
| Disk KV キャッシュ cold → warm      | 31,930 tok 再送が 115.2 秒 → 7.7 秒（約 15 倍） |
| コードレビュー（C ソース 35,088 tok） | 166.0 秒（うち prefill ≈ 264 tok/s）            |
| 連載記事要約（56,662 tok）          | 298.7 秒（うち prefill ≈ 238 tok/s）            |

## 再現手順（DGX Spark で）

```sh
# 1. ビルド
git clone https://github.com/antirez/ds4 ~/works/dwarfstar4/ds4
cd ~/works/dwarfstar4/ds4
export PATH=/usr/local/cuda/bin:$PATH
make
./download_model.sh q2-imatrix   # 約 81GiB

# 2. Phase 1: フロンティアスイープ
bash scripts/run-bench.sh

# 3. Phase 2: 長文 + disk KV キャッシュ
bash scripts/disk-kv-experiment.sh

# 4. Phase 3: エージェントタスク（別ターミナルで ds4-server を起動）
./ds4-server --ctx 200000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 24576 &
uv run --with httpx python scripts/bench-agent-task.py

# 5. 図の再生成
uv run --with matplotlib --with numpy python scripts/make-charts.py
```

Mac（M3 Max / M3 Ultra）の数値は本検証では取得しておらず、記事内の比較は DwarfStar 4 README のベンチ表からの引用です。

## ライセンス

スクリプトは MIT。生データは公開記事の補足としてそのまま参照可能です。検証対象の DwarfStar 4 本体・DeepSeek V4 Flash モデルのライセンスはそれぞれのリポジトリ／配布ページを参照してください。
