# DwarfStar 4 (DeepSeek V4 Flash) on DGX Spark — 検証スクリプト

ブログ記事「DwarfStar 4 (DeepSeek V4 Flash) を DGX Spark で動かしてみた」
（`workspace/blog/drafts/dgx-spark-dwarfstar4-deepseek-v4-flash-bench.md`）の
検証で使ったスクリプトと生データ。

## 対象

- エンジン: [antirez/ds4](https://github.com/antirez/ds4)（DwarfStar 4、commit は `results/env.txt` 参照）
- モデル: DeepSeek V4 Flash 284B-A13B MoE、`q2-imatrix` GGUF（routed expert のみ 2bit、約 81 GiB）
- ハード: DGX Spark（GB10 / sm_121 / aarch64）、128 GB LPDDR5x（実効 ~273 GB/s）、4 TB NVMe、DGX OS / Ubuntu 24.04 aarch64、CUDA 13.0

## 実機セットアップ（`~/works/dwarfstar4/`）

```sh
git clone https://github.com/antirez/ds4 ~/works/dwarfstar4/ds4
cd ~/works/dwarfstar4/ds4
export PATH=/usr/local/cuda/bin:$PATH
make                                  # CUDA ビルド（-arch=native で GB10 を自動認識）
./download_model.sh q2-imatrix        # 約 81 GiB を HF から取得 → ./ds4flash.gguf にリンク
```

## ファイル

| ファイル | 役割 |
| --- | --- |
| `run-bench.sh` | `ds4-bench` のフロンティアスイープラッパー（CSV 出力 + メモリログ並走） |
| `monitor-sys.sh` | `free` / `nvidia-smi` / `iostat` を並走サンプリング |
| `bench-agent-task.py` | `ds4-server` に長文プロンプトを投げて prefill/decode 時間と usage を計測（Phase 3） |
| `disk-kv-experiment.sh` | Disk KV キャッシュ有効時の cold prefill vs warm（cache hit）再 prefill を比較（Phase 2） |
| `make-charts.py` | `results/` から記事用 PNG 5 枚を生成（`uv run --with matplotlib --with numpy python make-charts.py`） |
| `results/` | `*-sweep.csv`（ds4-bench 出力）、`*-mem.log`、`summary.json`（手で集約した主要数値）、`env.txt` |

## 再現

```sh
# Phase 1: フロンティアスイープ
bash run-bench.sh

# Phase 2: 長文 + disk KV キャッシュ
bash disk-kv-experiment.sh

# Phase 3: エージェントタスク（別ターミナルで ds4-server を起動しておく）
./ds4-server --ctx 200000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 16384 &
uv run --with httpx python bench-agent-task.py

# 図の生成
uv run --with matplotlib --with numpy python make-charts.py
```

Mac（M3 Max / M3 Ultra）の数値は本検証では取得しておらず、記事内の比較は
DwarfStar 4 README のベンチ表からの引用です。
