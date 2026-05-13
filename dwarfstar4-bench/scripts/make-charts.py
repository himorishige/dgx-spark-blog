#!/usr/bin/env python3
"""Generate charts for the DwarfStar 4 (DeepSeek V4 Flash) on DGX Spark article.

Reads ./results/*.csv and ./results/*.json (measured on a DGX Spark GB10, 128GB,
DwarfStar 4 commit a97e7a3, DeepSeek-V4-Flash q2-imatrix GGUF) and writes PNGs to
workspace/blog/images/dwarfstar4/:

  - ds4-context-vs-throughput.png  : ds4-bench sweep, prefill & decode t/s vs ctx (0..262k)
  - ds4-kvcache-vs-context.png     : compressed KV cache bytes vs ctx, extrapolated to 1M
  - ds4-prefill-decode-bar.png     : DGX Spark vs Mac M3 Max vs Mac M3 Ultra (q2, ~12k-tok prompt)
  - ds4-diskkv-cold-vs-warm.png    : 32k-token prompt: cold prefill vs disk-cache warm hit
  - ds4-feasibility-matrix.png     : DGX Spark DwarfStar 4 feasibility table

Run:  uv run --with matplotlib --with numpy python make-charts.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 130

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
IMG = HERE.parent.parent / "images" / "dwarfstar4"
IMG.mkdir(parents=True, exist_ok=True)

C_PREFILL = "#2563eb"
C_DECODE = "#dc2626"
C_KV = "#0ea5e9"
C_MUTED = "#94a3b8"
C_OK = "#16a34a"
C_WARN = "#d97706"
C_BAD = "#dc2626"
C_COLD = "#64748b"
C_WARM = "#16a34a"


def read_bench_csv(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append(
                {
                    "ctx": int(r["ctx_tokens"]),
                    "prefill_tps": float(r["prefill_tps"]),
                    "gen_tps": float(r["gen_tps"]),
                    "kv_bytes": int(r["kvcache_bytes"]),
                }
            )
    return rows


def load_sweep():
    rows = read_bench_csv(RESULTS / "dgx-spark-q2-sweep.csv")
    # append the long-context coarse sweep (skip the duplicated 65536 row)
    for r in read_bench_csv(RESULTS / "dgx-spark-q2-longctx.csv"):
        if r["ctx"] > rows[-1]["ctx"]:
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
def chart_context_vs_throughput():
    rows = load_sweep()
    ctx = [r["ctx"] / 1000 for r in rows]
    pf = [r["prefill_tps"] for r in rows]
    gen = [r["gen_tps"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    l1, = ax1.plot(ctx, pf, "-o", color=C_PREFILL, ms=4, label="prefill（増分スループット）")
    ax1.set_xlabel("コンテキスト長（千トークン）")
    ax1.set_ylabel("prefill スループット（トークン/秒）", color=C_PREFILL)
    ax1.tick_params(axis="y", labelcolor=C_PREFILL)
    ax1.set_ylim(0, max(pf) * 1.2)
    ax1.grid(alpha=0.25)
    ax1.axvspan(0, 30, color="#fef3c7", alpha=0.5, zorder=0)
    ax1.text(1.5, max(pf) * 1.1, "ウォームアップ域", fontsize=8, color="#92400e")

    ax2 = ax1.twinx()
    l2, = ax2.plot(ctx, gen, "-s", color=C_DECODE, ms=4, label="生成（decode）")
    ax2.set_ylabel("生成スループット（トークン/秒）", color=C_DECODE)
    ax2.tick_params(axis="y", labelcolor=C_DECODE)
    ax2.set_ylim(0, max(gen) * 1.5)

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="center right", framealpha=0.95)
    ax1.set_title("DGX Spark / DwarfStar 4 — コンテキスト長と prefill・生成スループット（q2-imatrix）")
    fig.tight_layout()
    out = IMG / "ds4-context-vs-throughput.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def chart_kvcache_vs_context():
    rows = load_sweep()
    ctx = np.array([r["ctx"] for r in rows], dtype=float)
    kv_gib = np.array([r["kv_bytes"] for r in rows], dtype=float) / (1024**3)
    a, b = np.polyfit(ctx, kv_gib, 1)
    xx = np.array([0.0, 1_000_000.0])
    yy = a * xx + b
    kv_1m = a * 1_000_000 + b

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ctx / 1000, kv_gib, "-o", color=C_KV, ms=5, label="実測（ds4-bench の kvcache_bytes）")
    ax.plot(xx / 1000, yy, "--", color=C_MUTED, label=f"線形外挿（約 {a*1e6:.1f} GiB / 100万トークン）")
    ax.scatter([1000], [kv_1m], color=C_BAD, zorder=5)
    ax.annotate(f"100万トークン ≈ {kv_1m:.1f} GiB",
                xy=(1000, kv_1m), xytext=(560, kv_1m * 0.62),
                color=C_BAD, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_BAD))
    ax.set_xlabel("コンテキスト長（千トークン）")
    ax.set_ylabel("圧縮済み KV キャッシュ（GiB）")
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, max(kv_1m * 1.1, kv_gib.max() * 1.2))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    ax.set_title("DeepSeek V4 Flash の圧縮 KV キャッシュ — コンテキスト長との関係（DGX Spark, q2-imatrix）")
    fig.tight_layout()
    out = IMG / "ds4-kvcache-vs-context.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"(1M extrapolation = {kv_1m:.1f} GiB)")


# ---------------------------------------------------------------------------
def chart_prefill_decode_bar():
    # All q2, prompt of roughly 7k-12k tokens (a "real" prefill, not a tiny one).
    # DGX Spark: own measurement (65,536-token single prefill from the long-ctx sweep).
    # Mac: DwarfStar 4 README speed table (cited, not re-measured here).
    rows = [
        ("DGX Spark GB10\n128GB / 273 GB/s\n(q2, CUDA)", 247.1, 11.4, C_PREFILL),
        ("MacBook Pro M3 Max\n128GB / ~400 GB/s\n(q2, Metal)", 250.1, 21.5, C_PREFILL),
        ("Mac Studio M3 Ultra\n512GB / ~819 GB/s\n(q2, Metal)", 468.0, 27.4, C_PREFILL),
    ]
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))

    fig, (axp, axg) = plt.subplots(1, 2, figsize=(11, 4.6))
    bars1 = axp.bar(x, [r[1] for r in rows], 0.55, color=[C_PREFILL, "#60a5fa", "#1e3a8a"])
    for bi, r in zip(bars1, rows):
        axp.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 8, f"{r[1]:.0f}", ha="center", fontsize=10)
    axp.set_title("prefill スループット（トークン/秒）")
    axp.set_xticks(x); axp.set_xticklabels(labels, fontsize=8)
    axp.set_ylim(0, 540); axp.grid(axis="y", alpha=0.25)

    bars2 = axg.bar(x, [r[2] for r in rows], 0.55, color=[C_DECODE, "#f87171", "#7f1d1d"])
    for bi, r in zip(bars2, rows):
        axg.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 0.4, f"{r[2]:.1f}", ha="center", fontsize=10)
    axg.set_title("生成スループット（トークン/秒）")
    axg.set_xticks(x); axg.set_xticklabels(labels, fontsize=8)
    axg.set_ylim(0, 32); axg.grid(axis="y", alpha=0.25)

    fig.suptitle("DwarfStar 4 / DeepSeek V4 Flash q2 — prefill は計算律速、生成はメモリ帯域律速（Mac の数値は README ベンチ表からの引用）", fontsize=11)
    fig.tight_layout()
    out = IMG / "ds4-prefill-decode-bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def chart_diskkv_cold_vs_warm():
    d = json.loads((RESULTS / "disk-kv.json").read_text())
    cold, warm = d["cold_s"], d["warm_s"]
    ptok = d["cold_usage"]["prompt_tokens"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(["コールド\n(キャッシュなし)", "ウォーム\n(disk KV ヒット)"], [cold, warm], 0.5, color=[C_COLD, C_WARM])
    for b, v in zip(bars, [cold, warm]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + cold * 0.02, f"{v:.1f} 秒", ha="center", fontsize=12)
    ax.set_ylabel("応答までの所要時間（秒）")
    ax.set_ylim(0, cold * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate(f"{cold/warm:.0f}x 高速化",
                xy=(1, warm), xytext=(0.55, cold * 0.55),
                fontsize=13, color=C_WARM, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_WARM))
    ax.set_title(f"Disk KV キャッシュ — 約 {ptok:,} トークンの同一プロンプト再送（DGX Spark, q2-imatrix）", fontsize=11)
    fig.tight_layout()
    out = IMG / "ds4-diskkv-cold-vs-warm.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
def chart_feasibility_matrix():
    rows = [
        ("make 一発で CUDA ビルド", "OK", "aarch64 + CUDA 13.0、-arch=native で GB10 を自動認識、約16秒"),
        ("284B-A13B q2-imatrix が 128GB に載る", "OK", "重み約81GiB + コンテキスト・KV ≈ 動作中 112〜118GiB"),
        ("生成（decode）スループット", "WARN", "約 8〜15 トークン/秒（メモリ帯域 273 GB/s が上限）"),
        ("prefill スループット", "OK", "増分で約 80〜250 トークン/秒、GB10 の計算力が効く"),
        ("長文（〜26万トークン）", "OK", "圧縮 KV が小さく RAM 内で完結、prefill 時間は分単位"),
        ("Disk KV キャッシュで再 prefill 省略", "OK", "32k トークンの再送が 115秒 → 7.7秒（約15倍）"),
        ("Claude Code / エージェント用途", "WARN", "長文一括（レビュー・要約）は実用、対話チャットは遅い"),
    ]
    sc = {"OK": (C_OK, "○"), "WARN": (C_WARN, "△"), "BAD": (C_BAD, "×")}
    fig, ax = plt.subplots(figsize=(10.5, 0.8 + 0.6 * len(rows)))
    ax.axis("off")
    y = len(rows)
    ax.text(0.015, y + 0.35, "項目", fontsize=11, fontweight="bold")
    ax.text(0.42, y + 0.35, "判定", fontsize=11, fontweight="bold", ha="center")
    ax.text(0.50, y + 0.35, "メモ", fontsize=11, fontweight="bold")
    for i, (name, st, note) in enumerate(rows):
        yy = y - i - 0.5
        if i % 2 == 0:
            ax.axhspan(yy - 0.5, yy + 0.5, color="#f1f5f9", zorder=0)
        ax.text(0.015, yy, name, fontsize=10, va="center")
        col, mark = sc[st]
        ax.text(0.42, yy, mark, fontsize=15, va="center", ha="center", color=col, fontweight="bold")
        ax.text(0.50, yy, note, fontsize=9, va="center", color="#334155")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y + 0.8)
    ax.set_title("DGX Spark で DwarfStar 4 / DeepSeek V4 Flash を動かす — 可否マトリクス", fontsize=12, pad=10)
    fig.tight_layout()
    out = IMG / "ds4-feasibility-matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    chart_context_vs_throughput()
    chart_kvcache_vs_context()
    chart_prefill_decode_bar()
    chart_diskkv_cold_vs_warm()
    # chart_feasibility_matrix()  # 記事側で markdown 表に置き換えたため未使用
