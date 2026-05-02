"""Generate matplotlib charts for the Omni2 article.

Reads jsonl checkpoints under ``results/{model}/{bench}/*.jsonl`` and writes
PNGs into ``workspace/blog/images/nemotron3-omni-omni2/``.

Outputs:
- heron-radar.png       3-model radar over Heron categories
- jmmmu-bars-top.png    Top 8 JMMMU subjects (best gap between models)
- jmmmu-bars-bottom.png Bottom 8 JMMMU subjects
- latency-comparison.png Mean per-question latency by model
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# Register Noto Sans CJK JP so Japanese labels render in PNGs.
for fpath in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    try:
        font_manager.fontManager.addfont(fpath)
    except (OSError, RuntimeError):
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = (
    ROOT.parent.parent / "images" / "nemotron3-omni-omni2"
)  # workspace/blog/images/...

MODELS = ["omni", "gemma4", "cr2"]
MODEL_LABEL = {"omni": "Nemotron 3 Nano Omni", "gemma4": "Gemma 4 31B", "cr2": "Cosmos-Reason2 8B"}
MODEL_COLOR = {"omni": "#76b900", "gemma4": "#1976d2", "cr2": "#7b1fa2"}


def _latest_jsonl(dir_: Path) -> Path | None:
    files = sorted(dir_.glob("*.jsonl"))
    return files[-1] if files else None


def _load_heron(model: str) -> list[dict]:
    p = _latest_jsonl(RESULTS_DIR / model / "heron")
    if not p:
        return []
    with p.open() as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _load_jmmmu(model: str) -> list[dict]:
    p = _latest_jsonl(RESULTS_DIR / model / "jmmmu")
    if not p:
        return []
    with p.open() as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def heron_radar() -> Path:
    """Heron-Bench 4-category radar across 3 models."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    cat_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for r in _load_heron(model):
            cat = r.get("category", "unknown")
            if r.get("score") is not None:
                cat_scores[cat][model].append(r["score"])

    categories = sorted(cat_scores.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for model in MODELS:
        means = [
            float(np.mean(cat_scores[c][model])) if cat_scores[c][model] else 0.0
            for c in categories
        ]
        means += means[:1]
        ax.plot(angles, means, label=MODEL_LABEL[model], color=MODEL_COLOR[model], linewidth=2)
        ax.fill(angles, means, color=MODEL_COLOR[model], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Heron-Bench スコア（カテゴリ別、1-5 点）")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    out = OUT_DIR / "heron-radar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def jmmmu_subject_bars() -> tuple[Path, Path]:
    """Top 8 / bottom 8 JMMMU subjects by accuracy spread."""
    subj_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for r in _load_jmmmu(model):
            subj = r.get("subject", "unknown")
            if r.get("correct") is not None:
                subj_scores[subj][model].append(r["correct"])

    means = {
        s: {m: float(np.mean(subj_scores[s][m])) if subj_scores[s][m] else 0.0 for m in MODELS}
        for s in subj_scores
    }
    if not means:
        return OUT_DIR / "jmmmu-bars-top.png", OUT_DIR / "jmmmu-bars-bottom.png"

    sort_key = sorted(means.items(), key=lambda kv: max(kv[1].values()), reverse=True)
    top = sort_key[:8]
    bottom = sort_key[-8:]

    out_paths = []
    for label, slice_ in (("top", top), ("bottom", bottom)):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(slice_))
        width = 0.27
        for i, model in enumerate(MODELS):
            vals = [s[1][model] for s in slice_]
            ax.bar(x + (i - 1) * width, vals, width, label=MODEL_LABEL[model], color=MODEL_COLOR[model])
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in slice_], rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Exact Match Accuracy")
        ax.set_title(
            f"JMMMU {'上位' if label == 'top' else '下位'} 8 分野（3 モデル比較）"
        )
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        out = OUT_DIR / f"jmmmu-bars-{label}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths[0], out_paths[1]


def latency_comparison() -> Path:
    """Per-question latency derived from bench wall-clock time.

    Wall-clock numbers come from the bench log timestamps:
        Omni:    Heron  7 min / 103 = 4.08 s/q   (max_concurrency=2 → ~8 s standalone)
        Omni:    JMMMU  7 min / 1320 = 0.32 s/q  (~0.6 s standalone)
        Gemma4:  Heron 34 min / 103 = 19.8 s/q   (~40 s standalone)
        Gemma4:  JMMMU 31 min / 1320 = 1.41 s/q  (~2.8 s standalone)
        CR2:     Heron 15 min / 103 = 8.74 s/q   (~17 s standalone)
        CR2:     JMMMU  8 min / 1320 = 0.36 s/q  (~0.7 s standalone)
    Values shown are wall-clock (max_concurrency=2 effective) per question.
    """
    heron_per_q = {"omni": 4.08, "gemma4": 19.80, "cr2": 8.74}
    jmmmu_per_q = {"omni": 0.32, "gemma4": 1.41, "cr2": 0.36}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, title in (
        (axes[0], heron_per_q, "Heron-Bench"),
        (axes[1], jmmmu_per_q, "JMMMU"),
    ):
        ax.bar(
            [MODEL_LABEL[m] for m in MODELS],
            [data[m] for m in MODELS],
            color=[MODEL_COLOR[m] for m in MODELS],
        )
        ax.set_ylabel("1 問あたり実時間 (秒)")
        ax.set_title(f"{title} 推論レイテンシ")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        for i, m in enumerate(MODELS):
            ax.text(i, data[m], f"{data[m]:.1f}s", ha="center", va="bottom", fontsize=10)
    fig.suptitle("3 モデル レイテンシ比較 (max_concurrency=2)", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "latency-comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [heron_radar(), *jmmmu_subject_bars(), latency_comparison()]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
