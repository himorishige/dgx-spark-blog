"""Long-form generation benchmark for Gemma 4 MTP on vLLM.

Phase 1 Day 2 (2026-05-08). Measures the regime where MTP is supposed to shine:
sustained decode with max_tokens=256.

Reports cold (round 1, kernel JIT included) and warm (rounds 2..N) statistics:
p50/p95 latency, mean tok/s, optionally vLLM acceptance rate from /metrics.

Usage:
    python bench-longform.py --model gemma4-e2b --label e2b-longform-baseline
    python bench-longform.py --model gemma4-e2b --label e2b-longform-mtp2 --grab-metrics

Outputs:
    workspace/blog/scripts/data/g4-mtp/{label}.long.jsonl  -- per-round records
    workspace/blog/scripts/data/g4-mtp/{label}.long.summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "g4-mtp"

PROMPTS_JA = [
    "DGX Spark という製品について、ユーザー視点で紹介する文章を 200 字程度で書いてください。",
    "Multi-Token Prediction (MTP) と従来の autoregressive デコードの違いを、初学者向けに 200 字程度で説明してください。",
    "vLLM の speculative decoding 機能について、なぜ高速化されるのかを 200 字程度で解説してください。",
    "Gemma 4 シリーズに含まれる E2B / E4B / 26B / 31B の違いを、200 字程度でまとめてください。",
    "Triton attention backend が選ばれる場面について、200 字程度で技術的に説明してください。",
    "NVFP4 量子化が weight-only 圧縮として動作する条件について、200 字程度で説明してください。",
    "MoE モデルの drafter が batch_size=1 で speedup を出しにくい理由を、200 字程度で説明してください。",
    "centroids masking という最適化手法について、想像を交えつつ 200 字程度で書いてください。",
    "DGX Spark の統合メモリ 128GB が大規模モデル運用にもたらす利点を 200 字程度でまとめてください。",
    "speculative decoding の採択率 (acceptance rate) という指標が何を意味するか、200 字程度で説明してください。",
]

# Mirror set in English. Same topics, similar length (~150 words for 256 tokens).
PROMPTS_EN = [
    "Write a short user-oriented introduction to the DGX Spark product in around 150 words.",
    "Explain the difference between Multi-Token Prediction (MTP) and traditional autoregressive decoding for beginners in around 150 words.",
    "Describe why vLLM's speculative decoding feature accelerates inference, in around 150 words.",
    "Summarize the differences among the Gemma 4 sizes E2B, E4B, 26B, and 31B in around 150 words.",
    "Explain the situations in which the Triton attention backend is chosen, in around 150 words technical detail.",
    "Describe the conditions under which NVFP4 quantization operates as weight-only compression, in around 150 words.",
    "Explain why a MoE-model drafter has difficulty producing speedup at batch_size=1, in around 150 words.",
    "Discuss the optimization technique called centroids masking with some speculation, in around 150 words.",
    "Summarize the benefits the DGX Spark 128GB unified memory brings to large-model operation, in around 150 words.",
    "Explain what the acceptance rate metric means in speculative decoding, in around 150 words.",
]

PROMPT_SETS = {"ja": PROMPTS_JA, "en": PROMPTS_EN}


def call(model: str, prompt: str, max_tokens: int, port: int) -> tuple[float, int, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    completion = payload["choices"][0]["message"]["content"] or ""
    completion_tokens = payload["usage"]["completion_tokens"]
    return elapsed, completion_tokens, completion


def grab_metrics(port: int) -> dict[str, float]:
    """Pull spec_decode acceptance + token counts from vLLM /metrics."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

    keys = [
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_emitted_tokens_total",
    ]
    out: dict[str, float] = {}
    for k in keys:
        m = re.search(rf"^{re.escape(k)}\b[^\n]*?\s+([0-9.eE+-]+)$", text, re.MULTILINE)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    if (
        "vllm:spec_decode_num_draft_tokens_total" in out
        and out["vllm:spec_decode_num_draft_tokens_total"] > 0
    ):
        out["acceptance_rate"] = (
            out.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
            / out["vllm:spec_decode_num_draft_tokens_total"]
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--grab-metrics", action="store_true")
    p.add_argument("--lang", choices=["ja", "en"], default="ja")
    args = p.parse_args()
    prompts = PROMPT_SETS[args.lang]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{args.label}.long.jsonl"

    print(f"=== long-form bench: {args.model} ({args.rounds} rounds, max_tokens={args.max_tokens})")

    metrics_before = grab_metrics(args.port) if args.grab_metrics else {}

    rounds: list[dict] = []
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.rounds):
            prompt = prompts[i % len(prompts)]
            try:
                elapsed, ctok, completion = call(args.model, prompt, args.max_tokens, args.port)
            except urllib.error.HTTPError as exc:
                err = exc.read().decode("utf-8", errors="replace")[:300]
                print(f"  round {i + 1}: HTTPError {exc.code}: {err}")
                continue
            except Exception as exc:
                print(f"  round {i + 1}: {type(exc).__name__}: {exc}")
                continue

            tps = ctok / elapsed if elapsed > 0 else 0
            rec = {
                "round": i + 1,
                "elapsed_s": elapsed,
                "completion_tokens": ctok,
                "tok_per_s": tps,
                "head": completion[:80].replace("\n", " "),
            }
            rounds.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  r{i + 1:02d}: {elapsed:6.2f}s / {ctok:4d} tok / {tps:6.2f} tok/s")

    metrics_after = grab_metrics(args.port) if args.grab_metrics else {}

    if not rounds:
        print("no successful rounds")
        return

    cold = rounds[0]
    warm = rounds[1:] if len(rounds) > 1 else rounds

    def stats(rs: list[dict]) -> dict:
        elapsed = [r["elapsed_s"] for r in rs]
        ctok = [r["completion_tokens"] for r in rs]
        tps = [r["tok_per_s"] for r in rs]
        return {
            "n": len(rs),
            "p50_s": statistics.median(elapsed) if elapsed else 0,
            "p95_s": (
                statistics.quantiles(elapsed, n=20)[18] if len(elapsed) >= 20 else max(elapsed, default=0)
            ),
            "mean_tps": (sum(ctok) / sum(elapsed)) if sum(elapsed) > 0 else 0,
            "median_tps": statistics.median(tps) if tps else 0,
            "stdev_tps": statistics.stdev(tps) if len(tps) > 1 else 0,
            "total_completion_tokens": sum(ctok),
            "total_elapsed_s": sum(elapsed),
        }

    summary = {
        "label": args.label,
        "model": args.model,
        "lang": args.lang,
        "rounds": len(rounds),
        "max_tokens": args.max_tokens,
        "cold": {
            "elapsed_s": cold["elapsed_s"],
            "completion_tokens": cold["completion_tokens"],
            "tok_per_s": cold["tok_per_s"],
        },
        "warm": stats(warm),
        "all": stats(rounds),
    }
    if args.grab_metrics:
        summary["metrics_before"] = metrics_before
        summary["metrics_after"] = metrics_after
        # Compute deltas for spec metrics during this run
        delta = {}
        for k, v in metrics_after.items():
            if isinstance(v, (int, float)) and isinstance(metrics_before.get(k), (int, float)):
                delta[k] = v - metrics_before[k]
        if delta.get("vllm:spec_decode_num_draft_tokens_total", 0) > 0:
            delta["acceptance_rate"] = (
                delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
                / delta["vllm:spec_decode_num_draft_tokens_total"]
            )
        summary["metrics_delta"] = delta

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
