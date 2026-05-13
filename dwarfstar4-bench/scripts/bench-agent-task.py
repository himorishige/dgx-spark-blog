#!/usr/bin/env python3
"""Phase 3: measure DwarfStar 4 (ds4-server) on long-context "agent style" tasks.

Sends a couple of large prompts to a running ds4-server and records, per task:
the input/output token counts (from the `usage` field), wall time, and the
derived prefill vs decode split using a tiny warm-up probe to estimate decode
t/s. Writes results/agent-tasks.json.

Start the server first, e.g.:
  ./ds4-server --ctx 200000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 16384

Run:  uv run --with httpx python bench-agent-task.py [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def chat(base: str, messages: list[dict], max_tokens: int, think: bool = False) -> dict:
    body = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not think:
        body["thinking"] = {"type": "disabled"}
    t0 = time.time()
    r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=3600)
    dt = time.time() - t0
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    text = j["choices"][0]["message"].get("content") or ""
    return {
        "wall_s": round(dt, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "out_chars": len(text),
        "out_preview": text[:400],
    }


def derived_split(res: dict, decode_tps: float) -> dict:
    """Estimate prefill vs decode seconds given a decode t/s reference."""
    if not res.get("completion_tokens") or not decode_tps:
        return res
    decode_s = res["completion_tokens"] / decode_tps
    prefill_s = max(res["wall_s"] - decode_s, 0.0)
    pt = res.get("prompt_tokens") or 0
    res["est_decode_s"] = round(decode_s, 1)
    res["est_prefill_s"] = round(prefill_s, 1)
    res["est_prefill_tps"] = round(pt / prefill_s, 1) if prefill_s > 0 else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--code-dir", default=str(Path.home() / "works/dwarfstar4/ds4"),
                    help="directory whose source files are concatenated as a 'review this' prompt")
    ap.add_argument("--longdoc", default=str(HERE.parent.parent / "published"),
                    help="directory of .md files concatenated as a 'summarize this' prompt")
    args = ap.parse_args()
    base = args.base
    out: dict = {"base": base, "tasks": {}}

    # 0) warm-up probe: short prompt, measure decode t/s
    probe = chat(base, [{"role": "user", "content": "Say 'ok' and nothing else."}], max_tokens=32)
    # do a slightly longer generation to estimate decode tps
    gen_probe = chat(base, [{"role": "user", "content": "Count from 1 to 80, space separated, nothing else."}], max_tokens=200)
    decode_tps = None
    if gen_probe.get("completion_tokens") and gen_probe["wall_s"] > 0:
        # crude: assume prefill of a ~10-token prompt is negligible
        decode_tps = round(gen_probe["completion_tokens"] / gen_probe["wall_s"], 2)
    out["probe"] = {"short": probe, "gen": gen_probe, "decode_tps_est": decode_tps}
    print("decode_tps_est:", decode_tps)

    # 1) long-context code review
    src_files = sorted(Path(args.code_dir).glob("*.c")) + sorted(Path(args.code_dir).glob("*.h"))
    blob = []
    for f in src_files:
        try:
            blob.append(f"==== FILE: {f.name} ====\n" + f.read_text(errors="replace"))
        except Exception:
            pass
    code_text = "\n\n".join(blob)
    # cap to keep within ctx; ds4-server --ctx must be large enough
    code_text = code_text[:600_000]
    review_prompt = (
        "You are reviewing a C codebase. Read the files below and list, in 5 bullet "
        "points, the most important potential bugs, risky assumptions, or maintenance "
        "concerns you can identify. Be concrete and reference file names.\n\n" + code_text
    )
    print(f"code review prompt chars: {len(review_prompt)}")
    r = chat(base, [{"role": "user", "content": review_prompt}], max_tokens=600)
    out["tasks"]["code_review"] = derived_split(r, decode_tps)
    print("code_review:", json.dumps({k: v for k, v in r.items() if k != "out_preview"}, ensure_ascii=False))

    # 2) long-document summarization
    md_files = sorted(Path(args.longdoc).glob("*.md"))[:12]
    doc_blob = []
    for f in md_files:
        try:
            doc_blob.append(f"==== {f.name} ====\n" + f.read_text(errors="replace"))
        except Exception:
            pass
    doc_text = "\n\n".join(doc_blob)[:600_000]
    sum_prompt = (
        "Below are several technical blog posts. Summarize the whole collection in "
        "one short paragraph, then give a 5-item bullet list of the recurring themes.\n\n"
        + doc_text
    )
    print(f"summarization prompt chars: {len(sum_prompt)}")
    r = chat(base, [{"role": "user", "content": sum_prompt}], max_tokens=600)
    out["tasks"]["long_doc_summary"] = derived_split(r, decode_tps)
    print("long_doc_summary:", json.dumps({k: v for k, v in r.items() if k != "out_preview"}, ensure_ascii=False))

    (RESULTS / "agent-tasks.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", RESULTS / "agent-tasks.json")


if __name__ == "__main__":
    main()
