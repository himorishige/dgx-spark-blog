"""JCQ (JCommonsenseQA v1.1) benchmark for Gemma 4 MTP on vLLM.

Compares baseline (target only) vs MTP (target + drafter) on the same 200-question
slice. Captures: accuracy, p50/p95 latency, tok/s, exact-match rate between modes.

Phase 1 (5/9-13). Reuses the JCQ split used in the G1 article (leemeng/jcommonsenseqa-v1.1).

Usage:
    python bench-jcq.py --model gemma4-e2b --label e2b-baseline
    python bench-jcq.py --model gemma4-e2b --label e2b-mtp2
    # Then re-run with the MTP server up and a different label.

Outputs:
    workspace/blog/scripts/data/g4-mtp/{label}.jsonl  -- one line per question
    Also prints a summary line.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "g4-mtp"
DEFAULT_DATASET = "leemeng/jcommonsenseqa-v1.1"
DEFAULT_SEED = 42
DEFAULT_NUM_QUESTIONS = 200
PROMPT_LETTERS = ["A", "B", "C", "D", "E"]


def load_jcq(num_questions: int, seed: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    ds = load_dataset(DEFAULT_DATASET, split="validation")
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    fewshot = rows[:3]
    test = rows[3 : 3 + num_questions]
    return fewshot, test


def render_q(row: dict) -> str:
    parts = [f"質問: {row['question']}"]
    for i, letter in enumerate(PROMPT_LETTERS):
        parts.append(f"{letter}. {row[f'choice{i}']}")
    return "\n".join(parts)


def build_prompt(fewshot: list[dict], q: dict) -> str:
    lines = [
        "次の質問について、選択肢 A〜E から最も適切な答えを 1 つだけ選び、"
        "その記号のみを最初の行に出力してください。説明は不要です。",
        "",
    ]
    for ex in fewshot:
        lines.append(render_q(ex))
        letter = PROMPT_LETTERS[ex["label"]]
        lines.append(f"答え: {letter}")
        lines.append("")
    lines.append(render_q(q))
    lines.append("答え:")
    return "\n".join(lines)


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    head = text.strip().splitlines()[0].strip()
    for ch in head:
        if ch in PROMPT_LETTERS:
            return ch
    return None


def call(model: str, prompt: str, port: int, max_tokens: int) -> tuple[float, int, str]:
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    completion = payload["choices"][0]["message"]["content"] or ""
    completion_tokens = payload["usage"]["completion_tokens"]
    return elapsed, completion_tokens, completion


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="vLLM served-model-name (e.g. gemma4-e2b)")
    p.add_argument("--label", required=True, help="output label, e.g. e2b-baseline / e2b-mtp2")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{args.label}.jsonl"

    print(f"=== JCQ bench: {args.model} → {out_path}")
    fewshot, test = load_jcq(args.num_questions, args.seed)
    print(f"loaded {len(fewshot)}-shot + {len(test)} test items")

    correct = 0
    parsed = 0
    elapsed_list: list[float] = []
    ctok_list: list[int] = []

    with out_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(test):
            prompt = build_prompt(fewshot, q)
            try:
                elapsed, ctok, completion = call(args.model, prompt, args.port, args.max_tokens)
            except urllib.error.HTTPError as exc:
                err = exc.read().decode("utf-8", errors="replace")[:300]
                print(f"  [{i + 1}] HTTPError {exc.code}: {err}")
                continue
            except Exception as exc:
                print(f"  [{i + 1}] {type(exc).__name__}: {exc}")
                continue

            pred = parse_answer(completion)
            gold = PROMPT_LETTERS[q["label"]]
            ok = pred == gold
            if pred is not None:
                parsed += 1
            if ok:
                correct += 1
            elapsed_list.append(elapsed)
            ctok_list.append(ctok)

            row = {
                "idx": i,
                "qid": q.get("q_id", q.get("id")),
                "gold": gold,
                "pred": pred,
                "completion": completion[:200],
                "elapsed_s": elapsed,
                "completion_tokens": ctok,
                "correct": ok,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 25 == 0:
                acc_so_far = correct / (i + 1)
                tps_so_far = sum(ctok_list) / sum(elapsed_list) if elapsed_list else 0
                print(
                    f"  {i + 1:3d}/{len(test)} | acc={acc_so_far:.3f} parsed={parsed}/{i + 1} "
                    f"| mean_tps={tps_so_far:.1f}",
                    flush=True,
                )

    n = len(elapsed_list)
    acc = correct / max(n, 1)
    parsed_rate = parsed / max(n, 1)
    p50 = statistics.median(elapsed_list) if elapsed_list else 0
    p95 = (
        statistics.quantiles(elapsed_list, n=20)[18] if len(elapsed_list) >= 20 else max(elapsed_list, default=0)
    )
    mean_tps = sum(ctok_list) / sum(elapsed_list) if elapsed_list else 0
    summary = {
        "label": args.label,
        "model": args.model,
        "n": n,
        "accuracy": acc,
        "parsed_rate": parsed_rate,
        "p50_s": p50,
        "p95_s": p95,
        "mean_tps": mean_tps,
        "total_completion_tokens": sum(ctok_list),
        "total_elapsed_s": sum(elapsed_list),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
