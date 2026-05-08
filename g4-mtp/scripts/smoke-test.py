"""Smoke test for vLLM Gemma 4 MTP server.

Phase 0 Go/No-Go gate (2026-05-07).

Sends a small chat completion to http://localhost:8001/v1/chat/completions
and reports tok/s. Optionally runs N iterations and prints the mean tok/s.

Usage:
    python smoke-test.py [--rounds N] [--max-tokens 256] [--model gemma4-e2b]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

PROMPTS = [
    "DGX Spark を 1 文で説明してください。",
    "Gemma 4 のサイズバリエーションを列挙してください。",
    "Multi-Token Prediction の利点を 2 つ挙げてください。",
    "vLLM の speculative decoding における acceptance rate とは何か。",
    "Triton attention backend が必要な Gemma 4 の固有事情を述べてください。",
]


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
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    completion = payload["choices"][0]["message"]["content"] or ""
    completion_tokens = payload["usage"]["completion_tokens"]
    return elapsed, completion_tokens, completion


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--model", default="gemma4-e2b")
    p.add_argument("--port", type=int, default=8001)
    args = p.parse_args()

    print(f"=== smoke test: {args.model} on :{args.port} ===")
    samples: list[tuple[float, int]] = []
    for i in range(args.rounds):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            elapsed, ctok, completion = call(args.model, prompt, args.max_tokens, args.port)
        except urllib.error.HTTPError as exc:
            print(f"  round {i + 1}: HTTPError {exc.code} {exc.reason}")
            print(exc.read().decode("utf-8", errors="replace")[:500])
            return
        except Exception as exc:
            print(f"  round {i + 1}: {type(exc).__name__}: {exc}")
            return

        tps = ctok / elapsed if elapsed > 0 else 0.0
        samples.append((elapsed, ctok))
        head = completion.replace("\n", " ")[:80]
        print(f"  round {i + 1}: {elapsed:6.2f}s / {ctok:4d} tok / {tps:6.2f} tok/s | {head}…")

    if not samples:
        return
    mean_elapsed = sum(s[0] for s in samples) / len(samples)
    mean_ctok = sum(s[1] for s in samples) / len(samples)
    mean_tps = mean_ctok / mean_elapsed if mean_elapsed > 0 else 0.0
    print(f"=== mean: {mean_elapsed:.2f}s / {mean_ctok:.1f} tok / {mean_tps:.2f} tok/s ===")


if __name__ == "__main__":
    main()
