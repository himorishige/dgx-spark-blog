"""LLM-as-a-Judge for Heron-Bench using Claude Haiku 4.5 with prompt caching.

Heron-Bench officially scores responses 1–5 against a reference answer. We
adapt the same rubric for Japanese multimodal output and cache the long judge
instructions so the cost stays roughly $5 across 102 questions x 3 models.

Usage:
    from lib_judge import judge_heron
    score, reasoning, cached = judge_heron(question, reference, candidate, image_caption)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic

JUDGE_MODEL = "claude-haiku-4-5"

# Long, stable system prompt -> goes in a cache_control block. Short user
# message (the actual Q&A pair) stays uncached. Prompt caching needs the same
# *prefix* every call, so we include the rubric and an example here.
JUDGE_SYSTEM_PROMPT = """あなたは日本語マルチモーダル応答の採点担当です。Heron-Bench の評価基準に沿って、視覚言語モデルの応答を 1〜5 のスコアで採点してください。

## 採点基準

- **5 点**: 参照回答と意味的にほぼ完全に一致し、画像の内容を正確に捉えている
- **4 点**: 参照回答と概ね一致しているが、軽微な抜けや表現の違いがある
- **3 点**: 部分的に正しいが、重要な要素が欠落しているか誤りを含む
- **2 点**: 一部関連はあるが、画像内容や参照回答からの乖離が大きい
- **1 点**: 完全に誤っているか、画像と無関係な応答

## 出力形式

必ず以下の JSON を返してください。コードフェンスは付けないでください。

{"score": <1-5 の整数>, "reasoning": "<採点理由を 1〜2 文の日本語で>"}

## 採点例

質問: この画像に写っている人物は何をしていますか？
参照回答: 公園で犬と散歩している
モデル応答: 公園で犬を連れて歩いている女性
出力: {"score": 5, "reasoning": "参照回答の核心要素（公園、犬、散歩）を全て捉えており、自然な日本語で表現されている"}

質問: ボトルの色を教えてください
参照回答: 青色
モデル応答: 緑色のペットボトル
出力: {"score": 1, "reasoning": "参照回答は青色だがモデルは緑色と回答しており、視覚的に致命的な誤り"}
"""


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    cached_input_tokens: int
    input_tokens: int
    output_tokens: int
    raw_response: str


_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _parse_json(raw: str) -> tuple[int, str]:
    """Extract {score, reasoning} from raw text. Tolerates code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(cleaned)
        score = int(obj.get("score", 0))
        reasoning = str(obj.get("reasoning", ""))
        return max(1, min(5, score)), reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fall back: pull a digit 1-5 with regex so a malformed response still
        # produces a usable score.
        m = re.search(r'"score"\s*:\s*([1-5])', raw)
        score = int(m.group(1)) if m else 1
        return score, raw[:200]


def judge_heron(
    question: str,
    reference: str,
    candidate: str,
) -> JudgeResult:
    """Score a Heron-Bench candidate answer 1-5 with Claude Haiku 4.5."""
    user_message = (
        f"質問: {question}\n"
        f"参照回答: {reference}\n"
        f"モデル応答: {candidate}\n"
        "上記の応答を採点してください。"
    )

    msg = _get_client().messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": JUDGE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    raw = "".join(b.text for b in msg.content if b.type == "text")
    score, reasoning = _parse_json(raw)

    usage = msg.usage
    return JudgeResult(
        score=score,
        reasoning=reasoning,
        cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        raw_response=raw,
    )
