"""Minimal httpx client for the local vLLM /v1/chat/completions endpoint.

Targets a Japanese-capable LLM served by vLLM on DGX Spark. Default port is
8001 to avoid clashing with the SKAB chronos2 stack. The article positions
this LLM as the "Reasoner" layer that turns numeric anomaly scores into a
short Japanese maintenance comment.

Tested model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 served as
"nemotron-3-nano-nvfp4-local" by ~/works/langfuse-handson/scripts/start-vllm.sh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


SYSTEM_PROMPT_JA = (
    "あなたは産業設備の保全エンジニアです。"
    "時系列モデルによる異常スコアとセンサー値を見て、"
    "現場で参考になる短い保全コメント（1〜2文）を返してください。"
    "推測ではなく、観測された数値だけを根拠にしてください。"
    "実在しない過去事例や根拠のない確定的表現（必ず故障する等）は避けてください。"
)

PROMPT_HEADER_JA = "以下のデータに基づいて保全コメントを返してください。"


@dataclass
class VLLMClient:
    base_url: str = "http://127.0.0.1:8001"
    served_model_name: str = "nemotron-3-nano-nvfp4-local"
    timeout_s: float = 120.0

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 200,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.served_model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        payload.update(kwargs)
        with httpx.Client(timeout=self.timeout_s) as cli:
            resp = cli.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as cli:
                r = cli.get(f"{self.base_url}/v1/models")
            return r.status_code == 200
        except Exception:
            return False


def build_user_prompt(observation: dict[str, Any]) -> str:
    """Render the observation dict as a JSON block prefixed with a short header."""
    body = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"{PROMPT_HEADER_JA}\n\n```json\n{body}\n```"
