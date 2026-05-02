"""Common vLLM (OpenAI-compatible) client + Langfuse instrumentation.

Used by bench_heron.py and bench_jmmmu.py. Each call to ``chat_with_image`` is
recorded as a Langfuse ``generation`` observation, so latency, token usage, and
model name surface automatically in Langfuse Traces and Dataset Runs.

Usage:
    from lib_vllm_client import make_vllm_client, chat_with_image
    client = make_vllm_client()
    resp, latency_s = chat_with_image(client, "nemotron-omni", img_b64, prompt)
"""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Tuple

from openai import OpenAI
from PIL import Image

# Langfuse @observe is the lowest-friction way to wrap a function call into a
# generation observation. Importing late so that scripts which only need image
# encoding (no API call) do not require LANGFUSE_* env vars.
from langfuse import get_client, observe


def encode_image_to_b64(
    img: Image.Image, fmt: str = "JPEG", max_pixels: int = 2_000_000
) -> str:
    """PIL Image -> base64 string. Resize down if larger than ``max_pixels``.

    Default 2M px matches the vLLM ``--media-io-kwargs max_pixels`` setting so
    the server side does not silently downscale and confuse latency comparisons.
    """
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)))
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_vllm_client(base_url: str = "http://localhost:8001/v1") -> OpenAI:
    """OpenAI-compatible client pointed at the local vLLM serve."""
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=180)


@observe(name="vllm_chat", as_type="generation")
def chat_with_image(
    client: OpenAI,
    model: str,
    image_b64: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> Tuple[object, float]:
    """Single image + text chat call. Returns ``(response, latency_seconds)``.

    Set ``enable_thinking=False`` for short Heron / JMMMU answers; reasoning
    traces blow the 8192 max_model_len budget on multi-image questions.
    """
    extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    latency = time.perf_counter() - start

    # Surface the input/output/usage in Langfuse so the generation observation
    # is browsable. ``input`` is the user prompt + a placeholder for the image
    # (we keep image bytes out of Langfuse to avoid blowing storage).
    lf = get_client()
    lf.update_current_generation(
        model=model,
        input=[{"prompt": prompt, "image_bytes": len(image_b64) * 3 // 4}],
        output=resp.choices[0].message.content,
        usage_details={
            "input": resp.usage.prompt_tokens if resp.usage else 0,
            "output": resp.usage.completion_tokens if resp.usage else 0,
            "total": resp.usage.total_tokens if resp.usage else 0,
        },
        metadata={"latency_seconds": latency},
    )
    return resp, latency
