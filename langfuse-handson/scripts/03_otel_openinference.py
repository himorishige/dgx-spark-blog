"""Phase 1 (alt route): send the Phoenix-style OpenInference traces to Langfuse.

This script is the "endpoint swap" demo from chapter 4. The instrumentation
itself is identical to the Phoenix article (``phoenix.otel.register`` + the
OpenInference LangChain auto-instrumentor); only the OTLP endpoint and the
auth header change.

Langfuse exposes an OTel HTTP receiver at ``/api/public/otel/v1/traces`` with
HTTP Basic auth ``Authorization: Basic base64(public_key:secret_key)``.

Usage:
    cd ~/works/langfuse-handson
    uv pip install arize-phoenix-otel openinference-instrumentation-langchain
    uv run --env-file=.env scripts/03_otel_openinference.py
"""

import base64
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from phoenix.otel import register

public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
secret_key = os.environ["LANGFUSE_SECRET_KEY"]
basic = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

# Same `register()` call as the Phoenix article; only `endpoint` and `headers`
# change. `auto_instrument=True` finds the OpenInference LangChain hook by
# package discovery, so no other code needs to know about the swap.
register(
    project_name="otel-openinference-route",
    endpoint=f"{os.environ['LANGFUSE_HOST']}/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {basic}"},
    auto_instrument=True,
)

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Reply in a single sentence."),
            ("human", "{question}"),
        ]
    )
    | llm
    | StrOutputParser()
)

for q in [
    "What is the capital of Japan?",
    "Who wrote 'Pride and Prejudice'?",
]:
    print(f"Q: {q}\nA: {chain.invoke({'question': q})}\n")

print("UI: http://localhost:3000/project/handson-project/traces")
