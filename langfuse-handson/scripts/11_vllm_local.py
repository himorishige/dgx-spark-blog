"""Phase 4: trace a DGX Spark local vLLM (Nemotron 3 Nano 30B-A3B NVFP4) call.

The custom model definition ``nemotron-3-nano-nvfp4-local`` was registered at
$0/$0 per token, so the resulting Generation observations show up in Langfuse
with token counts but $0 cost - which is the right answer for an on-prem LLM.

Usage (after start-vllm.sh has reached HTTP 200):
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/11_vllm_local.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
langfuse = Langfuse()

llm = ChatOpenAI(
    model="nemotron-3-nano-nvfp4-local",
    temperature=0,
    max_tokens=256,
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",  # vLLM doesn't validate the key
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant. Reply in 1-2 sentences."),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()

questions = [
    "What is the capital of Japan?",
    "Who invented the World Wide Web?",
    "In one sentence: why do neural networks need a non-linear activation?",
]

for q in questions:
    answer = chain.invoke(
        {"question": q},
        config={
            "callbacks": [handler],
            "run_name": "vllm-local-qa",
            "metadata": {"langfuse_user_id": "dgx-spark", "stack": "vllm-nvfp4"},
        },
    )
    print(f"Q: {q}\nA: {answer}\n")

langfuse.flush()
print("UI: http://localhost:3000/project/handson-project/traces")
