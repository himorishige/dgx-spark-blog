"""Phase 1: instrument a LangChain QA chain with Langfuse.

Uses the Langfuse-native ``CallbackHandler`` (sibling of OpenInference / OTel
options that are demonstrated in 03_otel_openinference.py).

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/02_instrument_langchain.py
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
langfuse = Langfuse()

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant. Reply in a single sentence."),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()

questions = [
    "What is the capital of Japan?",
    "Who wrote 'Pride and Prejudice'?",
    "Briefly: why is the sky blue?",
]

for q in questions:
    answer = chain.invoke(
        {"question": q},
        config={"callbacks": [handler], "run_name": "qa-handson"},
    )
    print(f"Q: {q}")
    print(f"A: {answer}\n")

langfuse.flush()
print("UI: http://localhost:3000/project/handson-project/traces")
