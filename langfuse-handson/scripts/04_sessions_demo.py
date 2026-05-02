"""Phase 1: bind multi-turn chat conversations to Langfuse Sessions.

Phoenix's ``using_session`` is replaced here by passing
``metadata={"langfuse_session_id": ...}`` (and a per-user tag) to LangChain's
invoke config. The Langfuse CallbackHandler picks them up and writes them onto
the resulting trace, so the Sessions tab can roll up turns by session_id.

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/04_sessions_demo.py
"""

import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
langfuse = Langfuse()
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)


def run_session(session_id: str, user_id: str, turns: list[str]) -> None:
    history = [
        SystemMessage(
            content="You are a concise assistant. Keep every reply under 30 words."
        )
    ]
    for user_text in turns:
        history.append(HumanMessage(content=user_text))
        result = llm.invoke(
            history,
            config={
                "callbacks": [handler],
                "run_name": "chat-turn",
                "metadata": {
                    "langfuse_session_id": session_id,
                    "langfuse_user_id": user_id,
                },
            },
        )
        history.append(result)
        print(f"[{session_id[:14]}] {user_text}")
        print(f"   -> {result.content}\n")


run_session(
    session_id=f"handson-{uuid.uuid4()}",
    user_id="alice",
    turns=[
        "Tell me three famous sights in Kyoto.",
        "Which one is closest to Kyoto Station?",
        "How long does it take on foot?",
    ],
)

run_session(
    session_id=f"handson-{uuid.uuid4()}",
    user_id="bob",
    turns=[
        "Recommend two day trips from Tokyo.",
        "Which is better for first-time visitors?",
        "What is the round-trip cost roughly?",
    ],
)

langfuse.flush()
print("UI: http://localhost:3000/project/handson-project/sessions")
