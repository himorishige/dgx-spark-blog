"""Phase 3: register two prompt versions and tag the latest as production.

Phoenix uses ``PromptVersion`` + ``client.prompts.tags.create``. Langfuse rolls
the same idea into a single ``create_prompt`` call where ``labels`` doubles as
the tag mechanism (a label is unique across versions, so re-pinning to a new
version automatically demotes the old one).

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/09_prompts_versioned.py
"""

from langfuse import Langfuse

PROMPT_NAME = "handson-concise-qa"

langfuse = Langfuse()

v1 = langfuse.create_prompt(
    name=PROMPT_NAME,
    prompt=[
        {
            "role": "system",
            "content": "You are a concise assistant. Reply in a single sentence.",
        },
        {"role": "user", "content": "{{question}}"},
    ],
    config={"model": "claude-haiku-4-5", "temperature": 0, "max_tokens": 128},
    labels=["baseline"],
    type="chat",
    commit_message="v1 baseline concise prompt",
)
print(f"v1 version={v1.version} labels={v1.labels}")

v2 = langfuse.create_prompt(
    name=PROMPT_NAME,
    prompt=[
        {
            "role": "system",
            "content": (
                "You are a precise factual assistant. Answer with exactly one "
                "short sentence, no preamble."
            ),
        },
        {"role": "user", "content": "{{question}}"},
    ],
    config={"model": "claude-haiku-4-5", "temperature": 0, "max_tokens": 128},
    labels=["production"],
    type="chat",
    commit_message="v2 stricter one-sentence prompt with no preamble",
)
print(f"v2 version={v2.version} labels={v2.labels}")

production = langfuse.get_prompt(PROMPT_NAME, label="production")
print(
    f"production resolves to version={production.version} "
    f"system_preview={production.prompt[0]['content'][:60]!r}..."
)

langfuse.flush()
print(f"UI: http://localhost:3000/project/handson-project/prompts/{PROMPT_NAME}")
