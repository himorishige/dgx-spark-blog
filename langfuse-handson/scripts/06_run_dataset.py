"""Phase 2: run two prompt variants over the dataset and compare.

Same setup as Phoenix's Experiments chapter:
  prompt-A: ``"Answer in exactly one word."``
  prompt-B: ``"Provide a short factual answer in at most 3 words."``

The Langfuse v4 SDK exposes ``langfuse.run_experiment()`` for this. Each call
materializes a Dataset Run and surfaces it under the dataset's Experiments tab
for side-by-side comparison.

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/06_run_dataset.py
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse
from langfuse.experiment import Evaluation
from langfuse.langchain import CallbackHandler

DATASET_NAME = "handson-general-qa"
RUNS = [
    ("prompt-A-one-word-v4", "Answer in exactly one word."),
    ("prompt-B-short-phrase-v4", "Provide a short factual answer in at most 3 words."),
]

handler = CallbackHandler()
langfuse = Langfuse()
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0, max_tokens=128)


def build_chain(system_prompt: str):
    return (
        ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{question}")]
        )
        | llm
        | StrOutputParser()
    )


def make_task(chain):
    def task(*, item, **_):
        return chain.invoke(
            {"question": item.input["question"]},
            config={"callbacks": [handler]},
        ).strip()
    return task


def contains_match(*, input, output, expected_output, **_):
    expected = (expected_output or {}).get("answer", "")
    return Evaluation(
        name="contains_match",
        value=1.0 if expected.lower() in str(output).lower() else 0.0,
        comment=f"expected={expected!r}",
    )


dataset = langfuse.get_dataset(DATASET_NAME)
print(f"dataset items: {len(dataset.items)}")

for run_name, system_prompt in RUNS:
    chain = build_chain(system_prompt)
    result = langfuse.run_experiment(
        name=DATASET_NAME,
        run_name=run_name,
        description=f"system prompt: {system_prompt!r}",
        data=dataset.items,
        task=make_task(chain),
        evaluators=[contains_match],
        max_concurrency=4,
    )
    avg = (
        sum(
            s.value
            for r in result.item_results
            for s in r.evaluations
            if s.name == "contains_match"
        )
        / max(len(result.item_results), 1)
    )
    print(f"{run_name}: contains_match avg={avg:.2f}")

langfuse.flush()
print(
    f"UI: http://localhost:3000/project/handson-project/datasets/{dataset.id}"
)
