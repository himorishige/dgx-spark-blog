"""Phase 2: register 30 general-knowledge QAs as a Langfuse dataset.

Usage:
    cd ~/works/langfuse-handson
    uv run --env-file=.env scripts/05_upload_dataset.py
"""

from _general_qa import GENERAL_QA
from langfuse import Langfuse

DATASET_NAME = "handson-general-qa"

langfuse = Langfuse()

langfuse.create_dataset(
    name=DATASET_NAME,
    description="30 general-knowledge QAs for the langfuse handson article",
    metadata={"source": "blog-handson", "size": len(GENERAL_QA)},
)

for i, (question, answer) in enumerate(GENERAL_QA):
    langfuse.create_dataset_item(
        dataset_name=DATASET_NAME,
        input={"question": question},
        expected_output={"answer": answer},
        metadata={"qa_id": f"qa-{i:03d}"},
    )

langfuse.flush()
print(f"dataset={DATASET_NAME} items={len(GENERAL_QA)}")
print(
    f"UI: http://localhost:3000/project/handson-project/datasets/{DATASET_NAME}"
)
