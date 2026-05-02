"""Heron-Bench benchmark runner with Langfuse Dataset Run.

Heron-Bench distributes raw JSONL files (questions_ja.jsonl + answers_gpt4.jsonl
+ context_ja.jsonl + images/*.jpg) rather than a HF Datasets table, so we read
the snapshot directory directly. 102 questions across 21 images and 3 question
categories (conv / detail / complex).

Workflow:
    uv run --env-file=/home/morishige/works/langfuse-handson/.env \\
        --python /home/morishige/works/langfuse-handson/.venv/bin/python \\
        python bench_heron.py --upload
    uv run --env-file=/home/morishige/works/langfuse-handson/.env \\
        --python /home/morishige/works/langfuse-handson/.venv/bin/python \\
        python bench_heron.py --model omni
    uv run --env-file=/home/morishige/works/langfuse-handson/.env \\
        --python /home/morishige/works/langfuse-handson/.venv/bin/python \\
        python bench_heron.py --model omni --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from langfuse import Langfuse
from langfuse.experiment import Evaluation
from PIL import Image

from lib_judge import judge_heron
from lib_vllm_client import chat_with_image, encode_image_to_b64, make_vllm_client

DATASET_NAME = "omni2-japanese-heron-bench"
HERON_HF_ID = "turing-motors/Japanese-Heron-Bench"
HERON_PROMPT_TEMPLATE = "{question}\n\n上記の質問に日本語で答えてください。"

SERVED_MODEL_MAP = {
    "omni": "nemotron-omni",
    "gemma4": "gemma4-a4b",
    "cr2": "cosmos-reason2",
}

RESULTS_DIR = Path(__file__).parent / "results"


def _heron_root() -> Path:
    """Return the local snapshot dir for the Heron dataset (downloads if needed)."""
    return Path(
        snapshot_download(
            repo_id=HERON_HF_ID, repo_type="dataset", local_files_only=False
        )
    )


def _read_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_heron_questions() -> list[dict]:
    """Merge questions_ja.jsonl + answers_gpt4.jsonl + context_ja.jsonl on question_id / image."""
    root = _heron_root()
    questions = _read_jsonl(root / "questions_ja.jsonl")
    answers = {a["question_id"]: a for a in _read_jsonl(root / "answers_gpt4.jsonl")}
    contexts = {c["image"]: c for c in _read_jsonl(root / "context_ja.jsonl")}
    rows: list[dict] = []
    for q in questions:
        qid = q["question_id"]
        ans = answers.get(qid, {})
        ctx = contexts.get(q["image"], {})
        rows.append(
            {
                "question_id": qid,
                "image": q["image"],
                "image_path": str(root / "images" / q["image"]),
                "category": q.get("category", "unknown"),
                "image_category": q.get("image_category", "unknown"),
                "context": ctx.get("caption", q.get("context", "")),
                "question": q["text"],
                "reference": ans.get("text", ""),
            }
        )
    return rows


def upload_dataset(langfuse: Langfuse) -> None:
    rows = _load_heron_questions()
    print(f"Loaded {len(rows)} Heron-Bench questions from local snapshot")
    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Heron-Bench 102 questions (Omni2 article)",
            metadata={"source": HERON_HF_ID, "size": len(rows)},
        )
    except Exception as e:  # noqa: BLE001
        print(f"create_dataset: {e} (likely already exists, continuing)")

    for r in rows:
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={
                "question_id": r["question_id"],
                "question": r["question"],
                "category": r["category"],
                "image_category": r["image_category"],
                "context": r["context"],
                "image_path": r["image_path"],
            },
            expected_output={"answer": r["reference"]},
            metadata={
                "qa_id": f"heron-{r['question_id']:03d}",
                "image": r["image"],
                "category": r["category"],
                "image_category": r["image_category"],
            },
        )
    langfuse.flush()
    print(f"Uploaded {len(rows)} items to dataset {DATASET_NAME!r}")


def make_task(client, served_model_name: str):
    def task(*, item, **_):
        img = Image.open(item.input["image_path"])
        img_b64 = encode_image_to_b64(img)
        prompt = HERON_PROMPT_TEMPLATE.format(question=item.input["question"])
        resp, _latency = chat_with_image(client, served_model_name, img_b64, prompt)
        return resp.choices[0].message.content

    return task


def heron_judge_evaluation(*, input, output, expected_output, **_):  # noqa: A002
    result = judge_heron(
        question=input["question"],
        reference=expected_output["answer"],
        candidate=str(output),
    )
    return Evaluation(
        name="heron_score",
        value=float(result.score),
        comment=result.reasoning,
        metadata={
            "category": input.get("category", "unknown"),
            "image_category": input.get("image_category", "unknown"),
            "judge_input_tokens": result.input_tokens,
            "judge_cached_tokens": result.cached_input_tokens,
            "judge_output_tokens": result.output_tokens,
        },
    )


def save_local_checkpoint(model_key: str, run_name: str, scores: list[dict]) -> Path:
    out_dir = RESULTS_DIR / model_key / "heron"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(SERVED_MODEL_MAP.keys()))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=2)
    args = parser.parse_args()

    langfuse = Langfuse()
    if args.upload:
        upload_dataset(langfuse)
        return
    if args.model is None:
        parser.error("--model required when not using --upload")

    served_model_name = SERVED_MODEL_MAP[args.model]
    client = make_vllm_client()
    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items[: args.limit] if args.limit else dataset.items

    run_name = f"{args.model}-heron-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Run: {run_name} ({len(items)} items, served={served_model_name})")

    result = langfuse.run_experiment(
        name=DATASET_NAME,
        run_name=run_name,
        description=f"{args.model} on Heron-Bench {len(items)} items",
        data=items,
        task=make_task(client, served_model_name),
        evaluators=[heron_judge_evaluation],
        max_concurrency=args.max_concurrency,
    )

    scored = [
        {
            "qa_id": ir.item.metadata.get("qa_id"),
            "category": ir.item.input.get("category"),
            "image_category": ir.item.input.get("image_category"),
            "score": next(
                (e.value for e in ir.evaluations if e.name == "heron_score"), None
            ),
            "reasoning": next(
                (e.comment for e in ir.evaluations if e.name == "heron_score"), ""
            ),
            "candidate": ir.output,
            "expected": ir.item.expected_output.get("answer"),
        }
        for ir in result.item_results
    ]
    out = save_local_checkpoint(args.model, run_name, scored)
    valid = [s["score"] for s in scored if s["score"] is not None]
    avg = sum(valid) / max(1, len(valid))
    print(f"\n[done] avg heron_score={avg:.3f} | local backup={out}")
    langfuse.flush()


if __name__ == "__main__":
    main()
