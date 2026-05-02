"""JMMMU benchmark runner with Langfuse Dataset Run + exact-match scoring.

JMMMU is multi-choice (A/B/C/D), so we parse the model's answer and score with
exact match. No LLM judge -> ~zero API cost beyond the vLLM call itself.

Workflow:
    uv run --env-file=/home/morishige/works/langfuse-handson/.env \\
        python bench_jmmmu.py --upload
    uv run --env-file=/home/morishige/works/langfuse-handson/.env \\
        python bench_jmmmu.py --model omni
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from langfuse import Langfuse
from langfuse.experiment import Evaluation

from lib_vllm_client import chat_with_image, encode_image_to_b64, make_vllm_client

DATASET_NAME = "omni2-jmmmu"
JMMMU_HF_ID = "JMMMU/JMMMU"
JMMMU_PROMPT_TEMPLATE = (
    "{question}\n\n選択肢:\n{options_block}\n\n"
    "正解の選択肢の記号 (A, B, C, D など) のみを答えてください。"
)

SERVED_MODEL_MAP = {
    "omni": "nemotron-omni",
    "gemma4": "gemma4-a4b",
    "cr2": "cosmos-reason2",
}

RESULTS_DIR = Path(__file__).parent / "results"

ANSWER_RE = re.compile(r"\b([A-Z])\b")


def _format_options(options) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if isinstance(options, list):
        return "\n".join(f"{chr(ord('A') + i)}. {v}" for i, v in enumerate(options))
    return str(options)


def _extract_answer(text: str) -> str | None:
    """Pull the first standalone uppercase letter A-Z from the response."""
    m = ANSWER_RE.search(text or "")
    return m.group(1) if m else None


def _images_from_row(row: dict):
    """Collect image_1..image_4 in order, skipping None."""
    imgs = []
    for k in ("image_1", "image_2", "image_3", "image_4", "image"):
        v = row.get(k)
        if v is not None:
            imgs.append(v)
    return imgs


def upload_dataset(langfuse: Langfuse) -> None:
    """Load JMMMU validation split and register as a Langfuse dataset.

    JMMMU has multiple subject splits; we iterate every config (28 subjects)
    and concatenate. ``--limit`` on the runner side keeps smoke tests bounded.
    """
    from datasets import get_dataset_config_names

    subjects = get_dataset_config_names(JMMMU_HF_ID)
    print(f"Found {len(subjects)} JMMMU subject configs")

    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description=f"JMMMU test split across {len(subjects)} subjects",
            metadata={"source": JMMMU_HF_ID, "subjects": len(subjects)},
        )
    except Exception as e:  # noqa: BLE001
        print(f"create_dataset: {e} (likely already exists, continuing)")

    total = 0
    for subject in subjects:
        ds = load_dataset(JMMMU_HF_ID, subject, split="test")
        for i, row in enumerate(ds):
            uid = row.get("id") or f"{subject}-{i:03d}"
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input={
                    "question": row["question"],
                    "options_block": _format_options(row.get("options")),
                    "subject": subject,
                    "row_uid": uid,
                },
                expected_output={"answer": str(row.get("answer", "")).strip().upper()},
                metadata={"qa_id": uid, "subject": subject},
            )
            total += 1
    langfuse.flush()
    print(f"Uploaded {total} items across {len(subjects)} subjects")


def make_task(client, served_model_name: str, row_lookup: dict):
    def task(*, item, **_):
        row = row_lookup[item.input["row_uid"]]
        imgs = _images_from_row(row)
        # Use the first image only for now to fit max_model_len 8192; multi-
        # image questions get a lower expected score (mention in article).
        if not imgs:
            return ""
        img_b64 = encode_image_to_b64(imgs[0])
        prompt = JMMMU_PROMPT_TEMPLATE.format(
            question=item.input["question"],
            options_block=item.input["options_block"],
        )
        resp, _ = chat_with_image(
            client, served_model_name, img_b64, prompt, max_tokens=64
        )
        return resp.choices[0].message.content

    return task


def jmmmu_exact_match(*, input, output, expected_output, **_):  # noqa: A002
    pred = _extract_answer(str(output))
    gold = (expected_output.get("answer") or "").strip().upper()
    correct = 1.0 if pred and pred == gold else 0.0
    return Evaluation(
        name="jmmmu_exact_match",
        value=correct,
        comment=f"pred={pred!r} gold={gold!r}",
        metadata={"subject": input.get("subject", "unknown")},
    )


def _build_row_lookup() -> dict:
    """uid -> raw HF row, so the task can fetch images by UID."""
    from datasets import get_dataset_config_names

    subjects = get_dataset_config_names(JMMMU_HF_ID)
    out = {}
    for subject in subjects:
        ds = load_dataset(JMMMU_HF_ID, subject, split="test")
        for i, row in enumerate(ds):
            uid = row.get("id") or f"{subject}-{i:03d}"
            out[uid] = row
    return out


def save_local_checkpoint(model_key: str, run_name: str, scores: list[dict]) -> Path:
    out_dir = RESULTS_DIR / model_key / "jmmmu"
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
    row_lookup = _build_row_lookup()

    dataset = langfuse.get_dataset(DATASET_NAME)
    items = dataset.items[: args.limit] if args.limit else dataset.items

    run_name = f"{args.model}-jmmmu-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"Run: {run_name} ({len(items)} items, served={served_model_name})")

    result = langfuse.run_experiment(
        name=DATASET_NAME,
        run_name=run_name,
        description=f"{args.model} on JMMMU {len(items)} items",
        data=items,
        task=make_task(client, served_model_name, row_lookup),
        evaluators=[jmmmu_exact_match],
        max_concurrency=args.max_concurrency,
    )

    scored = [
        {
            "qa_id": ir.item.metadata.get("qa_id"),
            "subject": ir.item.input.get("subject"),
            "correct": next(
                (e.value for e in ir.evaluations if e.name == "jmmmu_exact_match"),
                None,
            ),
            "candidate": ir.output,
            "expected": ir.item.expected_output.get("answer"),
        }
        for ir in result.item_results
    ]
    out = save_local_checkpoint(args.model, run_name, scored)
    acc = sum(1 for s in scored if s["correct"] == 1.0) / max(1, len(scored))
    print(f"\n[done] jmmmu_exact_match={acc:.3f} | local backup={out}")
    langfuse.flush()


if __name__ == "__main__":
    main()
