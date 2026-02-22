#!/usr/bin/env python3
"""N3 Evaluation Script

Evaluate Nemotron 9B-v2-Japanese RAG performance before/after RAFT fine-tuning.

Metrics:
  - EM (Exact Match): strict string match on extracted answer
  - F1: token-level F1 between predicted and expected answer
  - Unanswerable Precision/Recall: detecting "cannot answer" cases
  - JCQ (JCommonsenseQA): general capability regression check

Usage:
  # Evaluate RAG on test set (Ollama endpoint)
  python n3-evaluate.py rag \
    --test-file ./data/n3/test.jsonl \
    --model nemotron-9b-jp \
    --output-file ./data/n3/results_baseline.json

  # Evaluate with RAFT fine-tuned model
  python n3-evaluate.py rag \
    --test-file ./data/n3/test.jsonl \
    --model nemotron-9b-jp-raft \
    --output-file ./data/n3/results_raft.json

  # Evaluate unanswerable detection (strips oracle from test samples)
  python n3-evaluate.py rag-unanswerable \
    --test-file ./data/n3/test.jsonl \
    --model nemotron-9b-jp-raft \
    --output-file ./data/n3/unanswerable_raft.json

  # Compare two result files
  python n3-evaluate.py compare \
    --baseline ./data/n3/results_baseline.json \
    --raft ./data/n3/results_raft.json

  # Run JCQ regression check
  python n3-evaluate.py jcq \
    --model nemotron-9b-jp \
    --output-file ./data/n3/jcq_baseline.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
UNANSWERABLE_KEYWORDS = [
    "回答できません",
    "情報が含まれていません",
    "情報は見つかりません",
    "該当する記述は見つかりません",
    "回答するための十分な情報",
]


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------


def ollama_chat(model: str, messages: list[dict], base_url: str = OLLAMA_BASE_URL) -> str:
    """Send chat request to Ollama and return the response."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation
    text = text.rstrip("。．.、，,")
    return text


def compute_em(pred: str, gold: str) -> float:
    """Exact Match score."""
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def tokenize_ja(text: str) -> list[str]:
    """Simple character-level tokenization for Japanese F1."""
    # Remove whitespace and punctuation for token comparison
    text = normalize_text(text)
    text = re.sub(r"[。．.、，,\s]", "", text)
    return list(text)


def compute_f1(pred: str, gold: str) -> float:
    """Token-level F1 score."""
    pred_tokens = set(tokenize_ja(pred))
    gold_tokens = set(tokenize_ja(gold))

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_unanswerable_response(text: str) -> bool:
    """Check if the response indicates "cannot answer"."""
    return any(kw in text for kw in UNANSWERABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# RAG Evaluation
# ---------------------------------------------------------------------------


def cmd_rag(args):
    """Evaluate RAG performance on test set."""
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"Error: {test_path} not found")
        sys.exit(1)

    # Load test data
    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"=== RAG Evaluation ===")
    print(f"  Model: {args.model}")
    print(f"  Test samples: {len(samples)}")
    print(f"  Ollama URL: {args.ollama_url}")

    results = []
    em_scores = []
    f1_scores = []
    unanswerable_tp = 0  # True positive: predicted & actual unanswerable
    unanswerable_fp = 0  # False positive: predicted unanswerable but answerable
    unanswerable_fn = 0  # False negative: predicted answerable but actually unanswerable
    unanswerable_tn = 0  # True negative: predicted & actual answerable

    for sample in tqdm(samples, desc="Evaluating"):
        messages = sample["messages"]
        expected = sample["metadata"].get("expected_answer", "")
        # Support both has_oracle (train) and oracle_position (test) fields
        if "has_oracle" in sample["metadata"]:
            is_unanswerable = not sample["metadata"]["has_oracle"]
        else:
            is_unanswerable = sample["metadata"].get("oracle_position") is None

        try:
            prediction = ollama_chat(args.model, messages, args.ollama_url)
        except Exception as e:
            print(f"\n  Error: {e}")
            prediction = f"[ERROR: {e}]"

        pred_unanswerable = is_unanswerable_response(prediction)

        # Metrics
        if is_unanswerable:
            if pred_unanswerable:
                unanswerable_tp += 1
            else:
                unanswerable_fn += 1
            em = 0.0
            f1 = 0.0
        else:
            if pred_unanswerable:
                unanswerable_fp += 1
                em = 0.0
                f1 = 0.0
            else:
                unanswerable_tn += 1
                em = compute_em(prediction, expected)
                f1 = compute_f1(prediction, expected)

        em_scores.append(em)
        f1_scores.append(f1)

        results.append({
            "question": sample["metadata"]["question"],
            "expected": expected[:200],
            "prediction": prediction[:200],
            "em": em,
            "f1": f1,
            "is_unanswerable": is_unanswerable,
            "pred_unanswerable": pred_unanswerable,
        })

        # Rate limiting
        time.sleep(0.05)

    # Compute aggregated metrics
    answerable_results = [r for r in results if not r["is_unanswerable"]]
    answerable_em = [r["em"] for r in answerable_results] if answerable_results else [0]
    answerable_f1 = [r["f1"] for r in answerable_results] if answerable_results else [0]

    ua_precision = (
        unanswerable_tp / (unanswerable_tp + unanswerable_fp)
        if (unanswerable_tp + unanswerable_fp) > 0
        else 0.0
    )
    ua_recall = (
        unanswerable_tp / (unanswerable_tp + unanswerable_fn)
        if (unanswerable_tp + unanswerable_fn) > 0
        else 0.0
    )

    summary = {
        "model": args.model,
        "total_samples": len(results),
        "answerable_samples": len(answerable_results),
        "unanswerable_samples": len(results) - len(answerable_results),
        "metrics": {
            "em_mean": sum(answerable_em) / len(answerable_em),
            "f1_mean": sum(answerable_f1) / len(answerable_f1),
            "unanswerable_precision": ua_precision,
            "unanswerable_recall": ua_recall,
            "unanswerable_tp": unanswerable_tp,
            "unanswerable_fp": unanswerable_fp,
            "unanswerable_fn": unanswerable_fn,
            "unanswerable_tn": unanswerable_tn,
        },
        "results": results,
    }

    # Print summary
    print(f"\n=== Results ===")
    print(f"  Answerable EM:  {summary['metrics']['em_mean']:.4f}")
    print(f"  Answerable F1:  {summary['metrics']['f1_mean']:.4f}")
    print(f"  Unanswerable P: {ua_precision:.4f}")
    print(f"  Unanswerable R: {ua_recall:.4f}")
    print(f"  (TP={unanswerable_tp} FP={unanswerable_fp} FN={unanswerable_fn} TN={unanswerable_tn})")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved: {output_path}")


# ---------------------------------------------------------------------------
# RAG Unanswerable Evaluation
# ---------------------------------------------------------------------------


def strip_oracle_from_sample(sample: dict) -> dict:
    """Remove oracle document from a test sample to create an unanswerable variant.

    Identifies the oracle by oracle_position in metadata, removes it,
    and renumbers the remaining documents.

    User message format:
        以下の参考文書を基に質問に答えてください。

        参考文書1:
        [content]

        参考文書2:
        [content]

        質問: [question]
    """
    import copy

    sample = copy.deepcopy(sample)
    meta = sample["metadata"]
    oracle_pos = meta.get("oracle_position")

    if oracle_pos is None:
        return sample  # Already unanswerable

    user_msg = sample["messages"][1]["content"]

    # Split into documents using regex on "参考文書N:" headers
    doc_pattern = re.compile(r"参考文書(\d+):\n")
    parts = doc_pattern.split(user_msg)
    # parts = [preamble, "1", doc1_content, "2", doc2_content, ..., question_part_in_last]

    # Extract preamble (before first document)
    preamble = parts[0]  # "以下の参考文書を基に質問に答えてください。\n\n"

    # Extract numbered documents
    documents = {}
    for i in range(1, len(parts) - 1, 2):
        doc_num = int(parts[i])
        doc_content = parts[i + 1]
        documents[doc_num] = doc_content

    # The question is embedded in the last document's content
    last_doc_num = max(documents.keys())
    last_content = documents[last_doc_num]
    question_marker = "\n質問: "
    if question_marker in last_content:
        q_idx = last_content.index(question_marker)
        documents[last_doc_num] = last_content[:q_idx]
        question_part = last_content[q_idx:]
    else:
        question_part = f"\n質問: {meta['question']}"

    # Remove the oracle document
    del documents[oracle_pos]

    # Rebuild with renumbered documents
    new_user_msg = preamble
    for new_num, (_, content) in enumerate(sorted(documents.items()), 1):
        new_user_msg += f"参考文書{new_num}:\n{content}"
    new_user_msg += question_part

    sample["messages"][1]["content"] = new_user_msg
    meta["oracle_position"] = None
    meta["has_oracle"] = False

    return sample


def cmd_rag_unanswerable(args):
    """Evaluate unanswerable detection by stripping oracle documents."""
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"Error: {test_path} not found")
        sys.exit(1)

    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    # Strip oracle from all samples
    unanswerable_samples = [strip_oracle_from_sample(s) for s in samples]

    print(f"=== Unanswerable Detection Evaluation ===")
    print(f"  Model: {args.model}")
    print(f"  Samples: {len(unanswerable_samples)} (oracle stripped)")
    print(f"  Ollama URL: {args.ollama_url}")

    correct = 0
    total = 0
    results = []

    for sample in tqdm(unanswerable_samples, desc="Evaluating unanswerable"):
        messages = sample["messages"]
        question = sample["metadata"]["question"]

        try:
            prediction = ollama_chat(args.model, messages, args.ollama_url)
        except Exception as e:
            print(f"\n  Error: {e}")
            prediction = f"[ERROR: {e}]"

        detected = is_unanswerable_response(prediction)
        if detected:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "prediction": prediction[:200],
            "detected_unanswerable": detected,
        })

        time.sleep(0.05)

    recall = correct / total if total > 0 else 0.0

    summary = {
        "model": args.model,
        "total_samples": total,
        "detected": correct,
        "unanswerable_recall": recall,
        "results": results,
    }

    print(f"\n=== Unanswerable Detection Results ===")
    print(f"  Recall: {recall:.4f} ({correct}/{total})")
    print(f"  (= proportion of oracle-stripped samples correctly refused)")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def cmd_compare(args):
    """Compare baseline and RAFT results."""
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.raft, encoding="utf-8") as f:
        raft = json.load(f)

    print(f"=== Comparison ===")
    print(f"  Baseline model: {baseline['model']}")
    print(f"  RAFT model:     {raft['model']}")

    bm = baseline["metrics"]
    rm = raft["metrics"]

    headers = ["Metric", "Baseline", "RAFT", "Delta"]
    rows = [
        ("EM", bm["em_mean"], rm["em_mean"]),
        ("F1", bm["f1_mean"], rm["f1_mean"]),
        ("Unanswerable P", bm["unanswerable_precision"], rm["unanswerable_precision"]),
        ("Unanswerable R", bm["unanswerable_recall"], rm["unanswerable_recall"]),
    ]

    print(f"\n{'Metric':<20} {'Baseline':>10} {'RAFT':>10} {'Delta':>10}")
    print("-" * 52)
    for name, bv, rv in rows:
        delta = rv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{name:<20} {bv:>10.4f} {rv:>10.4f} {sign}{delta:>9.4f}")

    # Per-sample comparison (show biggest improvements)
    print(f"\n=== Top 5 Improvements (by F1) ===")
    baseline_by_q = {r["question"]: r for r in baseline["results"]}
    improvements = []
    for r in raft["results"]:
        br = baseline_by_q.get(r["question"])
        if br and not r["is_unanswerable"]:
            delta_f1 = r["f1"] - br["f1"]
            improvements.append((delta_f1, r["question"], br["f1"], r["f1"]))

    improvements.sort(reverse=True)
    for delta, q, bf1, rf1 in improvements[:5]:
        print(f"  +{delta:.3f} | {q[:60]}... (F1: {bf1:.3f} -> {rf1:.3f})")

    print(f"\n=== Top 5 Regressions (by F1) ===")
    for delta, q, bf1, rf1 in improvements[-5:]:
        print(f"  {delta:+.3f} | {q[:60]}... (F1: {bf1:.3f} -> {rf1:.3f})")


# ---------------------------------------------------------------------------
# JCQ (JCommonsenseQA) Regression Check
# ---------------------------------------------------------------------------

JCQ_PROMPT_TEMPLATE = """質問: {question}

選択肢:
0. {choice0}
1. {choice1}
2. {choice2}
3. {choice3}
4. {choice4}

正しい選択肢の番号のみを回答してください。"""


def cmd_jcq(args):
    """Run JCommonsenseQA regression check."""
    from datasets import load_dataset as hf_load

    print(f"=== JCommonsenseQA Evaluation ===")
    print(f"  Model: {args.model}")

    # Load JCQ dataset
    print("Loading JCommonsenseQA...")
    ds = hf_load("sbintuitions/JCommonsenseQA", split="validation")
    total = len(ds)
    print(f"  Total questions: {total}")

    if args.limit:
        ds = ds.select(range(min(args.limit, total)))
        print(f"  Limited to: {len(ds)}")

    correct = 0
    total_eval = 0
    results = []

    for sample in tqdm(ds, desc="JCQ"):
        question = sample["question"]
        choices = [sample[f"choice{i}"] for i in range(5)]
        gold = sample["label"]

        prompt = JCQ_PROMPT_TEMPLATE.format(
            question=question,
            choice0=choices[0],
            choice1=choices[1],
            choice2=choices[2],
            choice3=choices[3],
            choice4=choices[4],
        )

        try:
            response = ollama_chat(
                args.model,
                [{"role": "user", "content": prompt}],
                args.ollama_url,
            )
        except Exception as e:
            print(f"\n  Error: {e}")
            response = "[ERROR]"

        # Extract answer number
        pred = -1
        match = re.search(r"[0-4]", response.strip())
        if match:
            pred = int(match.group())

        is_correct = pred == gold
        if is_correct:
            correct += 1
        total_eval += 1

        results.append({
            "question": question,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": response[:100],
        })

        time.sleep(0.05)

    accuracy = correct / total_eval if total_eval > 0 else 0.0

    summary = {
        "model": args.model,
        "total": total_eval,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    print(f"\n=== JCQ Results ===")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total_eval})")
    print(f"  Baseline reference: 91.2% (base model)")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="N3 Evaluation Script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # rag
    rag = subparsers.add_parser("rag", help="Evaluate RAG performance")
    rag.add_argument("--test-file", required=True, help="Test JSONL file")
    rag.add_argument("--model", required=True, help="Ollama model name")
    rag.add_argument("--output-file", help="Output JSON file")
    rag.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    rag.set_defaults(func=cmd_rag)

    # rag-unanswerable
    rua = subparsers.add_parser("rag-unanswerable", help="Evaluate unanswerable detection")
    rua.add_argument("--test-file", required=True, help="Test JSONL file (oracle will be stripped)")
    rua.add_argument("--model", required=True, help="Ollama model name")
    rua.add_argument("--output-file", help="Output JSON file")
    rua.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    rua.set_defaults(func=cmd_rag_unanswerable)

    # compare
    cmp = subparsers.add_parser("compare", help="Compare baseline and RAFT results")
    cmp.add_argument("--baseline", required=True, help="Baseline results JSON")
    cmp.add_argument("--raft", required=True, help="RAFT results JSON")
    cmp.set_defaults(func=cmd_compare)

    # jcq
    jcq = subparsers.add_parser("jcq", help="JCommonsenseQA regression check")
    jcq.add_argument("--model", required=True, help="Ollama model name")
    jcq.add_argument("--output-file", help="Output JSON file")
    jcq.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    jcq.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    jcq.set_defaults(func=cmd_jcq)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
