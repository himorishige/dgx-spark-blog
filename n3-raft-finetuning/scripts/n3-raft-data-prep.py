#!/usr/bin/env python3
"""N3 RAFT Data Preparation Script

Convert JaGovFaqs-22k to RAFT format for SFT training on
Nemotron 9B-v2-Japanese.

RAFT (Retrieval Augmented Fine Tuning):
  - Oracle document (answer) + 4 distractors per sample
  - P=80%: oracle included in context
  - P=20%: oracle excluded (model must say "cannot answer")
  - CoT answer: citation -> reasoning -> concise answer

Usage:
  # Explore dataset statistics
  python n3-raft-data-prep.py explore

  # Sample train/test splits
  python n3-raft-data-prep.py sample --output-dir ./data/n3

  # Generate RAFT training data (requires ANTHROPIC_API_KEY)
  python n3-raft-data-prep.py generate --output-dir ./data/n3

  # Run full pipeline
  python n3-raft-data-prep.py all --output-dir ./data/n3

  # Dry run (no API calls, use placeholder CoT)
  python n3-raft-data-prep.py all --output-dir ./data/n3 --dry-run

Reference:
  - RAFT: arXiv:2403.10131
  - JaGovFaqs-22k: https://huggingface.co/datasets/matsuxr/JaGovFaqs-22k
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets not installed. Run: pip install datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: no progress bar
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "matsuxr/JaGovFaqs-22k"
TRAIN_SIZE = 1000
TEST_SIZE = 200
TEST_UNSEEN_SIZE = 100
UNANSWERABLE_EXTRA = 100  # Additional "cannot answer" samples
NUM_DISTRACTORS = 4
ORACLE_PROB = 0.8
ANSWER_MIN_LEN = 50
ANSWER_MAX_LEN = 2000
RANDOM_SEED = 42

# CoT generation backend: "ollama", "bedrock", "anthropic"
DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_MODEL = "nemotron-9b-jp-think"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_BEDROCK_MODEL = "apac.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "あなたは税務に関する質問に答えるAIアシスタントです。"
    "提供された参考文書を基に正確に回答してください。"
    "参考文書に回答に必要な情報がない場合は、"
    "「提供された情報からは回答できません」と答えてください。"
    "回答する際は、根拠となる参考文書の該当箇所を引用してから、"
    "簡潔に回答してください。"
)


def build_user_prompt(question: str, documents: list[str]) -> str:
    """Build the user prompt with context documents and question."""
    parts = ["以下の参考文書を基に質問に答えてください。\n"]
    for i, doc in enumerate(documents, 1):
        parts.append(f"参考文書{i}:\n{doc}\n")
    parts.append(f"質問: {question}")
    return "\n".join(parts)


COT_GENERATION_PROMPT = """\
以下の質問と参考文書が与えられています。正解は参考文書{oracle_idx}に含まれています。

質問: {question}

{documents_text}

以下の要件で回答を生成してください。
- 参考文書{oracle_idx}から回答の根拠となる箇所を引用する
- 引用を基に1-2文で推論する
- 最後に簡潔な回答を述べる

出力例:
参考文書Nに「○○」と記載されています。これは△△を意味しており、□□と考えられます。したがって、回答は××です。

上記の形式に従い、自然な日本語で回答を生成してください。形式は厳密に守る必要はありません。"""

# Template responses for "cannot answer" samples (varied for training)
UNANSWERABLE_TEMPLATES = [
    "提供された参考文書を確認しましたが、この質問に直接回答するための情報は含まれていませんでした。提供された情報からは回答できません。",
    "各参考文書を確認しましたが、質問に対する回答に必要な情報が見つかりませんでした。提供された情報からは回答できません。",
    "参考文書の内容を確認しましたが、この質問に該当する記述は見つかりませんでした。提供された情報からは回答できません。",
    "提供された参考文書にはこの質問に回答するための十分な情報が含まれていません。提供された情報からは回答できません。",
    "参考文書を精査しましたが、質問の内容に合致する情報は記載されていませんでした。提供された情報からは回答できません。",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_jagovfaqs() -> list[dict]:
    """Load JaGovFaqs-22k and return as list of dicts."""
    print("Loading JaGovFaqs-22k from HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train")
    data = [row for row in ds]
    print(f"  Loaded {len(data)} records")
    return data


def filter_by_answer_length(
    data: list[dict], min_len: int = ANSWER_MIN_LEN, max_len: int = ANSWER_MAX_LEN
) -> list[dict]:
    """Filter records by answer length."""
    filtered = [r for r in data if min_len <= len(r["Answer"]) <= max_len]
    print(f"  Filtered by answer length ({min_len}-{max_len}): {len(filtered)} records")
    return filtered


# ---------------------------------------------------------------------------
# Phase 1: Explore
# ---------------------------------------------------------------------------


def cmd_explore(args):
    """Explore dataset statistics."""
    data = load_jagovfaqs()

    print("\n=== Dataset Overview ===")
    print(f"Total records: {len(data)}")
    print(f"Fields: {list(data[0].keys())}")

    # Copyright distribution
    copyright_counts = Counter(r["copyright"] for r in data)
    print(f"\n=== Copyright Distribution (top 20) ===")
    print(f"Unique copyrights: {len(copyright_counts)}")
    for name, count in copyright_counts.most_common(20):
        print(f"  {name}: {count}")

    # Answer length distribution
    answer_lengths = [len(r["Answer"]) for r in data]
    print(f"\n=== Answer Length Distribution ===")
    print(f"  Min: {min(answer_lengths)}")
    print(f"  Max: {max(answer_lengths)}")
    print(f"  Mean: {sum(answer_lengths) / len(answer_lengths):.0f}")
    print(f"  Median: {sorted(answer_lengths)[len(answer_lengths) // 2]}")

    # Length buckets
    buckets = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    print("\n  Length buckets:")
    for i in range(len(buckets) - 1):
        count = sum(1 for l in answer_lengths if buckets[i] <= l < buckets[i + 1])
        print(f"    {buckets[i]:>5}-{buckets[i + 1]:>5}: {count:>5} ({count / len(data) * 100:.1f}%)")
    count = sum(1 for l in answer_lengths if l >= buckets[-1])
    print(f"    {buckets[-1]:>5}+     : {count:>5} ({count / len(data) * 100:.1f}%)")

    # Filtered count
    filtered = filter_by_answer_length(data)
    filtered_copyrights = Counter(r["copyright"] for r in filtered)
    print(f"\n=== After Filtering ({ANSWER_MIN_LEN}-{ANSWER_MAX_LEN} chars) ===")
    print(f"  Records: {len(filtered)}")
    print(f"  Copyrights: {len(filtered_copyrights)}")

    # Sample records
    print("\n=== Sample Records ===")
    for i, r in enumerate(random.Random(RANDOM_SEED).sample(data, 3)):
        print(f"\n--- Sample {i + 1} ---")
        print(f"  Q: {r['Question'][:100]}...")
        print(f"  A: {r['Answer'][:100]}...")
        print(f"  Copyright: {r['copyright']}")
        print(f"  URL: {r.get('url', 'N/A')}")


# ---------------------------------------------------------------------------
# Phase 2: Sample
# ---------------------------------------------------------------------------


def stratified_sample(
    data: list[dict],
    n: int,
    rng: random.Random,
    exclude_copyrights: set[str] | None = None,
    exclude_indices: set[int] | None = None,
) -> list[dict]:
    """Stratified sampling by copyright field.

    Proportional allocation: each copyright gets floor(n * ratio) samples,
    remaining slots filled round-robin from largest groups.
    """
    if exclude_copyrights is None:
        exclude_copyrights = set()
    if exclude_indices is None:
        exclude_indices = set()

    # Group by copyright, excluding specified
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, r in enumerate(data):
        if r["copyright"] in exclude_copyrights:
            continue
        if i in exclude_indices:
            continue
        groups[r["copyright"]].append(r)

    total_available = sum(len(v) for v in groups.values())
    if total_available < n:
        print(f"  Warning: only {total_available} available, requested {n}")
        n = total_available

    # Proportional allocation
    sampled = []
    remaining_slots = n
    group_items = sorted(groups.items(), key=lambda x: -len(x[1]))

    allocation = {}
    for name, items in group_items:
        alloc = max(1, int(n * len(items) / total_available))
        alloc = min(alloc, len(items), remaining_slots)
        allocation[name] = alloc
        remaining_slots -= alloc
        if remaining_slots <= 0:
            break

    # Fill remaining slots round-robin
    if remaining_slots > 0:
        for name, items in group_items:
            extra = min(len(items) - allocation.get(name, 0), remaining_slots)
            if extra > 0:
                allocation[name] = allocation.get(name, 0) + extra
                remaining_slots -= extra
            if remaining_slots <= 0:
                break

    # Sample from each group
    for name, alloc in allocation.items():
        items = groups[name]
        sampled.extend(rng.sample(items, min(alloc, len(items))))

    rng.shuffle(sampled)
    return sampled[:n]


def get_record_indices(data: list[dict], sampled: list[dict]) -> set[int]:
    """Get indices of sampled records in the original data."""
    sampled_keys = {(r["Question"], r["Answer"]) for r in sampled}
    return {i for i, r in enumerate(data) if (r["Question"], r["Answer"]) in sampled_keys}


def _sample_single_copyright(
    data: list[dict],
    copyright_name: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Sample train/test/unseen/unanswerable for a single copyright.

    Returns (train, test, test_unseen, unanswerable_extra).
    - train: TRAIN_SIZE from target copyright
    - test: TEST_SIZE from target copyright (no overlap with train)
    - test_unseen: TEST_UNSEEN_SIZE from OTHER copyrights (stratified)
    - unanswerable: UNANSWERABLE_EXTRA from target copyright (remaining)
    """
    target = [r for r in data if r["copyright"] == copyright_name]
    others = [r for r in data if r["copyright"] != copyright_name]

    print(f"\n=== Single Copyright Sampling: {copyright_name} ===")
    print(f"  Target records: {len(target)}")
    print(f"  Other records: {len(others)}")

    needed = TRAIN_SIZE + TEST_SIZE + UNANSWERABLE_EXTRA
    if len(target) < needed:
        print(f"  Warning: {copyright_name} has {len(target)} records, need {needed}")

    # Shuffle target pool
    target_shuffled = list(target)
    rng.shuffle(target_shuffled)

    # 1. Train: first TRAIN_SIZE
    train = target_shuffled[:TRAIN_SIZE]
    train_questions = {r["Question"] for r in train}

    # 2. Test: next TEST_SIZE (no question overlap with train)
    remaining = [r for r in target_shuffled[TRAIN_SIZE:] if r["Question"] not in train_questions]
    test = remaining[:TEST_SIZE]
    test_questions = {r["Question"] for r in test}

    # 3. Test-unseen: stratified from other copyrights
    test_unseen = stratified_sample(others, TEST_UNSEEN_SIZE, rng)

    # 4. Unanswerable: from remaining target records
    used_questions = train_questions | test_questions
    unanswerable_pool = [r for r in target if r["Question"] not in used_questions]
    unanswerable_extra = rng.sample(
        unanswerable_pool, min(UNANSWERABLE_EXTRA, len(unanswerable_pool))
    )

    return train, test, test_unseen, unanswerable_extra


def cmd_sample(args):
    """Sample train/test splits with stratification."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    data = load_jagovfaqs()
    data = filter_by_answer_length(data)

    # Branch: single copyright or stratified
    copyright_filter = getattr(args, "copyright_filter", None)
    if copyright_filter:
        train, test, test_unseen, unanswerable_extra = _sample_single_copyright(
            data, copyright_filter, rng
        )
    else:
        # --- Original stratified sampling ---
        # Get all copyrights
        all_copyrights = set(r["copyright"] for r in data)
        copyright_counts = Counter(r["copyright"] for r in data)

        # Determine "unseen" copyrights for test-unseen
        # Use copyrights with enough samples but not too many (minority groups)
        sorted_copyrights = sorted(copyright_counts.items(), key=lambda x: x[1])
        unseen_copyrights = set()
        unseen_pool_size = 0
        for name, count in sorted_copyrights:
            if count < 5:
                continue  # Too few for meaningful test
            unseen_copyrights.add(name)
            unseen_pool_size += count
            if unseen_pool_size >= TEST_UNSEEN_SIZE * 3:  # Need 3x for good sampling
                break

        print(f"\n=== Sampling Strategy ===")
        print(f"  Unseen copyrights ({len(unseen_copyrights)}): {unseen_copyrights}")
        print(f"  Unseen pool size: {unseen_pool_size}")

        # 1. Sample test-unseen from unseen copyrights
        unseen_data = [r for r in data if r["copyright"] in unseen_copyrights]
        test_unseen = rng.sample(unseen_data, min(TEST_UNSEEN_SIZE, len(unseen_data)))
        unseen_questions = {r["Question"] for r in test_unseen}

        # 2. Sample train from remaining copyrights (not unseen)
        # Exclude records whose Question appeared in test-unseen
        train_pool = [
            r for r in data
            if r["copyright"] not in unseen_copyrights
            and r["Question"] not in unseen_questions
        ]
        train = stratified_sample(train_pool, TRAIN_SIZE, rng)
        train_questions = {r["Question"] for r in train}

        # 3. Sample test from remaining (exclude by Question text, not index)
        test_pool = [
            r for r in data
            if r["copyright"] not in unseen_copyrights
            and r["Question"] not in train_questions
            and r["Question"] not in unseen_questions
        ]
        test = rng.sample(test_pool, min(TEST_SIZE, len(test_pool)))
        test_questions = {r["Question"] for r in test}

        # 4. Sample extra "cannot answer" source questions
        used_questions = train_questions | test_questions | unseen_questions
        unanswerable_pool = [
            r for r in data
            if r["copyright"] not in unseen_copyrights
            and r["Question"] not in used_questions
        ]
        unanswerable_extra = rng.sample(
            unanswerable_pool, min(UNANSWERABLE_EXTRA, len(unanswerable_pool))
        )

    # Report
    print(f"\n=== Sampling Results ===")
    print(f"  Train: {len(train)} samples")
    print(f"    Copyrights: {len(set(r['copyright'] for r in train))}")
    print(f"  Test: {len(test)} samples")
    print(f"    Copyrights: {len(set(r['copyright'] for r in test))}")
    print(f"  Test-unseen: {len(test_unseen)} samples")
    print(f"    Copyrights: {len(set(r['copyright'] for r in test_unseen))}")
    print(f"  Unanswerable extra: {len(unanswerable_extra)} samples")

    # Verify no overlap
    train_qs = {r["Question"] for r in train}
    test_qs = {r["Question"] for r in test}
    unseen_qs = {r["Question"] for r in test_unseen}
    assert len(train_qs & test_qs) == 0, "Train/test overlap!"
    assert len(train_qs & unseen_qs) == 0, "Train/unseen overlap!"
    assert len(test_qs & unseen_qs) == 0, "Test/unseen overlap!"
    print("  No overlap between splits: OK")

    # Save
    for name, split_data in [
        ("sampled_train.json", train),
        ("sampled_test.json", test),
        ("sampled_test_unseen.json", test_unseen),
        ("sampled_unanswerable.json", unanswerable_extra),
    ]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Phase 3: Generate RAFT Training Data
# ---------------------------------------------------------------------------


def select_distractors(
    record: dict,
    pool: list[dict],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Select n distractor documents.

    Prefer same-copyright (same ministry) distractors for harder negatives.
    """
    same_copyright = [
        r["Answer"]
        for r in pool
        if r["copyright"] == record["copyright"] and r["Question"] != record["Question"]
    ]

    if len(same_copyright) >= n:
        return rng.sample(same_copyright, n)

    # Fall back to mix of same-copyright and random
    distractors = list(same_copyright)
    other = [
        r["Answer"]
        for r in pool
        if r["copyright"] != record["copyright"] and r["Question"] != record["Question"]
    ]
    needed = n - len(distractors)
    distractors.extend(rng.sample(other, min(needed, len(other))))

    return distractors[:n]


def _build_cot_prompt(question: str, documents: list[str], oracle_idx: int) -> str:
    """Build CoT generation prompt."""
    docs_text = ""
    for i, doc in enumerate(documents, 1):
        docs_text += f"参考文書{i}:\n{doc}\n\n"
    return COT_GENERATION_PROMPT.format(
        oracle_idx=oracle_idx,
        question=question,
        documents_text=docs_text.strip(),
    )


def generate_cot_ollama(
    question: str,
    documents: list[str],
    oracle_idx: int,
    model: str,
    base_url: str,
) -> str:
    """Generate CoT answer using Ollama."""
    prompt = _build_cot_prompt(question, documents, oracle_idx)
    resp = _requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    # Strip <think>...</think> block if present
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def generate_cot_bedrock(
    question: str,
    documents: list[str],
    oracle_idx: int,
    model: str,
) -> str:
    """Generate CoT answer using Amazon Bedrock (Claude)."""
    import boto3

    client = boto3.client("bedrock-runtime")
    prompt = _build_cot_prompt(question, documents, oracle_idx)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=model, body=body)
    result = json.loads(resp["body"].read())
    return result["content"][0]["text"].strip()


def generate_cot_anthropic(
    question: str,
    documents: list[str],
    oracle_idx: int,
    model: str,
    client,
) -> str:
    """Generate CoT answer using Anthropic API."""
    prompt = _build_cot_prompt(question, documents, oracle_idx)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_cot(
    question: str,
    documents: list[str],
    oracle_idx: int,
    backend: str,
    model: str,
    client=None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """Generate CoT answer using the specified backend."""
    if backend == "ollama":
        return generate_cot_ollama(question, documents, oracle_idx, model, ollama_url)
    elif backend == "bedrock":
        return generate_cot_bedrock(question, documents, oracle_idx, model)
    elif backend == "anthropic":
        return generate_cot_anthropic(question, documents, oracle_idx, model, client)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def generate_placeholder_cot(
    question: str,
    documents: list[str],
    oracle_idx: int,
    oracle_answer: str,
) -> str:
    """Generate placeholder CoT for dry-run mode."""
    # Extract first sentence of oracle as citation
    first_sentence = oracle_answer.split("。")[0] + "。" if "。" in oracle_answer else oracle_answer[:100]
    return (
        f"参考文書{oracle_idx}に「{first_sentence[:80]}」と記載されています。"
        f"この記述に基づくと、質問に対する回答が得られます。"
        f"したがって、{oracle_answer[:100]}です。"
    )


def build_raft_sample(
    record: dict,
    pool: list[dict],
    rng: random.Random,
    include_oracle: bool,
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_OLLAMA_MODEL,
    client=None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    dry_run: bool = False,
) -> dict:
    """Build a single RAFT training sample.

    Returns a dict with 'messages' key in SFT format.
    """
    question = record["Question"]
    oracle_answer = record["Answer"]
    distractors = select_distractors(record, pool, NUM_DISTRACTORS, rng)

    if include_oracle:
        # Insert oracle at random position among distractors
        oracle_pos = rng.randint(0, len(distractors))
        documents = list(distractors)
        documents.insert(oracle_pos, oracle_answer)
        oracle_idx = oracle_pos + 1  # 1-indexed

        # Generate CoT answer
        if dry_run:
            assistant_content = generate_placeholder_cot(
                question, documents, oracle_idx, oracle_answer
            )
        else:
            assistant_content = generate_cot(
                question, documents, oracle_idx,
                backend=backend, model=model, client=client,
                ollama_url=ollama_url,
            )
    else:
        # No oracle: all distractors
        # Add one more distractor to fill the oracle slot
        extra = select_distractors(record, pool, 1, rng)
        documents = list(distractors) + extra
        rng.shuffle(documents)
        documents = documents[: NUM_DISTRACTORS + 1]  # Keep 5 total

        # "Cannot answer" response
        assistant_content = rng.choice(UNANSWERABLE_TEMPLATES)

    user_content = build_user_prompt(question, documents)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "JaGovFaqs-22k",
            "copyright": record["copyright"],
            "has_oracle": include_oracle,
            "question": question,
        },
    }


def build_unanswerable_sample(
    record: dict,
    pool: list[dict],
    rng: random.Random,
) -> dict:
    """Build a "cannot answer" sample using only distractors."""
    question = record["Question"]

    # Select 5 distractors (no oracle)
    distractors = select_distractors(record, pool, NUM_DISTRACTORS + 1, rng)
    documents = distractors[: NUM_DISTRACTORS + 1]
    rng.shuffle(documents)

    user_content = build_user_prompt(question, documents)
    assistant_content = rng.choice(UNANSWERABLE_TEMPLATES)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "JaGovFaqs-22k",
            "copyright": record["copyright"],
            "has_oracle": False,
            "question": question,
            "is_extra_unanswerable": True,
        },
    }


def build_test_sample(record: dict, pool: list[dict], rng: random.Random) -> dict:
    """Build a test sample (always with oracle, no CoT)."""
    question = record["Question"]
    oracle_answer = record["Answer"]
    distractors = select_distractors(record, pool, NUM_DISTRACTORS, rng)

    oracle_pos = rng.randint(0, len(distractors))
    documents = list(distractors)
    documents.insert(oracle_pos, oracle_answer)

    user_content = build_user_prompt(question, documents)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "metadata": {
            "source": "JaGovFaqs-22k",
            "copyright": record["copyright"],
            "question": question,
            "expected_answer": oracle_answer,
            "oracle_position": oracle_pos + 1,
        },
    }


def cmd_generate(args):
    """Generate RAFT training data."""
    output_dir = Path(args.output_dir)

    # Load sampled data
    train_path = output_dir / "sampled_train.json"
    test_path = output_dir / "sampled_test.json"
    unseen_path = output_dir / "sampled_test_unseen.json"
    unanswerable_path = output_dir / "sampled_unanswerable.json"

    for p in [train_path, test_path, unseen_path, unanswerable_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run 'sample' first.")
            sys.exit(1)

    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    with open(unseen_path, encoding="utf-8") as f:
        unseen_data = json.load(f)
    with open(unanswerable_path, encoding="utf-8") as f:
        unanswerable_data = json.load(f)

    rng = random.Random(RANDOM_SEED)
    dry_run = args.dry_run
    backend = args.backend
    ollama_url = getattr(args, "ollama_url", DEFAULT_OLLAMA_URL)

    # Resolve model name based on backend
    if hasattr(args, "model") and args.model:
        model = args.model
    else:
        model = {
            "ollama": DEFAULT_OLLAMA_MODEL,
            "bedrock": DEFAULT_BEDROCK_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
        }.get(backend, DEFAULT_OLLAMA_MODEL)

    # Initialize backend client (if not dry-run)
    client = None
    if not dry_run:
        if backend == "ollama":
            if _requests is None:
                print("Error: requests not installed. Run: pip install requests")
                sys.exit(1)
            # Verify Ollama is reachable
            try:
                resp = _requests.get(f"{ollama_url}/api/tags", timeout=5)
                resp.raise_for_status()
                print(f"Ollama backend ready (model: {model}, url: {ollama_url})")
            except Exception as e:
                print(f"Error: Ollama not reachable at {ollama_url}: {e}")
                sys.exit(1)
        elif backend == "bedrock":
            try:
                import boto3
                boto3.client("bedrock-runtime")
                print(f"Bedrock backend ready (model: {model})")
            except Exception as e:
                print(f"Error: Bedrock not available: {e}")
                sys.exit(1)
        elif backend == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic()
                print(f"Anthropic API ready (model: {model})")
            except Exception as e:
                print(f"Error: Anthropic API not available: {e}")
                sys.exit(1)

    # Use all train data as distractor pool
    distractor_pool = train_data + unanswerable_data

    # --- Generate train RAFT samples ---
    print(f"\n=== Generating RAFT Train Data ===")
    print(f"  Total: {len(train_data)} RAFT + {len(unanswerable_data)} unanswerable")
    print(f"  Oracle probability: {ORACLE_PROB}")
    print(f"  Backend: {backend} (model: {model})")
    print(f"  Dry run: {dry_run}")

    train_samples = []
    oracle_count = 0
    no_oracle_count = 0

    # Checkpoint: resume from existing progress
    checkpoint_path = output_dir / "train_checkpoint.jsonl"
    start_idx = 0
    if checkpoint_path.exists() and not dry_run:
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                train_samples.append(json.loads(line))
        start_idx = len(train_samples)
        oracle_count = sum(1 for s in train_samples if s["metadata"]["has_oracle"])
        no_oracle_count = start_idx - oracle_count
        print(f"  Resuming from checkpoint: {start_idx} samples done")

    for i in tqdm(range(start_idx, len(train_data)), desc="RAFT samples"):
        record = train_data[i]
        include_oracle = rng.random() < ORACLE_PROB

        sample = build_raft_sample(
            record=record,
            pool=distractor_pool,
            rng=rng,
            include_oracle=include_oracle,
            backend=backend,
            model=model,
            client=client,
            ollama_url=ollama_url,
            dry_run=dry_run,
        )
        train_samples.append(sample)

        if include_oracle:
            oracle_count += 1
        else:
            no_oracle_count += 1

        # Save checkpoint every 50 samples (for API interruption recovery)
        if not dry_run and (i + 1) % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for s in train_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Rate limiting for API calls
        if not dry_run and include_oracle:
            time.sleep(0.1)

    # Add extra unanswerable samples
    for record in tqdm(unanswerable_data, desc="Unanswerable"):
        sample = build_unanswerable_sample(record, distractor_pool, rng)
        train_samples.append(sample)
        no_oracle_count += 1

    # Shuffle all train samples
    rng.shuffle(train_samples)

    print(f"\n  Generated: {len(train_samples)} train samples")
    print(f"    With oracle: {oracle_count} ({oracle_count / len(train_samples) * 100:.1f}%)")
    print(f"    Without oracle: {no_oracle_count} ({no_oracle_count / len(train_samples) * 100:.1f}%)")

    # Save train JSONL
    train_jsonl_path = output_dir / "train.jsonl"
    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Saved: {train_jsonl_path}")

    # --- Generate test samples ---
    print(f"\n=== Generating Test Data ===")

    test_pool = train_data + test_data  # Wider pool for distractors

    for split_name, split_data in [("test", test_data), ("test_unseen", unseen_data)]:
        samples = []
        for record in tqdm(split_data, desc=f"Test ({split_name})"):
            sample = build_test_sample(record, test_pool, rng)
            samples.append(sample)

        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  Saved: {path} ({len(samples)} samples)")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Cleaned up checkpoint file")

    # --- Statistics ---
    print(f"\n=== Output Summary ===")
    print(f"  train.jsonl: {len(train_samples)} samples")
    print(f"  test.jsonl: {len(test_data)} samples")
    print(f"  test_unseen.jsonl: {len(unseen_data)} samples")

    # Token estimate
    total_chars = sum(
        sum(len(m["content"]) for m in s["messages"]) for s in train_samples
    )
    est_tokens = total_chars / 2  # Rough: ~2 chars per token for Japanese
    print(f"\n  Estimated train tokens: {est_tokens:,.0f}")
    print(f"  Avg tokens per sample: {est_tokens / len(train_samples):,.0f}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def cmd_all(args):
    """Run full pipeline."""
    print("=" * 60)
    print("Phase 1: Explore")
    print("=" * 60)
    cmd_explore(args)

    print("\n" + "=" * 60)
    print("Phase 2: Sample")
    print("=" * 60)
    cmd_sample(args)

    print("\n" + "=" * 60)
    print("Phase 3: Generate")
    print("=" * 60)
    cmd_generate(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="N3 RAFT Data Preparation: JaGovFaqs-22k -> RAFT SFT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # explore
    sub_explore = subparsers.add_parser("explore", help="Explore dataset statistics")
    sub_explore.set_defaults(func=cmd_explore)

    # sample
    sub_sample = subparsers.add_parser("sample", help="Sample train/test splits")
    sub_sample.add_argument(
        "--output-dir", default="./data/n3", help="Output directory (default: ./data/n3)"
    )
    sub_sample.add_argument(
        "--copyright-filter", default=None,
        help="Filter to specific copyright (e.g., '国税庁')"
    )
    sub_sample.set_defaults(func=cmd_sample)

    # Shared backend options
    def add_backend_args(sub):
        sub.add_argument(
            "--output-dir", default="./data/n3", help="Output directory (default: ./data/n3)"
        )
        sub.add_argument(
            "--dry-run", action="store_true", help="Use placeholder CoT (no LLM calls)"
        )
        sub.add_argument(
            "--backend", default=DEFAULT_BACKEND, choices=["ollama", "bedrock", "anthropic"],
            help=f"CoT generation backend (default: {DEFAULT_BACKEND})"
        )
        sub.add_argument(
            "--model", default=None,
            help="Model name (default: auto-select per backend)"
        )
        sub.add_argument(
            "--ollama-url", default=DEFAULT_OLLAMA_URL,
            help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})"
        )
        sub.add_argument(
            "--copyright-filter", default=None,
            help="Filter to specific copyright (e.g., '国税庁')"
        )

    # generate
    sub_generate = subparsers.add_parser("generate", help="Generate RAFT training data")
    add_backend_args(sub_generate)
    sub_generate.set_defaults(func=cmd_generate)

    # all
    sub_all = subparsers.add_parser("all", help="Run full pipeline")
    add_backend_args(sub_all)
    sub_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
