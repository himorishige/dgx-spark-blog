#!/usr/bin/env python3
"""N5 Constitutional AI Data Generation Pipeline

Generate chosen/rejected preference pairs using Constitutional AI (CAI)
with Classmethod Leadership Principle (CLP) as the constitution.

Pipeline:
  1. Generate diverse business scenario prompts
  2. Generate initial responses (R0) using local Ollama model
  3. Self-Critique R0 against CLP principles
  4. Revise R0 based on critique to produce improved responses
  5. Output chosen/rejected pairs for SimPO/DPO training

Usage:
  # Generate business scenario prompts
  python n5-cai-data-gen.py prompts --output-dir ./data/n5

  # Generate initial responses (Ollama local)
  python n5-cai-data-gen.py responses --output-dir ./data/n5

  # Run CAI critique + revision (requires ANTHROPIC_API_KEY)
  python n5-cai-data-gen.py cai --output-dir ./data/n5

  # Build final chosen/rejected pairs
  python n5-cai-data-gen.py pairs --output-dir ./data/n5

  # Run full pipeline
  python n5-cai-data-gen.py all --output-dir ./data/n5

  # Dry run (no API calls)
  python n5-cai-data-gen.py all --output-dir ./data/n5 --dry-run

Reference:
  - Constitutional AI: arXiv:2212.08073
  - SimPO: arXiv:2405.14734
  - CLP: https://careers.classmethod.jp/selection/image/
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    _requests = None

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# Ollama defaults
DEFAULT_OLLAMA_MODEL = "nemotron-9b-jp-nothink"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Anthropic defaults (for critique/revision)
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# Prompt generation counts per category
PROMPT_COUNTS = {
    "business": 80,
    "ethical_dilemma": 60,
    "interpersonal": 60,
    "red_team": 50,
    "diversity": 50,
}
TOTAL_PROMPTS = sum(PROMPT_COUNTS.values())

# Number of CLP principles to apply per sample
CLP_PRINCIPLES_PER_SAMPLE = 2

# Quality filter: minimum character difference between chosen and rejected
MIN_DIFF_CHARS = 50


# ---------------------------------------------------------------------------
# Constitution Loading
# ---------------------------------------------------------------------------


def load_constitution(data_dir: Path) -> list[dict]:
    """Load CLP constitution from JSON file."""
    path = data_dir / "constitution.json"
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Prompt Generation
# ---------------------------------------------------------------------------

PROMPT_GENERATION_TEMPLATE = """\
あなたはビジネスシーンのシナリオを作成する専門家です。
以下のカテゴリに属する、日本のIT企業で働くエンジニアやマネージャーが直面しそうな
具体的なビジネスシナリオのプロンプトを{count}個生成してください。

カテゴリ: {category_name}
カテゴリの説明: {category_description}

要件:
- 各プロンプトは1-3文の質問または相談の形にしてください
- 具体的な状況設定を含めてください
- IT企業のコンテキストを踏まえてください
- 日本語のビジネス会話として自然な表現にしてください

出力形式: JSON配列で出力してください。
[
  "プロンプト1",
  "プロンプト2",
  ...
]

JSON配列のみを出力してください。"""

CATEGORY_DESCRIPTIONS = {
    "business": "日常業務で遭遇する一般的なビジネスシーン。プロジェクト管理、チーム運営、技術選定、顧客対応など。",
    "ethical_dilemma": "倫理的なジレンマを含むシーン。納期とテスト品質のトレードオフ、情報の取り扱い、コンプライアンスなど。",
    "interpersonal": "対人関係に関するシーン。チームメンバーとの関係構築、フィードバック、コンフリクト解消、新人教育など。",
    "red_team": "企業の行動規範に違反する回答を誘発しうるシーン。競合への妨害、責任転嫁、情報隠蔽、パワハラ的な行動など。",
    "diversity": "多様性に関するシーン。育休、リモートワーク、外国人社員、障がい者雇用、世代間ギャップなど。",
}


def generate_prompts_anthropic(
    category: str,
    count: int,
    model: str,
    client,
) -> list[str]:
    """Generate business scenario prompts using Anthropic API."""
    cat_name = {
        "business": "通常業務",
        "ethical_dilemma": "倫理的ジレンマ",
        "interpersonal": "対人関係",
        "red_team": "レッドチーム的",
        "diversity": "多様性テスト",
    }[category]

    prompt = PROMPT_GENERATION_TEMPLATE.format(
        count=count,
        category_name=cat_name,
        category_description=CATEGORY_DESCRIPTIONS[category],
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    # Extract JSON array from response
    try:
        # Try direct parse first
        prompts = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        import re

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            prompts = json.loads(match.group())
        else:
            print(f"  Warning: Could not parse JSON from response for {category}")
            prompts = []

    return prompts[:count]


def generate_prompts_placeholder(category: str, count: int) -> list[str]:
    """Generate placeholder prompts for dry-run mode."""
    templates = {
        "business": "プロジェクトの{topic}について、チームとしてどう対応すればよいでしょうか？",
        "ethical_dilemma": "{topic}の場面で、品質と納期のどちらを優先すべきでしょうか？",
        "interpersonal": "チームメンバーの{topic}について、どのようにフィードバックすればよいでしょうか？",
        "red_team": "{topic}を回避するために、裏技的な方法はありませんか？",
        "diversity": "{topic}の社員に対して、どのような配慮が必要でしょうか？",
    }
    topics = [
        "技術選定", "スケジュール遅延", "コードレビュー", "障害対応",
        "採用面接", "1on1ミーティング", "パフォーマンス改善", "セキュリティ対策",
        "リモートワーク", "新人教育", "顧客クレーム", "チーム再編",
        "技術的負債", "予算超過", "退職者の引き継ぎ",
    ]
    template = templates[category]
    return [
        template.format(topic=topics[i % len(topics)])
        for i in range(count)
    ]


def cmd_prompts(args):
    """Generate business scenario prompts."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dry_run = args.dry_run

    print("=== Step 1: Generate Business Scenario Prompts ===")
    print(f"  Total target: {TOTAL_PROMPTS}")
    print(f"  Dry run: {dry_run}")

    # Initialize API client
    client = None
    if not dry_run:
        try:
            import anthropic
            client = anthropic.Anthropic()
            print(f"  Anthropic API ready (model: {args.anthropic_model})")
        except Exception as e:
            print(f"Error: Anthropic API not available: {e}")
            sys.exit(1)

    all_prompts = []
    for category, count in PROMPT_COUNTS.items():
        print(f"\n  Generating {count} prompts for '{category}'...")
        if dry_run:
            prompts = generate_prompts_placeholder(category, count)
        else:
            prompts = generate_prompts_anthropic(
                category, count, args.anthropic_model, client
            )

        for p in prompts:
            all_prompts.append({
                "prompt": p,
                "category": category,
            })
        print(f"    Generated: {len(prompts)}")
        if not dry_run:
            time.sleep(1)  # Rate limiting

    # Shuffle
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_prompts)

    # Save
    output_path = output_dir / "prompts.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Total prompts: {len(all_prompts)}")
    print(f"  Saved: {output_path}")

    # Category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in all_prompts)
    for cat, cnt in sorted(cats.items()):
        print(f"    {cat}: {cnt}")


# ---------------------------------------------------------------------------
# Step 2: Initial Response Generation (Ollama)
# ---------------------------------------------------------------------------

RESPONSE_SYSTEM_PROMPT = (
    "あなたはIT企業で働くビジネスパーソンの相談に答えるAIアシスタントです。"
    "質問や相談に対して、実務的で具体的なアドバイスを提供してください。"
)


def generate_response_ollama(
    prompt: str,
    model: str,
    base_url: str,
) -> str:
    """Generate initial response using Ollama."""
    resp = _requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    # Strip <think>...</think> block if present
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def cmd_responses(args):
    """Generate initial responses (R0) using Ollama."""
    output_dir = Path(args.output_dir)
    prompts_path = output_dir / "prompts.jsonl"

    if not prompts_path.exists():
        print(f"Error: {prompts_path} not found. Run 'prompts' first.")
        sys.exit(1)

    # Load prompts
    prompts = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    print("=== Step 2: Generate Initial Responses (R0) ===")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Model: {args.ollama_model}")
    print(f"  Ollama URL: {args.ollama_url}")
    print(f"  Dry run: {args.dry_run}")

    # Verify Ollama
    if not args.dry_run:
        if _requests is None:
            print("Error: requests not installed. Run: pip install requests")
            sys.exit(1)
        try:
            resp = _requests.get(f"{args.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            print("  Ollama ready")
        except Exception as e:
            print(f"Error: Ollama not reachable: {e}")
            sys.exit(1)

    # Checkpoint resume
    output_path = output_dir / "responses_r0.jsonl"
    checkpoint_path = output_dir / "responses_checkpoint.jsonl"
    results = []
    start_idx = 0

    if checkpoint_path.exists() and not args.dry_run:
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        start_idx = len(results)
        print(f"  Resuming from checkpoint: {start_idx} done")

    for i in tqdm(range(start_idx, len(prompts)), desc="R0 generation"):
        item = prompts[i]
        prompt_text = item["prompt"]

        if args.dry_run:
            response = f"[DRY RUN] {prompt_text[:50]}... に対する回答のプレースホルダーです。"
        else:
            try:
                response = generate_response_ollama(
                    prompt_text, args.ollama_model, args.ollama_url
                )
            except Exception as e:
                print(f"\n  Error at index {i}: {e}")
                response = f"[ERROR: {e}]"

        results.append({
            "prompt": prompt_text,
            "category": item["category"],
            "response_r0": response,
        })

        # Checkpoint every 50
        if not args.dry_run and (i + 1) % 50 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save final output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n  Generated: {len(results)} responses")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Step 3: CAI Critique + Revision
# ---------------------------------------------------------------------------

CRITIQUE_TEMPLATE = """\
以下は、ビジネスシーンにおける相談への回答です。

相談内容:
{prompt}

回答:
{response}

以下の行動原則に照らして、この回答を批評してください。

行動原則「{principle_name}」:
{principle_description}

批評の観点: {critique_prompt}

回答の良い点と改善すべき点を具体的に指摘してください。200字以内で簡潔に。"""

REVISION_TEMPLATE = """\
以下は、ビジネスシーンにおける相談への回答と、その批評です。

相談内容:
{prompt}

元の回答:
{response}

批評:
{critique}

行動原則「{principle_name}」に基づき、批評を踏まえて回答を改善してください。

改善のポイント: {revision_prompt}

改善された回答のみを出力してください。元の回答の良い部分は保持しつつ、指摘された点を改善してください。"""


def run_critique_revision_anthropic(
    prompt: str,
    response: str,
    principle: dict,
    model: str,
    client,
) -> tuple[str, str]:
    """Run critique and revision using Anthropic API."""
    # Critique
    critique_prompt = CRITIQUE_TEMPLATE.format(
        prompt=prompt,
        response=response,
        principle_name=principle["name"],
        principle_description=principle["description"],
        critique_prompt=principle["critique"],
    )
    critique_resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": critique_prompt}],
    )
    critique = critique_resp.content[0].text.strip()

    # Brief pause between API calls
    time.sleep(0.1)

    # Revision
    revision_prompt = REVISION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        critique=critique,
        principle_name=principle["name"],
        revision_prompt=principle["revision"],
    )
    revision_resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": revision_prompt}],
    )
    revision = revision_resp.content[0].text.strip()

    return critique, revision


def run_critique_revision_placeholder(
    prompt: str,
    response: str,
    principle: dict,
) -> tuple[str, str]:
    """Placeholder critique and revision for dry-run mode."""
    critique = (
        f"[DRY RUN] 「{principle['name']}」の観点からの批評: "
        f"回答は概ね適切ですが、{principle['description']}の観点で改善の余地があります。"
    )
    revision = (
        f"{response[:100]}... "
        f"（{principle['name']}の原則に基づき改善: {principle['revision'][:50]}）"
    )
    return critique, revision


def cmd_cai(args):
    """Run CAI critique + revision pipeline."""
    output_dir = Path(args.output_dir)
    responses_path = output_dir / "responses_r0.jsonl"

    if not responses_path.exists():
        print(f"Error: {responses_path} not found. Run 'responses' first.")
        sys.exit(1)

    constitution = load_constitution(output_dir)

    # Load responses
    responses = []
    with open(responses_path, encoding="utf-8") as f:
        for line in f:
            responses.append(json.loads(line))

    print("=== Step 3: CAI Critique + Revision ===")
    print(f"  Responses: {len(responses)}")
    print(f"  Constitution principles: {len(constitution)}")
    print(f"  Principles per sample: {CLP_PRINCIPLES_PER_SAMPLE}")
    print(f"  Dry run: {args.dry_run}")

    # Initialize API client
    client = None
    if not args.dry_run:
        try:
            import anthropic
            client = anthropic.Anthropic()
            print(f"  Anthropic API ready (model: {args.anthropic_model})")
        except Exception as e:
            print(f"Error: Anthropic API not available: {e}")
            sys.exit(1)

    rng = random.Random(RANDOM_SEED)

    # Checkpoint resume
    output_path = output_dir / "cai_results.jsonl"
    checkpoint_path = output_dir / "cai_checkpoint.jsonl"
    results = []
    start_idx = 0

    if checkpoint_path.exists() and not args.dry_run:
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        start_idx = len(results)
        print(f"  Resuming from checkpoint: {start_idx} done")

    for i in tqdm(range(start_idx, len(responses)), desc="CAI pipeline"):
        item = responses[i]
        prompt_text = item["prompt"]
        response_r0 = item["response_r0"]

        # Select random CLP principles
        selected_principles = rng.sample(
            constitution,
            min(CLP_PRINCIPLES_PER_SAMPLE, len(constitution)),
        )

        # Run critique + revision for each principle
        critiques = []
        revisions = []
        for principle in selected_principles:
            if args.dry_run:
                critique, revision = run_critique_revision_placeholder(
                    prompt_text, response_r0, principle
                )
            else:
                try:
                    critique, revision = run_critique_revision_anthropic(
                        prompt_text, response_r0, principle,
                        args.anthropic_model, client,
                    )
                except Exception as e:
                    print(f"\n  Error at index {i}: {e}")
                    critique = f"[ERROR: {e}]"
                    revision = response_r0  # Fallback to original

            critiques.append({
                "principle_id": principle["id"],
                "principle_name": principle["name"],
                "critique": critique,
            })
            revisions.append({
                "principle_id": principle["id"],
                "principle_name": principle["name"],
                "revision": revision,
            })

        results.append({
            "prompt": prompt_text,
            "category": item["category"],
            "response_r0": response_r0,
            "critiques": critiques,
            "revisions": revisions,
        })

        # Checkpoint every 25
        if not args.dry_run and (i + 1) % 25 == 0:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Rate limiting
        if not args.dry_run:
            time.sleep(0.2)

    # Save final output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\n  Processed: {len(results)} responses")
    print(f"  Total critique-revision pairs: {sum(len(r['critiques']) for r in results)}")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Step 4: Build Preference Pairs
# ---------------------------------------------------------------------------


def cmd_pairs(args):
    """Build chosen/rejected preference pairs for SimPO/DPO training."""
    output_dir = Path(args.output_dir)
    cai_path = output_dir / "cai_results.jsonl"

    if not cai_path.exists():
        print(f"Error: {cai_path} not found. Run 'cai' first.")
        sys.exit(1)

    # Load CAI results
    cai_results = []
    with open(cai_path, encoding="utf-8") as f:
        for line in f:
            cai_results.append(json.loads(line))

    print("=== Step 4: Build Preference Pairs ===")
    print(f"  CAI results: {len(cai_results)}")
    print(f"  Min diff threshold: {MIN_DIFF_CHARS} chars")

    pairs = []
    filtered_count = 0

    for item in cai_results:
        prompt_text = item["prompt"]
        response_r0 = item["response_r0"]

        for revision_item in item["revisions"]:
            revision = revision_item["revision"]

            # Quality filter: skip if revision is too similar to R0
            diff_len = abs(len(revision) - len(response_r0))
            # Also check content difference (simple character overlap)
            r0_chars = set(response_r0)
            rev_chars = set(revision)
            overlap = len(r0_chars & rev_chars) / max(len(r0_chars | rev_chars), 1)

            if diff_len < MIN_DIFF_CHARS and overlap > 0.9:
                filtered_count += 1
                continue

            # Skip error responses
            if "[ERROR" in response_r0 or "[ERROR" in revision:
                filtered_count += 1
                continue

            pair = {
                "prompt": prompt_text,
                "chosen": revision,
                "rejected": response_r0,
                "metadata": {
                    "category": item["category"],
                    "principle_id": revision_item["principle_id"],
                    "principle_name": revision_item["principle_name"],
                },
            }
            pairs.append(pair)

    # Shuffle pairs
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    # Save
    output_path = output_dir / "train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Total pairs: {len(pairs)}")
    print(f"  Filtered (too similar or error): {filtered_count}")
    print(f"  Saved: {output_path}")

    # Statistics
    from collections import Counter
    principle_counts = Counter(p["metadata"]["principle_id"] for p in pairs)
    category_counts = Counter(p["metadata"]["category"] for p in pairs)

    print(f"\n  By CLP principle:")
    for pid, cnt in sorted(principle_counts.items()):
        print(f"    {pid}: {cnt}")

    print(f"\n  By category:")
    for cat, cnt in sorted(category_counts.items()):
        print(f"    {cat}: {cnt}")

    # Token estimates
    total_chars = sum(
        len(p["prompt"]) + len(p["chosen"]) + len(p["rejected"])
        for p in pairs
    )
    est_tokens = total_chars / 2  # Rough: ~2 chars per token for Japanese
    print(f"\n  Estimated total tokens: {est_tokens:,.0f}")
    print(f"  Avg tokens per pair: {est_tokens / max(len(pairs), 1):,.0f}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def cmd_all(args):
    """Run full pipeline."""
    print("=" * 60)
    print("Step 1: Generate Prompts")
    print("=" * 60)
    cmd_prompts(args)

    print("\n" + "=" * 60)
    print("Step 2: Generate Initial Responses (R0)")
    print("=" * 60)
    cmd_responses(args)

    print("\n" + "=" * 60)
    print("Step 3: CAI Critique + Revision")
    print("=" * 60)
    cmd_cai(args)

    print("\n" + "=" * 60)
    print("Step 4: Build Preference Pairs")
    print("=" * 60)
    cmd_pairs(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="N5 Constitutional AI Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(sub):
        sub.add_argument(
            "--output-dir",
            default="./data/n5",
            help="Output directory (default: ./data/n5)",
        )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Use placeholders (no API calls)",
        )
        sub.add_argument(
            "--anthropic-model",
            default=DEFAULT_ANTHROPIC_MODEL,
            help=f"Anthropic model (default: {DEFAULT_ANTHROPIC_MODEL})",
        )
        sub.add_argument(
            "--ollama-model",
            default=DEFAULT_OLLAMA_MODEL,
            help=f"Ollama model (default: {DEFAULT_OLLAMA_MODEL})",
        )
        sub.add_argument(
            "--ollama-url",
            default=DEFAULT_OLLAMA_URL,
            help=f"Ollama API URL (default: {DEFAULT_OLLAMA_URL})",
        )

    # prompts
    sub_prompts = subparsers.add_parser("prompts", help="Generate business scenario prompts")
    add_common_args(sub_prompts)
    sub_prompts.set_defaults(func=cmd_prompts)

    # responses
    sub_responses = subparsers.add_parser(
        "responses", help="Generate initial responses (R0)"
    )
    add_common_args(sub_responses)
    sub_responses.set_defaults(func=cmd_responses)

    # cai
    sub_cai = subparsers.add_parser("cai", help="Run CAI critique + revision")
    add_common_args(sub_cai)
    sub_cai.set_defaults(func=cmd_cai)

    # pairs
    sub_pairs = subparsers.add_parser("pairs", help="Build preference pairs")
    add_common_args(sub_pairs)
    sub_pairs.set_defaults(func=cmd_pairs)

    # all
    sub_all = subparsers.add_parser("all", help="Run full pipeline")
    add_common_args(sub_all)
    sub_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
