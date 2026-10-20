#!/usr/bin/env python
"""Run the generation eval (LLM-as-judge) against a golden-set tier.

For each question: call ask() unfiltered (no regulation/part/article_num
filters — same realism rule as run_eval.py), then judge() the answer
against the retrieved sources.

- principal tier: judged for faithfulness + citation_accuracy
- unanswerable tier: judged for abstention (expect_abstention=True)

k defaults to 5 to match the production API's AskRequest default — the
generation eval measures the system a user actually gets, not the wider
k=10 window the retrieval eval needs for ranking metrics.

Usage:
    uv run python scripts/run_eval_gen.py --tier principal
    uv run python scripts/run_eval_gen.py --tier unanswerable
    uv run python scripts/run_eval_gen.py --tier unanswerable
    uv run python scripts/run_eval_gen.py --tier principal \
        --output evals/results/gen_principal.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from pt_amb_rag.api.ask import ask
from pt_amb_rag.config import load_config
from pt_amb_rag.eval.judge import JudgeError, judge

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path("evals/golden")
RESULTS_DIR = Path("evals/results")

# Weighted faithfulness: "partial" credit is real — an answer with one
# unsupported span among many supported claims is not equal to fabrication.
FAITHFULNESS_SCORE = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def run_generation_eval(golden_path: Path, k: int = 5) -> dict:
    per_question: list[dict] = []
    failures: list[str] = []
    unjudged: list[str] = []

    lines = [
        line for line in golden_path.read_text().splitlines() if line.strip()
    ]

    for i, line in enumerate(lines, 1):
        q = json.loads(line)
        q_id = q.get("q_id")
        expect_abstention = q.get("tier") == "unanswerable"

        try:
            result = ask(q["question"], k=k)
        except Exception:
            logger.exception("ask() failed for question: %s", q_id)
            failures.append(q_id)
            continue

        try:
            verdict = judge(
                q["question"],
                result["sources"],
                result["answer"],
                expect_abstention=expect_abstention,
            )
        except JudgeError:
            logger.warning("question unjudged (judge failed twice): %s", q_id)
            unjudged.append(q_id)
            continue

        per_question.append(
            {
                "q_id": q_id,
                "question": q["question"],
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "expect_abstention": expect_abstention,
                "answer": result["answer"],
                "verdict": verdict,
            }
        )

        v = (
            verdict["abstention"]
            if expect_abstention
            else verdict["faithfulness"]
        )
        extra = (
            "" if expect_abstention else f" cite={verdict['citation_accuracy']}"
        )
        label = "abstain" if expect_abstention else "faithful"
        print(f"  [{i}/{len(lines)}] {q_id}: {label}={v}{extra}")

    answerable = [r for r in per_question if not r["expect_abstention"]]
    abstention = [r for r in per_question if r["expect_abstention"]]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    faithfulness_scores = [
        FAITHFULNESS_SCORE[r["verdict"]["faithfulness"]] for r in answerable
    ]
    faithfulness_counts = {
        key: sum(1 for r in answerable if r["verdict"]["faithfulness"] == key)
        for key in ("yes", "partial", "no")
    }
    citation_scores = [
        1.0 if r["verdict"]["citation_accuracy"] == "yes" else 0.0
        for r in answerable
    ]
    abstention_scores = [
        1.0 if r["verdict"]["abstention"] == "yes" else 0.0 for r in abstention
    ]

    return {
        "faithfulness_rate": _mean(faithfulness_scores),
        "faithfulness_counts": faithfulness_counts,
        "citation_accuracy_rate": _mean(citation_scores),
        "abstention_rate": _mean(abstention_scores),
        "n_answerable_judged": len(answerable),
        "n_abstention_judged": len(abstention),
        "n_questions": len(per_question),
        "n_failed": len(failures),
        "failed_q_ids": failures,
        "n_unjudged": len(unjudged),
        "unjudged_q_ids": unjudged,
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        required=True,
        choices=list(load_config().tiers),
        help="golden-set tier to evaluate (from corpora.yaml)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="sources retrieved per question (default 5, matching the API "
        "default)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "MINIMAX_API_KEY" not in os.environ:
        sys.exit(
            "MINIMAX_API_KEY is not set. The generation eval makes live LLM "
            "calls (ask + judge per question). Export it and retry."
        )

    golden_path = GOLDEN_DIR / f"{args.tier}.jsonl"
    if not golden_path.exists():
        raise FileNotFoundError(
            f"no golden file for tier {args.tier!r}: {golden_path}"
        )

    print(f"Running generation eval: tier={args.tier}, k={args.k}")
    t0 = time.monotonic()
    result = run_generation_eval(golden_path, k=args.k)
    elapsed = time.monotonic() - t0

    print(f"\nResults ({result['n_questions']} judged, {elapsed:.1f}s):")
    if result["n_answerable_judged"]:
        c = result["faithfulness_counts"]
        print(
            f"  Faithfulness:      {result['faithfulness_rate']:.3f} "
            f"(yes={c['yes']} partial={c['partial']} no={c['no']})"
        )
        print(f"  Citation accuracy: {result['citation_accuracy_rate']:.3f}")
    if result["n_abstention_judged"]:
        print(f"  Abstention rate:   {result['abstention_rate']:.3f}")
    if result["n_failed"]:
        print(
            f"  WARNING: {result['n_failed']} questions failed: "
            f"{result['failed_q_ids']}"
        )
    if result["n_unjudged"]:
        print(
            f"  WARNING: {result['n_unjudged']} questions unjudged: "
            f"{result['unjudged_q_ids']}"
        )

    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = RESULTS_DIR / f"gen-{args.tier}-{timestamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "per_question"}
    summary["tier"] = args.tier
    summary["k"] = args.k
    output_path.write_text(
        json.dumps(
            {**summary, "per_question": result["per_question"]}, indent=2
        )
    )
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
