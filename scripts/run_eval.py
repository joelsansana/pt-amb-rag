#!/usr/bin/env python
"""Run the retrieval eval against a golden-set tier.

Retrieves once per question at k_rank (default 10), then derives
Hit@k_hit (default 5) and MRR@k_rank/Recall@k_rank from that same list —
per the phase-4 design decision, this avoids the original tutorial's bug
of computing "MRR@10" from a 5-item retrieval.

Questions are asked unfiltered (no regulation/part/article_num) — golden
questions model a realistic user, who wouldn't pass internal filters.

Usage:
    uv run python scripts/run_eval.py --tier principal
    uv run python scripts/run_eval.py --tier demo \
        --output evals/results/demo_run.json
    uv run python scripts/run_eval.py --tier principal \
        --output evals/results/baseline.json --summary-only
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from pt_amb_rag.api.ask import ask
from pt_amb_rag.config import load_config
from pt_amb_rag.eval.metrics import hit_at_k, mrr_at_k, recall_at_k

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path("evals/golden")
RESULTS_DIR = Path("evals/results")


def run_retrieval_eval(
    golden_path: Path, k_hit: int = 5, k_rank: int = 10, no_llm: bool = False
) -> dict:
    hits, mrrs, recalls, failures = [], [], [], []
    per_question = []

    lines = [
        line for line in golden_path.read_text().splitlines() if line.strip()
    ]

    for line in lines:
        q = json.loads(line)
        try:
            result = ask(q["question"], k=k_rank, answer=not no_llm)
        except Exception:
            logger.exception("eval question failed: %s", q.get("q_id"))
            failures.append(q.get("q_id"))
            continue

        ids = [s["content_hash"] for s in result["sources"]]
        gold_id = q["gold_ids"][0]
        hit = hit_at_k(ids, gold_id, k_hit)
        mrr = mrr_at_k(ids, gold_id, k_rank)
        recall = recall_at_k(ids, set(q["all_acceptable_ids"]), k_rank)

        hits.append(hit)
        mrrs.append(mrr)
        recalls.append(recall)
        per_question.append(
            {
                "q_id": q.get("q_id"),
                "question": q["question"],
                "difficulty": q.get("difficulty"),
                "hit": hit,
                "mrr": mrr,
                "recall": recall,
                "retrieved_ids": ids,
            }
        )

    if failures:
        logger.warning(
            "%d/%d questions failed and were excluded: %s",
            len(failures),
            len(failures) + len(hits),
            failures,
        )

    return {
        "hit_at_5": statistics.mean(hits) if hits else 0.0,
        "mrr_at_10": statistics.mean(mrrs) if mrrs else 0.0,
        "recall_at_10": statistics.mean(recalls) if recalls else 0.0,
        "n_questions": len(hits),
        "n_failed": len(failures),
        "failed_q_ids": failures,
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
        "--k-hit",
        type=int,
        default=5,
        help="k for Hit@k (default 5)",
    )
    parser.add_argument(
        "--k-rank",
        type=int,
        default=10,
        help="k for MRR@k / Recall@k (default 10)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="retrieval-only mode: skip answer generation (ask(answer=False)). "
        "No MINIMAX_API_KEY needed — the embedder is local. Used by the "
        "keyless CI gate (.github/workflows/eval.yml).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit per-question detail from the output file — use this for "
        "evals/results/baseline.json, since that file is committed and "
        "per-question detail would churn the diff every regeneration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    golden_path = GOLDEN_DIR / f"{args.tier}.jsonl"
    if not golden_path.exists():
        raise FileNotFoundError(
            f"no golden file for tier {args.tier!r}: {golden_path}"
        )

    print(
        f"Running retrieval eval: tier={args.tier}, "
        f"k_hit={args.k_hit}, k_rank={args.k_rank}"
    )
    if args.no_llm:
        print("Mode: retrieval-only (no LLM calls)")
    t0 = time.monotonic()
    result = run_retrieval_eval(
        golden_path, k_hit=args.k_hit, k_rank=args.k_rank, no_llm=args.no_llm
    )
    elapsed = time.monotonic() - t0

    print(f"\nResults ({result['n_questions']} questions, {elapsed:.1f}s):")
    print(f"  Hit@{args.k_hit}:    {result['hit_at_5']:.3f}")
    print(f"  MRR@{args.k_rank}:    {result['mrr_at_10']:.3f}")
    print(f"  Recall@{args.k_rank}: {result['recall_at_10']:.3f}")
    if result["n_failed"]:
        print(
            f"  WARNING: {result['n_failed']} questions failed: "
            f"{result['failed_q_ids']}"
        )

    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = RESULTS_DIR / f"{args.tier}-{timestamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "per_question"}
    summary["tier"] = args.tier
    summary["k_hit"] = args.k_hit
    summary["k_rank"] = args.k_rank

    output_data = (
        summary
        if args.summary_only
        else {**summary, "per_question": result["per_question"]}
    )
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
