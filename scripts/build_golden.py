#!/usr/bin/env python
"""Generate candidate golden-set questions for one corpus's chunk file.

Usage:
    uv run python scripts/build_golden.py \
        --corpus dl_150_2015 --output evals/golden/dl_150_2015.candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pt_amb_rag.eval.golden_generator import (
    GoldenGenerationError,
    generate_candidates,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")


def load_chunks(corpus_id: str) -> list[dict]:
    path = DATA_DIR / f"{corpus_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no processed data for corpus {corpus_id!r}: {path}"
        )

    chunks = []
    for line in path.read_text().splitlines():
        if line.strip():
            chunks.append(json.loads(line))
    return chunks


def build_golden(
    corpus_id: str, output: Path, temperature: float = 0.2
) -> dict:
    chunks = load_chunks(corpus_id)

    written = 0
    skipped_no_hash = 0
    failed: list[str | None] = []

    with output.open("w") as out:
        for i, chunk in enumerate(chunks, start=1):
            # content_hash is how the golden set ties a question back to a
            # specific corpus chunk (see the eval-drift pitfall) — a chunk
            # without one can't be safely used, so skip rather than guess.
            if not chunk.get("content_hash"):
                skipped_no_hash += 1
                logger.warning(
                    "chunk %d/%d has no content_hash, skipping (header=%r)",
                    i,
                    len(chunks),
                    chunk.get("header"),
                )
                continue

            try:
                candidates = generate_candidates(chunk, temperature=temperature)
            except GoldenGenerationError:
                logger.exception(
                    "failed to generate candidates for chunk %d/%d (hash=%s)",
                    i,
                    len(chunks),
                    chunk.get("content_hash"),
                )
                failed.append(chunk.get("content_hash"))
                continue

            for c in candidates:
                out.write(json.dumps(c) + "\n")
                written += 1

            if i % 25 == 0 or i == len(chunks):
                print(
                    f"  {i}/{len(chunks)} chunks processed, "
                    f"{written} candidates written so far"
                )

    return {
        "corpus": corpus_id,
        "chunks_total": len(chunks),
        "chunks_skipped_no_hash": skipped_no_hash,
        "chunks_failed": len(failed),
        "candidates_written": written,
        "failed_hashes": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        required=True,
        help="corpus id from corpora.yaml, e.g. dl_150_2015",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = build_golden(
        args.corpus, args.output, temperature=args.temperature
    )

    written = summary["candidates_written"]
    print(f"\nDone. {written} candidates written to {args.output}")
    print(f"  chunks total: {summary['chunks_total']}")
    if summary["chunks_skipped_no_hash"]:
        skipped = summary["chunks_skipped_no_hash"]
        print(f"  chunks skipped (no content_hash): {skipped}")
    if summary["chunks_failed"]:
        print(f"  chunks failed generation: {summary['chunks_failed']}")
        print(f"  failed hashes: {summary['failed_hashes']}")
        print(
            "  Review these before vetting — a failed chunk contributes "
            "zero candidates."
        )


if __name__ == "__main__":
    main()
