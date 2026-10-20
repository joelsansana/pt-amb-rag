#!/usr/bin/env python
"""CLI to build the Qdrant index from processed chunks.

Loads every data/processed/*.jsonl file, embeds the chunks in batches of 64
(the model declared in corpora.yaml, via pt_amb_rag.retrieval.embedder),
upserts them into Qdrant, and reports the total number of points now in the
collection.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py
        --data-dir data/processed --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pt_amb_rag.retrieval.vector_store import (
    COLLECTION_NAME,
    ensure_collection,
    get_client,
    upsert_chunks,
)

DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_BATCH_SIZE = 64


def iter_chunks(data_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield one chunk dict per line across every *.jsonl file in data_dir."""
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_num}: {e}"
                    ) from e


def batched(
    iterable: Iterator[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_index(data_dir: Path, batch_size: int) -> int:
    """Embed and upsert every chunk found under data_dir. Returns the number
    of points upserted in this run."""
    client = get_client()
    ensure_collection(client, COLLECTION_NAME)

    upserted = 0
    for i, batch in enumerate(
        batched(iter_chunks(data_dir), batch_size), start=1
    ):
        n = upsert_chunks(batch, client=client)
        upserted += n
        print(f"  batch {i}: upserted {n} points (running total: {upserted})")

    return upserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Qdrant index from processed chunks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory of .jsonl files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding/upsert batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading chunks from {args.data_dir} ...")
    try:
        upserted = build_index(args.data_dir, args.batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    collection_total = client.count(
        collection_name=COLLECTION_NAME, exact=True
    ).count

    print(f"\nDone. Points upserted this run: {upserted}")
    print(f"Total points in '{COLLECTION_NAME}': {collection_total}")


if __name__ == "__main__":
    main()
