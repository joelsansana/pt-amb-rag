#!/usr/bin/env python
"""Fetch the corpus declared in `corpora.yaml` from the Diário da República.

The DRE website is a single-page app, so this script talks to the same
screenservice API the site itself uses (see `pt_amb_rag.ingestion.dre`)
rather than scraping HTML.

For every diploma in the manifest it writes:

* `data/raw/<corpus_id>.json` — the screenservice payload, verbatim, so
  a re-fetch can be diffed against it;
* `data/processed/<corpus_id>.jsonl` — the parsed structural units;
* `data/raw/<corpus_id>.sha256` — the hash of the raw payload.

The SHA-256 is what `corpus_pins` pins the golden set against: if the
DRE changes a text, the hash changes and the golden set fails loudly
instead of silently scoring against a different corpus.

Usage:
    uv run python scripts/fetch_corpus.py
    uv run python scripts/fetch_corpus.py --only dl_150_2015
    uv run python scripts/fetch_corpus.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from pt_amb_rag.config import Config, Diploma, load_config
from pt_amb_rag.ingestion.dre import DreClient, DreError, units_from_document

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_one(client: DreClient, diploma: Diploma) -> tuple[bytes, list[dict]]:
    """Fetch one diploma and return its raw payload and parsed units."""
    document = client.fetch_document(diploma.tipo, diploma.key)

    raw = json.dumps(
        {
            "id": diploma.id,
            "tipo": diploma.tipo,
            "key": diploma.key,
            "title": document.title,
            "number": document.number,
            "published": document.published,
            "emitter": document.emitter,
            "series": document.series,
            "pages": document.pages,
            "eli": document.eli,
            "pdf_url": document.pdf_url,
            "text": document.text,
            "formatted_text": document.formatted_text,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    units = units_from_document(
        document,
        corpus_id=diploma.id,
        effective_date=diploma.effective_date or diploma.published,
    )
    return raw, units


def write_corpus(
    diploma: Diploma, raw: bytes, units: list[dict]
) -> tuple[Path, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{diploma.id}.json"
    raw_path.write_bytes(raw)
    (RAW_DIR / f"{diploma.id}.sha256").write_text(
        sha256_bytes(raw) + "\n", encoding="utf-8"
    )

    processed_path = PROCESSED_DIR / f"{diploma.id}.jsonl"
    processed_path.write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n",
        encoding="utf-8",
    )
    return raw_path, processed_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="path to the corpus manifest (default: corpora.yaml)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="fetch only this corpus id (repeatable)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the diplomas in the manifest and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config: Config = load_config(args.manifest)

    if args.list:
        for diploma in config.diplomas:
            print(
                f"{diploma.id:18s} {diploma.title:28s} "
                f"{diploma.published}  {', '.join(diploma.tags)}"
            )
        return

    wanted = set(args.only) if args.only else None
    diplomas = [d for d in config.diplomas if wanted is None or d.id in wanted]
    if wanted:
        unknown = wanted - {d.id for d in config.diplomas}
        if unknown:
            parser.error(f"unknown corpus id(s): {', '.join(sorted(unknown))}")

    failures: list[str] = []

    with DreClient() as client:
        for diploma in diplomas:
            try:
                raw, units = fetch_one(client, diploma)
            except DreError as exc:
                logger.error("%s: %s", diploma.id, exc)
                failures.append(diploma.id)
                continue

            raw_path, processed_path = write_corpus(diploma, raw, units)
            print(
                f"{diploma.id:18s} {len(units):4d} units  "
                f"sha256={sha256_bytes(raw)[:12]}  "
                f"-> {raw_path}  {processed_path}"
            )

    if failures:
        raise SystemExit(
            f"failed to fetch {len(failures)} diploma(s): {', '.join(failures)}"
        )


if __name__ == "__main__":
    main()
