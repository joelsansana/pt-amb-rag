from __future__ import annotations

import hashlib
import re
from pathlib import Path

EXACT_SUFFIX = " (exact case)"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_term_hits(path: Path, term: str) -> int:
    """Count JSONL lines (chunks) containing term — the project's
    documented verification method: grep -ci, case-sensitive if the term
    is suffixed ' (exact case)'."""
    exact = term.endswith(EXACT_SUFFIX)
    bare_term = term[: -len(EXACT_SUFFIX)] if exact else term
    flags = 0 if exact else re.IGNORECASE
    pattern = re.compile(re.escape(bare_term), flags)

    count = 0
    for line in path.read_text().splitlines():
        if line.strip() and pattern.search(line):
            count += 1
    return count


def check_record_pins(record: dict, corpus_paths: dict[str, Path]) -> list[str]:
    """Returns human-readable problems; empty list means the record's
    pins are consistent with the corpus as it stands right now.

    `corpus_paths` maps a corpus id to the processed JSONL file for that
    diploma. The upstream implementation hardcoded two files (the AI Act
    and NIS2); this fork generalises it to the N diplomas declared in
    `corpora.yaml`, so adding an instrument does not change this
    module's signature.
    """
    problems: list[str] = []
    q_id = record.get("q_id", "?")

    current = {
        f"{corpus_id}_sha256": file_sha256(path)
        for corpus_id, path in corpus_paths.items()
    }
    snap = record.get("corpus_snapshot", {})
    for key, value in current.items():
        if snap.get(key) != value:
            problems.append(f"{q_id}: corpus_snapshot[{key}] is stale")

    corpus_ids = list(corpus_paths)
    for term, recorded in record.get("anchor_terms", {}).items():
        actual = [
            count_term_hits(corpus_paths[corpus_id], term)
            for corpus_id in corpus_ids
        ]
        if list(recorded) != actual:
            problems.append(
                f"{q_id}: anchor_terms[{term!r}] recorded "
                f"{recorded}, actual {actual}"
            )

    for term in record.get("must_be_zero", []):
        actual = [
            count_term_hits(corpus_paths[corpus_id], term)
            for corpus_id in corpus_ids
        ]
        if any(actual):
            problems.append(
                f"{q_id}: must_be_zero term {term!r} now has hits "
                f"{actual} — question may have become answerable"
            )

    return problems
