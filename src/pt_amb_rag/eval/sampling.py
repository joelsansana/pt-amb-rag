from __future__ import annotations

import random


def candidate_key(candidate: dict) -> str:
    """Stable identity for a candidate across runs, independent of
    file order."""
    return f"{candidate.get('content_hash')}::{candidate.get('question')}"


def sample_unreviewed(
    candidates: list[dict],
    state: dict[str, str],
    n: int,
    seed: int | None = None,
) -> list[dict]:
    """Randomly sample up to n candidates with no decision yet in state.

    Sampling is without replacement. If fewer than n candidates remain
    unreviewed, returns all of them — this is not an error, just a smaller
    batch. Pass seed for a reproducible sample (useful if you want to
    re-run the exact same batch, e.g. after an interrupted session).
    """
    unreviewed = [c for c in candidates if state.get(candidate_key(c)) is None]
    rng = random.Random(seed)
    rng.shuffle(unreviewed)
    return unreviewed[:n]
