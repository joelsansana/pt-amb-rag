from __future__ import annotations


def is_junk_candidate(candidate: dict, seen_questions: set[str]) -> bool:
    """Cheap heuristics to drop candidates not worth a human's time.

    Conservative on purpose: false negatives (junk that slips through)
    just cost you a few seconds in review; false positives (a good
    question wrongly dropped) are a silent quality loss. Keep the bar low.
    """
    q = (candidate.get("question") or "").strip()

    if len(q) < 15:
        return True
    if not q.endswith("?"):
        return True
    return q.lower() in seen_questions


def prefilter(candidates: list[dict]) -> tuple[list[dict], int]:
    """Returns (kept_candidates, n_dropped)."""
    seen: set[str] = set()
    kept = []
    dropped = 0

    for c in candidates:
        q = (c.get("question") or "").strip().lower()
        if is_junk_candidate(c, seen):
            dropped += 1
            continue
        seen.add(q)
        kept.append(c)

    return kept, dropped
