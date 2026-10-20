def hit_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> int:
    """1 if gold_id appears anywhere in the top k retrieved ids, else 0."""
    return int(gold_id in retrieved_ids[:k])


def mrr_at_k(retrieved_ids: list[str], gold_id: str, k: int) -> float:
    """Reciprocal rank of gold_id within the top k, or 0.0 if absent."""
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of all acceptable gold ids present in the top k retrieved ids.

    An empty gold_ids set is treated as trivially satisfied (1.0) rather than
    a division by zero — a question with no acceptable answers isn't a
    retrieval failure to score against.
    """
    if not gold_ids:
        return 1.0
    return len(gold_ids & set(retrieved_ids[:k])) / len(gold_ids)
