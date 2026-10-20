import pytest

from pt_amb_rag.eval.metrics import hit_at_k, mrr_at_k, recall_at_k

# --- hit_at_k ---


def test_hit_at_k_gold_present_at_top():
    assert hit_at_k(["a", "b", "c"], "a", k=5) == 1


def test_hit_at_k_gold_present_at_last_valid_position():
    assert hit_at_k(["a", "b", "c"], "c", k=3) == 1


def test_hit_at_k_gold_present_but_outside_k():
    assert hit_at_k(["a", "b", "c", "d"], "d", k=3) == 0


def test_hit_at_k_gold_absent():
    assert hit_at_k(["a", "b", "c"], "z", k=5) == 0


def test_hit_at_k_empty_retrieved():
    assert hit_at_k([], "a", k=5) == 0


def test_hit_at_k_k_larger_than_list():
    assert hit_at_k(["a"], "a", k=100) == 1


def test_hit_at_k_k_zero_never_hits():
    assert hit_at_k(["a", "b"], "a", k=0) == 0


# --- mrr_at_k ---


def test_mrr_at_k_gold_at_first_position():
    assert mrr_at_k(["a", "b", "c"], "a", k=5) == 1.0


def test_mrr_at_k_gold_at_third_position():
    assert mrr_at_k(["a", "b", "c"], "c", k=5) == pytest.approx(1 / 3)


def test_mrr_at_k_gold_at_last_valid_position():
    assert mrr_at_k(["a", "b", "c"], "c", k=3) == pytest.approx(1 / 3)


def test_mrr_at_k_gold_present_but_outside_k():
    assert mrr_at_k(["a", "b", "c", "d"], "d", k=3) == 0.0


def test_mrr_at_k_gold_absent():
    assert mrr_at_k(["a", "b", "c"], "z", k=5) == 0.0


def test_mrr_at_k_empty_retrieved():
    assert mrr_at_k([], "a", k=5) == 0.0


def test_mrr_at_k_k_larger_than_list():
    assert mrr_at_k(["a", "b"], "b", k=100) == pytest.approx(1 / 2)


def test_mrr_at_k_duplicate_ids_uses_first_occurrence():
    assert mrr_at_k(["x", "a", "a"], "a", k=5) == pytest.approx(1 / 2)


# --- recall_at_k ---


def test_recall_at_k_all_gold_present():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0


def test_recall_at_k_partial_overlap():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=5) == pytest.approx(0.5)


def test_recall_at_k_no_overlap():
    assert recall_at_k(["x", "y"], {"a", "b"}, k=5) == 0.0


def test_recall_at_k_empty_gold_ids_is_trivially_satisfied():
    assert recall_at_k(["a", "b"], set(), k=5) == 1.0


def test_recall_at_k_respects_k_cutoff():
    # gold "b" is retrieved, but outside the top k
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == pytest.approx(0.5)


def test_recall_at_k_empty_retrieved():
    assert recall_at_k([], {"a", "b"}, k=5) == 0.0


def test_recall_at_k_k_larger_than_list():
    assert recall_at_k(["a"], {"a", "b"}, k=100) == pytest.approx(0.5)
