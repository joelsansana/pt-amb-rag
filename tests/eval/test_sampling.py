from pt_amb_rag.eval.sampling import candidate_key, sample_unreviewed


def make(q: str, content_hash: str = "h1") -> dict:
    return {
        "question": q,
        "content_hash": content_hash,
        "difficulty": "easy",
        "answerable": True,
    }


def test_candidate_key_is_stable_for_same_input():
    c = make("What must deployers do?")
    assert candidate_key(c) == candidate_key(c)


def test_candidate_key_differs_for_different_questions():
    a = make("Question A?")
    b = make("Question B?")
    assert candidate_key(a) != candidate_key(b)


def test_excludes_already_reviewed():
    candidates = [make(f"q{i}?") for i in range(10)]
    state = {
        candidate_key(candidates[0]): "accepted:lepanto",
        candidate_key(candidates[1]): "rejected",
    }

    sample = sample_unreviewed(candidates, state, n=25, seed=0)

    sampled_keys = {candidate_key(c) for c in sample}
    assert candidate_key(candidates[0]) not in sampled_keys
    assert candidate_key(candidates[1]) not in sampled_keys
    assert len(sample) == 8


def test_returns_at_most_n():
    candidates = [make(f"q{i}?") for i in range(100)]
    sample = sample_unreviewed(candidates, {}, n=25, seed=0)
    assert len(sample) == 25


def test_returns_fewer_than_n_without_erroring_if_not_enough_remain():
    candidates = [make(f"q{i}?") for i in range(5)]
    sample = sample_unreviewed(candidates, {}, n=25, seed=0)
    assert len(sample) == 5


def test_empty_candidates_returns_empty():
    assert sample_unreviewed([], {}, n=25, seed=0) == []


def test_all_reviewed_returns_empty():
    candidates = [make(f"q{i}?") for i in range(5)]
    state = {candidate_key(c): "rejected" for c in candidates}
    assert sample_unreviewed(candidates, state, n=25, seed=0) == []


def test_same_seed_gives_same_sample():
    candidates = [make(f"q{i}?") for i in range(50)]
    a = sample_unreviewed(candidates, {}, n=10, seed=42)
    b = sample_unreviewed(candidates, {}, n=10, seed=42)
    assert [candidate_key(c) for c in a] == [candidate_key(c) for c in b]


def test_no_seed_still_returns_valid_sample():
    candidates = [make(f"q{i}?") for i in range(50)]
    sample = sample_unreviewed(candidates, {}, n=10)
    assert len(sample) == 10
    # every sampled candidate actually came from the input set
    input_keys = {candidate_key(c) for c in candidates}
    assert all(candidate_key(c) in input_keys for c in sample)
