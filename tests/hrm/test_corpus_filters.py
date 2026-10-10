from nla.hrm.build_prompt_corpus import filter_local_rows, prompt_hash


def _rows():
    return [{"prompt": f"p {i}", "dataset": "a" if i < 6 else "b", "world": f"w{i}", "doc_id": f"d{i}"} for i in range(10)]


def test_duplicate_drop_dataset_drop_and_cap():
    rows = _rows()
    kept, stats = filter_local_rows(rows, {prompt_hash("p  0")}, {"b"}, 3, seed=1)   # whitespace-normalised match
    assert stats == {"dropped_dataset": 4, "dropped_duplicate": 1, "dropped_cap": 2}
    assert len(kept) == 3 and all(r["dataset"] == "a" and r["prompt"] != "p 0" for r in kept)


def test_cap_is_seeded_and_per_dataset():
    rows = _rows()
    a, _ = filter_local_rows(list(rows), set(), set(), 2, seed=7)
    b, _ = filter_local_rows(list(rows), set(), set(), 2, seed=7)
    assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]
    assert sorted({r["dataset"] for r in a}) == ["a", "b"] and len(a) == 4
