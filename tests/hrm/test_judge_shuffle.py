from nla.hrm.eval import shuffle_permutation


def test_roll_is_a_derangement():
    rows = [{"dataset": "a"}] * 5
    perm = shuffle_permutation(rows, within_dataset=False)
    assert sorted(perm) == list(range(5))
    assert all(i != p for i, p in enumerate(perm))


def test_within_dataset_stays_in_group_and_keeps_singletons():
    rows = [{"dataset": d} for d in ["a", "b", "a", "c", "a", "b"]]
    perm = shuffle_permutation(rows, within_dataset=True)
    assert sorted(perm) == list(range(6))
    assert all(rows[i]["dataset"] == rows[p]["dataset"] for i, p in enumerate(perm))
    assert perm[3] == 3                                   # single-row dataset maps to itself
    assert all(perm[i] != i for i in (0, 1, 2, 4, 5))     # every other row gets a different row
