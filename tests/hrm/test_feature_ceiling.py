import numpy as np

from nla.hrm.feature_ceiling import compact, run_ladder


def _data(n=6000, d=12, seed=0):
    rng = np.random.default_rng(seed)
    tok = rng.integers(0, 20, n)
    pos = rng.integers(0, 20, n)
    table_l, table_h = rng.normal(size=(20, d)), rng.normal(size=(20, d))
    zl = table_l[tok] + 0.3 * rng.normal(size=(n, d))   # z_L depends on the token only
    zh = table_h[tok] + 0.3 * rng.normal(size=(n, d))
    z = np.concatenate([zl, zh], axis=1).astype(np.float32)
    cols = {"tok": tok.astype(np.int64) + 1, "pos20": pos.astype(np.int64), "abspos": np.zeros(n, np.int64),
            "lenb": np.zeros(n, np.int64), "last": np.zeros(n, np.int64), "ds": np.zeros(n, np.int64)}
    return cols, z, d


def test_ladder_finds_planted_token_effect_and_not_position():
    cols, z, d = _data()
    train, val, ev = np.arange(0, 4000), np.arange(4000, 4500), np.arange(4500, 6000)
    ztr = z[train]
    base = {"sum": float(((ztr[:, :d] + ztr[:, d:] - (ztr[:, :d] + ztr[:, d:]).mean(0)) ** 2).mean()),
            "L": float(((ztr[:, :d] - ztr[:, :d].mean(0)) ** 2).mean()),
            "H": float(((ztr[:, d:] - ztr[:, d:].mean(0)) ** 2).mean())}
    res = run_ladder(cols, None, z, d, train, val, {"iid": ev}, base, ["token", "position"], pcs=24, device="cpu")
    tok_sum, tok_l, tok_h = res["token"]["iid"]["all"]
    pos_sum, _, _ = res["position"]["iid"]["all"]
    assert tok_sum > 0.8 and tok_l > 0.8 and tok_h > 0.8
    assert abs(pos_sum) < 0.1
    assert res["pca ceiling"]["iid"]["all"][0] > 0.99


def test_compact_maps_rare_values_to_zero():
    col = np.array([5, 5, 5, 7, 7, 7, 9])
    out = compact(col, np.arange(7), min_count=3)
    assert out[6] == 0 and out[0] == out[1] and out[0] != out[3] and out[0] > 0


def test_ridge_cg_matches_the_dense_solution_and_group_means():
    import torch

    from nla.hrm.feature_ceiling import _design, ridge_cg

    rng = np.random.default_rng(1)
    n, k = 500, 7
    cols = {"tok": rng.integers(0, k, n).astype(np.int64)}
    y = (rng.normal(size=(n, 3)) + cols["tok"][:, None]).astype(np.float32)
    idx = np.arange(n)
    a, at = _design(cols, None, ["tok"], idx, {"tok": 0}, k, "cpu")
    w = ridge_cg(a, at, torch.tensor(y), lam=1e-6, iters=100).numpy()
    dense = a.to_dense().numpy()
    ref = np.linalg.solve(dense.T @ dense + 1e-6 * np.eye(k), dense.T @ y)
    assert np.allclose(w, ref, atol=1e-3)
    for v in range(k):  # a single categorical feature with ~no ridge reproduces the group means
        assert np.allclose(w[v], y[cols["tok"] == v].mean(0), atol=1e-2)
