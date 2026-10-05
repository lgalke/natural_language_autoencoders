import numpy as np

from nla.hrm.lh_variance import ridge_r2, shared_normalise


def test_ridge_r2_perfect_and_independent():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y_lin = x @ rng.normal(size=(8, 8))
    assert ridge_r2(x[:400], y_lin[:400], x[400:], y_lin[400:], lam=1e-6) > 0.99
    y_ind = rng.normal(size=(600, 8))
    assert abs(ridge_r2(x[:400], y_ind[:400], x[400:], y_ind[400:])) < 0.1


def test_shared_normalise_unit_joint_norm():
    zl, zh = np.random.default_rng(1).normal(size=(5, 4)), np.random.default_rng(2).normal(size=(5, 4))
    a, b = shared_normalise(zl, zh)
    assert np.allclose((a**2).sum(-1) + (b**2).sum(-1), 1.0)
