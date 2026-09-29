"""Fast, no-network unit tests for nla/hrm/recon.py."""

import torch

from nla.hrm.recon import ReconWeights, failed_reward, loss_to_reward, parse_fields, recon_loss, shared_normalize


def test_shared_normalize_additivity():
    torch.manual_seed(0)
    z_L = torch.randn(8, 16) * 3
    z_H = torch.randn(8, 16) * 5
    n_L, n_H, n_s = shared_normalize(z_L, z_H)
    assert torch.allclose(n_L + n_H, n_s, atol=1e-5)
    # combined norm is exactly 1 by construction
    combined = (n_L.float().pow(2).sum(-1) + n_H.float().pow(2).sum(-1)).sqrt()
    assert torch.allclose(combined, torch.ones(8), atol=1e-5)


def test_recon_loss_perfect_reconstruction_is_zero():
    torch.manual_seed(1)
    z_L = torch.randn(4, 16)
    z_H = torch.randn(4, 16)
    loss = recon_loss(z_L, z_H, z_L, z_H, ReconWeights())
    assert loss.total.item() < 1e-6
    assert loss.mse_sum.item() < 1e-6
    assert loss.mse_L.item() < 1e-6
    assert loss.mse_H.item() < 1e-6


def test_recon_loss_hand_check():
    # 1-D, d=2, so the shared-normalize math is checkable by hand.
    z_L = torch.tensor([[1.0, 0.0]])
    z_H = torch.tensor([[0.0, 1.0]])
    # scale = sqrt(1+1) = sqrt(2); z_L_n=[1/sqrt2,0], z_H_n=[0,1/sqrt2], s_n=[1/sqrt2,1/sqrt2]
    z_L_hat = torch.tensor([[0.0, 0.0]])  # predict zero for L
    z_H_hat = z_H.clone()  # predict H exactly
    loss = recon_loss(z_L_hat, z_H_hat, z_L, z_H, ReconWeights(w_sum=1.0, w_comp=1.0))
    inv_sqrt2 = 1.0 / (2 ** 0.5)
    expected_mse_L = ((0 - inv_sqrt2) ** 2 + (0 - 0) ** 2) / 2  # mean over d=2
    expected_mse_H = 0.0
    s_hat_n = torch.tensor([0.0, inv_sqrt2])  # z_L_hat_n=[0,0] + z_H_hat_n=[0,1/sqrt2]
    s_n = torch.tensor([inv_sqrt2, inv_sqrt2])
    expected_mse_sum = ((s_hat_n - s_n) ** 2).mean().item()
    assert abs(loss.mse_L.item() - expected_mse_L) < 1e-5
    assert abs(loss.mse_H.item() - expected_mse_H) < 1e-5
    assert abs(loss.mse_sum.item() - expected_mse_sum) < 1e-5


def test_loss_to_reward_and_failed_reward_ordering():
    weights = ReconWeights(w_sum=1.0, w_comp=0.25)
    good_loss = torch.tensor(0.1)
    bad_loss = torch.tensor(1.0)  # mean-predictor-ish baseline
    assert loss_to_reward(good_loss) > loss_to_reward(bad_loss)
    assert failed_reward(weights) < loss_to_reward(bad_loss)
    # log-reward variant preserves the same ordering
    assert loss_to_reward(good_loss, log_reward=True) > loss_to_reward(bad_loss, log_reward=True)
    assert failed_reward(weights, log_reward=True) < loss_to_reward(bad_loss, log_reward=True)


def test_parse_fields_well_formed():
    text = "<explanation>\nL: feature a\n\nfeature a2\nH: feature b\n\nfeature b2\n</explanation>"
    parsed = parse_fields(text)
    assert parsed == ("feature a\n\nfeature a2", "feature b\n\nfeature b2")


def test_parse_fields_rejects_malformed():
    assert parse_fields("no tags at all") is None
    assert parse_fields("<explanation>\nL: only l, no H\n</explanation>") is None
    assert parse_fields("<explanation>\nL: \nH: b\n</explanation>") is None  # empty L field
    assert parse_fields("<explanation>\nL: a\nH: \n</explanation>") is None  # empty H field
    assert parse_fields("<explanation>\nno L/H labels here\n</explanation>") is None
