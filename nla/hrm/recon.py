"""Normalization + reconstruction loss/reward for the (z_L, z_H) NLA.

Single source of truth for `shared_normalize` — `norm_stats.py` (mean-predictor
baselines), the AR training scripts, and `judge.py` (comparing ẑ vs z in the
same space the loss was computed in) all import it from here. Do not
re-derive it elsewhere: a second implementation that drifts even slightly
breaks the FVE baselines silently (`nla_meta.yaml`'s `norm_stats_hash` is the
guard against loading a checkpoint against a mismatched baseline).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

from nla.schema import extract_explanation

# Matches "L: <anything, incl. \n\n-separated features> \n H: <anything to end>"
# inside the already-extracted <explanation>...</explanation> payload.
_LH_RE = re.compile(r"L:\s*(.*?)\s*\nH:\s*(.*)$", re.DOTALL)


_OTHER_HEADER_RE = re.compile(r"\n\s*[LH]:\s")


def parse_single(completion_text: str, stream: str) -> str | None:
    """Split-AV output: `<explanation>\nL: ...\n</explanation>` (or H). Returns the field text, None if the tags or the
    header are missing or the field is empty; a second field header the model may add is cut off."""
    assert stream in ("L", "H")
    payload = extract_explanation(completion_text)
    if payload is None:
        return None
    m = re.match(rf"{stream}:\s*(.*)$", payload, re.DOTALL)
    if m is None:
        return None
    text = _OTHER_HEADER_RE.split(m.group(1))[0].strip()
    return text or None


def parse_fields(completion_text: str) -> tuple[str, str] | None:
    """Extract (L_field, H_field) from a raw AV completion. None if the
    `<explanation>` tags are missing/unclosed, the `L:`/`H:` fields don't
    both parse, or either field is empty after stripping — any of these get
    `recon.failed_reward` (see `nla/hrm/build.py`'s response format, which
    this must stay in lockstep with: `wrap_lh_explanation`)."""
    payload = extract_explanation(completion_text)
    if payload is None:
        return None
    m = _LH_RE.search(payload)
    if m is None:
        return None
    l_field, h_field = m.group(1).strip(), m.group(2).strip()
    if not l_field or not h_field:
        return None
    return l_field, h_field

# Reward for a completion whose fields don't parse (or one is empty) —
# strictly worse than the mean predictor (whose per-term MSE ≈ 1 under
# shared-normalization, since ||shared_normalize(z)|| ≈ 1 by construction and
# a zero-ish mean leaves MSE ≈ mean(||z_norm||²) ≈ 1). Mirrors the original
# NLA's FAILED_EXTRACTION_REWARD "orthogonal-equivalent" convention (see
# docs/design.md §3.4) — a malformed completion must never look better than
# "predict nothing".
FAILED_MSE = 2.0


def shared_normalize(z_L: torch.Tensor, z_H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Divide z_L, z_H, and s=z_L+z_H by the SAME per-token scalar
    norm = sqrt(||z_L||^2 + ||z_H||^2), so additivity is exact:
    normalize(z_L) + normalize(z_H) == normalize(z_L + z_H).

    This is why we don't normalize each stream by its own norm (that would
    make ẑ_L + ẑ_H incomparable to s — the "what H2 actually reads" anchor
    the whole loss design depends on). Computed in fp32 regardless of input
    dtype; returned dtype matches input z_L's dtype.

    z_L, z_H: [..., d_model], same shape.
    """
    assert z_L.shape == z_H.shape, f"z_L {z_L.shape} != z_H {z_H.shape}"
    zL32, zH32 = z_L.float(), z_H.float()
    scale = (zL32.pow(2).sum(-1, keepdim=True) + zH32.pow(2).sum(-1, keepdim=True)).clamp_min(1e-12).sqrt()
    s32 = zL32 + zH32
    out_dtype = z_L.dtype
    return (zL32 / scale).to(out_dtype), (zH32 / scale).to(out_dtype), (s32 / scale).to(out_dtype)


@dataclass
class ReconWeights:
    w_sum: float = 1.0
    w_comp: float = 0.25


@dataclass
class ReconLoss:
    total: torch.Tensor
    mse_sum: torch.Tensor
    mse_L: torch.Tensor
    mse_H: torch.Tensor

    def fve(self, mean_mse_sum: float, mean_mse_L: float, mean_mse_H: float) -> dict[str, float]:
        """Fraction of variance explained per term, relative to the
        train-set mean-predictor baseline (from `norm_stats.py`)."""
        return {
            "fve_sum": 1.0 - self.mse_sum.item() / mean_mse_sum,
            "fve_L": 1.0 - self.mse_L.item() / mean_mse_L,
            "fve_H": 1.0 - self.mse_H.item() / mean_mse_H,
        }


def recon_loss(
    z_L_hat: torch.Tensor, z_H_hat: torch.Tensor, z_L: torch.Tensor, z_H: torch.Tensor, weights: ReconWeights
) -> ReconLoss:
    """Plain MSE (no whitening) on shared-normalized vectors:
        loss = w_sum * MSE(ẑ_L+ẑ_H, s) + w_comp * 0.5*(MSE(ẑ_L,z_L) + MSE(ẑ_H,z_H))
    All four vectors are shared-normalized with the SAME per-token scale
    (derived from the GOLD z_L, z_H — the prediction is scored in the gold's
    normalization frame, not its own, so the sum term stays a meaningful
    "did you predict what H reads" signal even for an off-scale prediction).
    """
    # predictions may live on another device than the gold vectors (e.g. GPU rollouts vs CPU rows)
    z_L_hat, z_H_hat = z_L_hat.to(z_L.device), z_H_hat.to(z_L.device)
    z_L_n, z_H_n, s_n = shared_normalize(z_L, z_H)
    scale = (z_L.float().pow(2).sum(-1, keepdim=True) + z_H.float().pow(2).sum(-1, keepdim=True)).clamp_min(1e-12).sqrt()
    z_L_hat_n = (z_L_hat.float() / scale).to(z_L_hat.dtype)
    z_H_hat_n = (z_H_hat.float() / scale).to(z_H_hat.dtype)
    s_hat_n = z_L_hat_n + z_H_hat_n

    mse_sum = (s_hat_n.float() - s_n.float()).pow(2).mean()
    mse_L = (z_L_hat_n.float() - z_L_n.float()).pow(2).mean()
    mse_H = (z_H_hat_n.float() - z_H_n.float()).pow(2).mean()
    total = weights.w_sum * mse_sum + weights.w_comp * 0.5 * (mse_L + mse_H)
    return ReconLoss(total=total, mse_sum=mse_sum, mse_L=mse_L, mse_H=mse_H)


def loss_to_reward(loss: torch.Tensor, *, log_reward: bool = False) -> float:
    """reward = -loss, or -log(loss) under NLA_LOG_MSE_REWARD-style config
    (matches nla/reward.py's convention). log form compresses the huge
    dynamic range near loss->0 (good rollouts) at the cost of an unbounded
    penalty as loss->0+ from numerical error — clamp before logging."""
    val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
    if log_reward:
        return -math.log(max(val, 1e-6))
    return -val


def failed_reward(weights: ReconWeights, *, log_reward: bool = False) -> float:
    """Reward for a completion whose <explanation> fields don't parse.
    Derived from FAILED_MSE through the same reward transform used for real
    rollouts, so it stays on a consistent scale with `loss_to_reward` (worse
    than any real completion's mean-predictor-baseline MSE ≈ 1.0)."""
    worst_loss = (weights.w_sum + weights.w_comp) * FAILED_MSE
    return loss_to_reward(torch.tensor(worst_loss), log_reward=log_reward)
