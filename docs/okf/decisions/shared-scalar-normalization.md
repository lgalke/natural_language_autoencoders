---
type: Decision
title: Normalize z_L, z_H and their sum by one shared per-token scalar
description: Dividing all three by sqrt(||z_L||^2 + ||z_H||^2) keeps additivity exact; per-stream normalization would break it.
tags: [loss, normalization]
timestamp: 2026-09-30
---

# Decision

`nla/hrm/recon.py:shared_normalize` divides z_L, z_H and s = z_L + z_H by the same scalar, `sqrt(||z_L||^2 + ||z_H||^2)` per token, computed in float32. Then `normalize(z_L) + normalize(z_H) == normalize(z_L + z_H)` exactly.

# Why

The loss has a term on the reconstructed sum (what H actually reads). If each stream were normalized by its own norm, the sum of the normalized streams would no longer correspond to the normalized sum, and that term would stop meaning "did you predict what H reads".

# Details

- The prediction is scored in the gold vector's normalization frame (the scale comes from the gold z_L, z_H), so an off-scale prediction is penalised rather than silently rescaled.
- The combined norm of (z_L_n, z_H_n) is 1 by construction; the norm of s_n depends on cancellation between the streams.
- Consequence for interpreting numbers: a per-element MSE on a unit-norm 1536-dim vector is of order 1e-3 even for a trivial predictor, so raw loss values are tiny. Always read them against the mean-predictor baseline; see [RL run 1](/observations/rl-run-1.md).

# Not done

Data-gen still stores raw vectors (repo invariant); normalization happens only at injection and loss time.
