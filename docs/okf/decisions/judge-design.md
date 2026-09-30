---
type: Decision
title: Mimir patch-back judge is the primary faithfulness metric and is eval-only
description: Replace Mimir's H input (or z_H alone) with the reconstruction and measure KL to the clean forward pass.
tags: [evaluation, judge, faithfulness]
timestamp: 2026-09-30
---

# Decision

The faithfulness test runs Mimir as a frozen judge (`nla/hrm/judge.py`). It is never part of the RL loop.

- Primary patch: overwrite H_in@2 at the extraction position with `z_L_hat + z_H_hat`, run the rest of the forward pass, and report KL(clean || patched) over the next-token distribution at that position, at the last prompt position, and averaged.
- Secondary patch: overwrite H_out@1 with `z_H_hat` alone and let cycle 2 run normally. Because z_H persists through cycle 2, this is the causal test of the H component; the primary patch sees only the sum and is blind to the split.
- Checks: (a) patching back the vector recomputed in the same forward pass must reproduce the clean logits (KL below 1e-4, asserted: tests the hook itself); (b) patching the stored gold vector is reported as a noise floor, because in bf16 the stored vectors (from an extraction run with different batch composition) differ from recomputed ones by rounding noise (first GPU run: max KL 9e-3). A warning is printed above 5e-2 (bf16) or 1e-4 (float32), a hard error above 0.25.
- References: A mean-ablation baseline (train-set mean s, or mean z_H) gives a "fraction of KL recovered" = 1 - KL(pred)/KL(mean).
- The judge re-renders each row's exact original prompt from the stored `prompt_ids` column, not by re-tokenizing the decoded context text.

# Why eval-only

The judge needs an extra ~1B-parameter forward pass with a different model loaded; putting it in the per-step loop would dominate run time and make the primary metric also the training target.

# Smoke-test behaviour (mechanism check, not a result)

On 4 rows with gold vectors plus Gaussian noise (sigma 0.5): primary KL at the patch position 1.67, at the last prompt token 0.33, z_H-only patch 3.04. The true-vector assertions passed. This only shows the mechanism responds to reconstruction error.
