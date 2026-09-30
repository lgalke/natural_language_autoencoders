---
type: Observation
title: "RL run 1 (200 steps): stable and well-formed, but the reward is dominated by the failure penalty"
description: Reading of the training log; reconstruction quality relative to the mean predictor was not logged in this run.
tags: [rl, training, log]
timestamp: 2026-09-30
---

# Setup

Defaults: 200 steps, 8 rows x 8 rollouts = 64 per step, max 300 new tokens, policy lr 1e-5, AR lr 1e-4, kl_beta 0.01, w_sum 1.0, w_comp 0.25, single 95 GB GPU, `--micro-batch-size 8`. Total wall time 1 h 16 min (22.6 s per step). `--sanity` was on. The post-SFT AV and AR were the starting point.

# What the log shows (steps 60 to 200, printed every 5 steps)

- KL to the post-SFT reference stayed between about 0.03 and 0.06 with no upward drift (0.042 at step 60, 0.037 at step 200).
- Malformed completions: 0 to 3 of 64 between steps 60 and 115, then mostly 0 from step 120 on (0 at steps 120, 125, 135, 145, 155 and 170 to 200; 1 to 2 at steps 140, 150, 160, 165).
- CJK-character hits: 0 of 64 almost everywhere; 2 of 64 at step 75, 1 at steps 80, 195.
- `ar_loss` 0.0001 at step 60 and 0.0000 to 0.0001 at the end.
- Mean reward is either about 0 or an exact multiple of -0.039: one malformed completion contributes -2.5/64 = -0.039, so the reward is (well-formed ones about -1e-4) plus (malformed count x -0.039).
- The policy-gradient loss is about 0 throughout, with small sign changes.

# Interpretation (hedged)

- Training is mechanically healthy and RL mainly drove the malformed rate towards 0.
- The reconstruction part of the reward is about 1e-5 to 1e-4 in absolute terms. Because [normalization](/decisions/shared-scalar-normalization.md) makes even a trivial predictor score about 1e-3, this says nothing about quality without the mean-predictor baseline from `norm_stats.json`. That comparison was not logged, so this run alone cannot show that reconstruction improved.
- With the failure penalty 4 to 5 orders of magnitude larger than differences between well-formed completions, the group-normalized advantage among well-formed completions is close to noise.

# Follow-up tooling (added afterwards, not yet exercised on a real run)

Per-step FVE in the log, `samples.jsonl` from RL, `eval --dump-samples`, and `infer.py`; see [open questions](/open-questions.md) for the evaluation that should settle this.
