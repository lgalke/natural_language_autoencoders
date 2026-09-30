---
type: Decision
title: Both SFT stages are format-only; L/H differentiation is left to RL
description: Claude/GLM writes two independent context explanations per row and they are randomly assigned to the L and H fields.
tags: [sft, training, methodology]
timestamp: 2026-09-30
---

# Decision

No teacher can see z_L or z_H, so no teacher can write stream-specific targets. The SFT data therefore only teaches the output format:

- Per row, two independent explanations of the same context (`--samples-per-row 2`, temperature 1).
- AV-SFT response: `<explanation>\nL: <e_a>\nH: <e_b>\n</explanation>`, with the order of (e_0, e_1) chosen by a per-row keyed RNG.
- AR-SFT: one training row per field. `api_explanation_0` is paired with the L head and `z_L`, `api_explanation_1` with the H head and `z_H`. Fixed but arbitrary, since both are equally generic.
- AV-SFT stops early once the greedy format rate on eval reaches 0.99.

# Why

Any difference between the L and H fields after SFT would be an artefact of the random assignment, not evidence about the streams. Keeping SFT stream-neutral means any stream-specific content that appears later must come from the RL reward, which is the only stage where the reconstructor scores against the real z_L versus z_H.

# Consequences

- Do not read stream differences off SFT checkpoints.
- Evaluate the post-SFT checkpoints as a baseline next to the post-RL ones, to see what RL added; see [open questions](/open-questions.md).
- A risk is that the two fields collapse to duplicates; `eval.py` reports `lh_jaccard` (token overlap between the fields) to monitor this.
