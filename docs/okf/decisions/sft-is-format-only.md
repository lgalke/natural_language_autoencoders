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

# Correction (2026-10-02): SFT is NOT only format learning

The explanations carry no stream information, but they do describe the specific token and context, so the SFT cross-entropy is also what teaches the verbalizer to read the injected vector (the format is learned in a few dozen steps; reading takes far longer). The AV-SFT script's early stop at "greedy format rate >= 0.99" very likely ended training long before the vector was read: later held-out evaluation showed explanations unrelated to the vector (marked-token quote accuracy 0%, FVE about 0; see [RL run 2](/observations/rl-run-2-log-reward.md)). Hypothesis, to be confirmed with `nla.hrm.nll_check` (teacher-forced NLL with real versus shuffled vectors). `--target-format-rate 2` disables the early stop.

# Consequences

- Do not read stream differences off SFT checkpoints.
- Evaluate the post-SFT checkpoints as a baseline next to the post-RL ones, to see what RL added; see [open questions](/open-questions.md).
- A risk is that the two fields collapse to duplicates; `eval.py` reports `lh_jaccard` (token overlap between the fields) to monitor this.
