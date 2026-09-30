---
type: Open Questions
title: What is not yet known or not yet run
description: Unverified claims, missing measurements and untested components, to be resolved before writing results.
tags: [status, todo]
timestamp: 2026-09-30
---

# Central question, unanswered so far

Do the L and H fields of the AV's explanations carry stream-specific content, and does RL produce it? Nothing in hand answers this yet.

# Missing measurements

1. FVE per term (sum, L, H) of the final RL checkpoint and of the post-SFT checkpoints on `eval_iid` and `eval_ood`. The RL run did not log it; run `nla.hrm.eval` with `--norm-stats-json` on both.
2. Patch-back judge KL with mean-ablation references, for both the sum patch and the z_H-only patch (`eval --run-judge`).
3. The text-only context baseline (`train_context_baseline.py`): the bar the explanations must beat.
4. The cross-reconstruction matrix (`eval_cross.py`): diagonal versus off-diagonal FVE as the test for stream-specific content.
5. Field monitoring: `lh_jaccard` (duplicate fields) and whether any field collapses to empty.
6. Full-corpus cancellation statistics; the only numbers recorded are from a 12-row smoke corpus ([cancellation](/observations/cancellation.md)).
7. Quality of the GLM explanations: only the drop rate was checked (1.4%).

# Untested code

- `train_context_baseline.py` and `eval_cross.py` were never executed end to end.
- The GPU path of most new code (bf16) was only smoke-tested on CPU; the padded-batch equivalence was checked in float32 only ([PrefixLM](/decisions/prefixlm-rendering.md)).
- `infer.py`, RL FVE logging and sample dumps were smoke-tested on a tiny CPU setup only.

# Decisions that may need revisiting

- Reward scale: the failure penalty dominates; candidates are `--log-reward`, a milder failure reward, or a larger `w_sum`.
- Number of RL steps and group size: 200 steps of 64 rollouts is small.
- Whether the injection scale used the verbalizer p75 statistic or the 5.0 fallback is not recorded.
- Positions: dense sampling within prompts versus always using the last prompt position only for headline numbers.

# Data not yet available

The HRMMix corpus (local path pending), so the reasoning part of the mix is currently the public stand-in sources ([corpus](/decisions/corpus-and-splits.md)).
