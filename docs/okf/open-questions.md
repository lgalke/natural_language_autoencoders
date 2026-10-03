---
type: Open Questions
title: What is not yet known or not yet run
description: Unverified claims, missing measurements and untested components, to be resolved before writing results.
tags: [status, todo]
timestamp: 2026-09-30
---

# Central question, unanswered so far

Do the L and H fields of the AV's explanations carry stream-specific content, and does RL produce it? Nothing in hand answers this yet.

# Most urgent (as of RL run 3, E5)

Judge for E5: about mean-ablation level (fraction recovered iid -0.15, ood +0.18), so the explanations are vector-dependent but not yet functionally faithful; see [experiments](/experiments.md). Update: the plain shuffled-vector control shows the E5 FVE gain and the token reading depend on the specific vector (FVE iid 0.23 to -0.25, correct tokens 19% to 2%); word pieces are read at 19% iid. Still open: within-dataset shuffle, judge patch-back KL with mean ablation for E5, the 3% to 16% grounding of details, and how to raise token reading (token-span loss weight, structured targets).

First positive held-out FVE (iid 0.23), but explanations confabulate details (3% to 11% grounding) and RL did not improve token reading (19% iid). Is the FVE more than a dataset-level or token-class effect? Run `nla.hrm.baselines` (per-dataset and per-true-token mean predictors on the same eval rows) and a within-dataset shuffled-vector control; see [experiments](/experiments.md) E5.

# Earlier (as of RL run 2)

Update 2026-10-02: the SFT-length hypothesis is only partly supported (NLL gap after SFT is small but positive, +0.0136 nats/token; RL raised it to +0.0352; shuffled-vector FVE is lower than real-vector FVE, so the vector is used a little). Next diagnostic: `nla.hrm.probe_check` (linear-probe ceiling for reading the marked token from z_L/z_H), which decides whether "quote the marked token" is a learnable target.

Hypothesis to test first: the AV never learned to read the vector because AV-SFT stopped at the format target. Run `nla.hrm.nll_check` on `ckpt/av_sft` (gap between NLL with a shuffled and the real vector; about 0 means ignored). If so: retrain AV-SFT without early stopping (`--target-format-rate 2 --epochs N --save-every`), track the NLL gap, then redo RL.

Does the verbalizer use the vector at all, or does the positive training FVE come from text style? Needs: shuffled-vector control, held-out FVE with judge baselines, marked-token quote accuracy, held-out samples, the text-only baseline; see [RL run 2](/observations/rl-run-2-log-reward.md).

# Missing measurements

0. In-distribution eval prompts with near-duplicates in training: 8.8% (v1). Report iid numbers on `eval_iid_clean.parquet` (see [data collection v2](/plans/data-collection-v2.md)).

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

- The v1 data collection (ad-hoc corpus, uniform positions including the chat-template tail, prompt-only): see [data collection v2](/plans/data-collection-v2.md). Run `nla.hrm.data_report` and record the numbers there.

- Reward scale: the failure penalty dominates; candidates are `--log-reward`, a milder failure reward, or a larger `w_sum`.
- Number of RL steps and group size: 200 steps of 64 rollouts is small.
- Whether the injection scale used the verbalizer p75 statistic or the 5.0 fallback is not recorded.
- Positions: dense sampling within prompts versus always using the last prompt position only for headline numbers.

# Data not yet available

The HRMMix corpus (local path pending), so the reasoning part of the mix is currently the public stand-in sources ([corpus](/decisions/corpus-and-splits.md)).
