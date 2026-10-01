---
type: Observation
title: "RL run 2 (700 steps, log-reward): FVE turns positive, but samples look unfaithful and KL is large"
description: Training-log trends and a first look at sampled explanations; held-out evaluation and the shuffled-vector control were still pending when this was written.
tags: [rl, training, log, faithfulness]
timestamp: 2026-10-02
---

# Setup

Fresh start from the SFT checkpoints, `--log-reward` (reward = -log loss), 16 rows x 8 rollouts = 128 rollouts per step, policy lr 2e-5 (run 1: 1e-5), kl_beta 0.01, 700 steps, 6 h 2 min (about 31 s per step), output `ckpt/rl_logr`. Compare [RL run 1](/observations/rl-run-1.md).

# Training-log trends (training rollouts, sampled at temperature 1; single lines are noisy)

- Step 1: 53 of 128 completions malformed (sampling at temperature 1 is much noisier than the greedy format rate seen at the end of SFT); 3 to 11 of 128 by steps 10 to 40; mostly 0 to 4 at the end.
- `fve_sum`: -3.0 at step 1, near 0 (about -0.5 to 0) at steps 25 to 40, and mostly 0.05 to 0.35 (about 0.2) over steps 630 to 700; `fve_L` and `fve_H` similar. This is the first time reconstruction beat the mean predictor in any run (run 1 held-out FVE was about 0 or negative, n = 10).
- Reward about 9.3 early, about 10.1 at the end.
- KL to the post-SFT AV rose from about 0.05 (steps 10 to 40) to about 0.37 to 0.50 (steps 630 to 700): roughly ten times run 1.
- CJK-character hits persisted at 1 to 3 of 128 per batch (the regex also matches ordinary Chinese text and the marker characters themselves; not yet disambiguated).
- The `--sanity` pre-check printed real == shuffled == the failure reward, because all sanity completions were malformed; uninformative (it caps completions at 200 tokens).

# First look at samples (preliminary, two samples of one row from `samples.jsonl`)

For a prompt that is a math word problem (marked token " day"), both sampled explanations at step 660 describe a children's story, quote different invented marked tokens ("wiggled", "fell"), and mention characters not in the prompt. The text is fluent and in the style of the SFT explanations, but not faithful to this row. Possible reading: the reconstructor receives only the text, so style-level shortcuts can yield FVE above 0 without the explanation describing the particular vector. This is a hypothesis, not a finding; two samples of one row are not evidence about the distribution.

# Held-out evaluation of `ckpt/rl_logr/final` (n = 50 per split, greedy; shuffled control not yet run)

| split | fve sum / L / H | cos to gold L / H / sum | quoted marked token correct | judge sum-patch KL (noise floor) | fraction of KL recovered, sum / z_H-only |
|---|---|---|---|---|---|
| iid | 0.03 / 0.03 / 0.05 | 0.96 / 0.98 / 0.73 | 0% of the 70% that quote one | 6.11 (0.002) | -1.50 / 0.09 |
| ood (MuSR) | -0.17 / -0.23 / 0.04 | 0.95 / 0.98 / 0.67 | 0% of the 72% that quote one | 6.75 (0.002) | -0.13 / -4.20 |

Field overlap (lh_jaccard) 0.13 to 0.14. Reading: the training-rollout FVE of about 0.2 did not transfer to held-out prompts; explanations never quote the real marked token, so they are not reading it from the vector; reconstructions are no better than the mean vector under the judge. High cosines to gold reflect the large component shared by all vectors, not information; FVE against the mean predictor is the informative number.

# Leading explanation (hypothesis)

AV-SFT stopped as soon as the greedy format rate reached 0.99, which probably left the verbalizer never having learned to read the vector, so RL had nothing vector-dependent to amplify; see the correction in [SFT decision](/decisions/sft-is-format-only.md). Test: `nla.hrm.nll_check` on `ckpt/av_sft`.

# Shuffled-vector control and teacher-forced NLL (held-out rows, n = 50 / 300)

- Shuffled control (`eval --shuffle-vectors`, another row's vector injected, scored against the original gold): FVE sum / L / H drops from 0.03 / 0.03 / 0.05 to -0.23 / -0.16 / -0.30 on iid and from -0.17 / -0.23 / 0.04 to -0.31 / -0.33 / -0.17 on ood. Worse in all 6 comparisons, so the explanations do depend on the vector, but with the real vector FVE is still only about 0. Quote rates were identical in both runs (70% / 72% quoted, 0% correct).
- `nll_check` (mean NLL per token of the teacher explanation, AR-SFT-bucket rows the AV never trained on, 300 rows): `ckpt/av_sft` real 2.0985, shuffled 2.1122, gap +0.0136; `ckpt/rl_logr/final` real 2.5316, shuffled 2.5668, gap +0.0352. SFT taught the verbalizer to read the vector only slightly (0.6% of the NLL); RL increased the dependence about 2.6x while moving away from the teacher's wording (higher overall NLL).
- Reading: the verbalizer reads a small coarse signal (not token identity); the explanation text carries a modest amount of information relative to its large vector-independent part (wording, narrative details). A threshold of 0.01 nats/token printed by the script is an arbitrary rule of thumb, not a calibrated criterion.

# Checks queued to settle it

1. Shuffled-vector control (`eval --shuffle-vectors`): FVE with another row's vector injected. Equal FVE would mean the verbalizer ignores the vector.
2. Held-out FVE, geometry (cosine and norm ratio vs gold) and judge KL with mean-ablation baselines (`eval --mean-from rl.parquet --run-judge`).
3. Marked-token quote accuracy (`eval` now prints how often an explanation quotes a marked token and how often it is the real one).
4. Samples from held-out rows across datasets (`eval --dump-samples`), and an earlier checkpoint (for example step 300) to see whether held-out quality peaked before the KL grew.
5. The text-only baseline (`train_context_baseline`): how much FVE the true context text alone gives.

Results of these are TODO in [open questions](/open-questions.md).
