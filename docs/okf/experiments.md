---
type: Experiment Log
title: "Experiment log: every training run and evaluation, with commands, numbers and status"
description: Append-only record of runs (SFT, RL) and their evaluations; newest entries at the bottom. Add an entry for every run.
tags: [experiments, log, results]
timestamp: 2026-10-02
---

# How to use this log

- **Append-only.** Add a new entry at the bottom for every training run and every evaluation worth keeping; never rewrite old numbers (add a correction line instead).
- **Each entry:** ID, date, goal or hypothesis, what changed versus the previous run, the exact command(s), artifacts (paths on the cluster), results **with n**, conclusion, and what to do next. Mark anything not verified with CHECK and anything pending with TODO.
- Numbers from different n or different splits are not directly comparable; the summary table says which n each number has.
- Related pages: [open questions](/open-questions.md), [data collection v2](/plans/data-collection-v2.md), per-run readings in [observations](/index.md).
- Paths are relative to `/work/dfm/lukasgp/natural_language_autoencoders` on the cluster. `eval_iid_clean.parquet` is `eval_iid.parquet` minus prompts with near-duplicates in training (see [v2 plan](/plans/data-collection-v2.md)); entries before it existed used `eval_iid.parquet` (8.8% leaked prompts).

# Summary table (held-out, greedy; n per split in the entries)

| ID | model | held-out FVE sum / L / H (iid; ood) | non-last quote correct (iid; ood) | judge fraction of KL recovered (sum; z_H) | NLL gap (all tokens) | status |
|---|---|---|---|---|---|---|
| E1 | SFT v1 (`ckpt/av_sft`, `ckpt/ar_sft`) | not evaluated | not evaluated | not evaluated | +0.0136 | done |
| E2 | RL run 1 (`ckpt/rl`) | about 0 to -0.3 (n=10) | 0% (0 of 70% quoted) | iid -3.0 / -0.1, ood -0.3 / -1.0 (n=10) | not measured | done |
| E3 | RL run 2, log-reward (`ckpt/rl_logr`) | iid 0.03 / 0.03 / 0.05; ood -0.17 / -0.23 / 0.04 (n=50) | 0% | iid -1.5 / 0.09, ood -0.1 / -4.2 (n=50) | +0.0352 | done |
| E4 | SFT v2, token prefix (`ckpt/av_sft_tok`, `ckpt/ar_sft_tok`) | iid -2.9 / -2.8 / -8.0; ood -2.5 / -3.1 / -7.5 (n=100) | iid 22% (n=87); ood 9% (n=88) | not run | +0.0273 | done |
| E5 | RL run 3 on E4 (`ckpt/rl_tok`) | TODO | TODO | TODO | TODO | launched, results pending |

Reference points: chance for always guessing a frequent non-last token is about 5%; a linear probe reads the marked token with 99% accuracy from z_H (see [token probe](/observations/token-probe.md)). FVE after SFT only is strongly negative because the reconstructor's scale is uncalibrated; online AR training in RL repairs it within about 25 steps.

# E1: SFT v1 (format-targeted AV-SFT, AR-SFT)

- **Date:** 2026-09-30. **Goal:** warm-up of the AV and AR adapters on stream-neutral LLM explanations.
- **Data:** `av_sft.parquet` (9191 rows kept of 9324), `ar_sft.parquet`; explanations by GLM-5.3 ([teacher](/decisions/explanation-teacher.md)).
- **Commands (CHECK actual flags):** `python -m nla.hrm.train_ar_sft --train-parquet ar_sft.parquet --output ckpt/ar_sft`; `python -m nla.hrm.train_av_sft --train-parquet av_sft.parquet --norm-stats-json norm_stats.json --output ckpt/av_sft` (defaults: 1 epoch, early stop at greedy format rate 0.99).
- **Artifacts:** `ckpt/ar_sft`, `ckpt/av_sft`.
- **Results:** teacher-forced NLL per token, real versus shuffled vector, on 300 AR-SFT-bucket rows: 2.0985 vs 2.1122, gap +0.0136. Not evaluated by generation.
- **Conclusion:** the verbalizer reads the vector only slightly. The early stop at the format target probably ended training too soon (see the correction in [SFT decision](/decisions/sft-is-format-only.md)).

# E2: RL run 1 (200 steps, default reward)

- **Date:** 2026-09-30. **Command:** `python -m nla.hrm.train_rl --rl-parquet rl.parquet --av-sft-ckpt ckpt/av_sft --ar-sft-ckpt ckpt/ar_sft --sanity --output ckpt/rl` (defaults: 8 rows x 8 rollouts, 300 new tokens, lr 1e-5, kl_beta 0.01, plain -loss reward).
- **Training log:** stable, KL about 0.04, malformed from 3/64 to 0/64; reward dominated by the failure penalty; no FVE logged ([reading](/observations/rl-run-1.md)).
- **Eval (n=10 per split, `ckpt/rl/final`):** FVE sum / L / H iid -0.03 / -0.03 / -0.24, ood -0.15 / -0.18 / -0.30; marked-token quote 0% correct of the 70% that quote; judge sum-patch KL about 6.1 to 6.4 nats (noise floor 0.002 to 0.03); fraction of KL recovered vs mean ablation iid -3.0 sum / -0.14 z_H, ood -0.3 sum / -1.0 z_H.
- **Conclusion:** learned the format; reconstructions no better than the mean vector.

# E3: RL run 2 (700 steps, log-reward)

- **Date:** 2026-10-01 (overnight). **Change vs E2:** `--log-reward`, 16 rows x 8 rollouts, policy lr 2e-5.
- **Command:** `python -u -m nla.hrm.train_rl --rl-parquet rl.parquet --eval-parquet eval_iid.parquet --av-sft-ckpt ckpt/av_sft --ar-sft-ckpt ckpt/ar_sft --log-reward --batch-size 16 --group-size 8 --policy-lr 2e-5 --steps 700 --save-every 50 --eval-every 50 --sanity --output ckpt/rl_logr` (6 h 2 min, about 31 s/step).
- **Training log:** training-rollout `fve_sum` from -3.0 (step 1) to about 0.2 (steps 630 to 700); reward about 10.1; KL grew to 0.37 to 0.50; CJK hits 1 to 3 per 128.
- **Held-out eval (n=50 per split, `ckpt/rl_logr/final`):** FVE iid 0.032 / 0.030 / 0.048, ood -0.171 / -0.234 / 0.040; geometry cos to gold L 0.96, H 0.98, sum 0.73/0.67; quote correct 0% (70%/72% quoted); judge fraction of KL recovered iid -1.50 sum / 0.09 z_H, ood -0.13 / -4.20.
- **Controls:** shuffled vectors (`--shuffle-vectors`): FVE iid -0.230 / -0.164 / -0.295, ood -0.310 / -0.330 / -0.173, worse in all 6 comparisons; NLL gap +0.0352 (real 2.5316, shuffled 2.5668).
- **Samples:** a math prompt was verbalized as a children's story with invented marked tokens ([reading](/observations/rl-run-2-log-reward.md)).
- **Conclusion:** the training FVE did not transfer to held-out prompts; explanations depend on the vector only a little and never name the real token.

# E4: SFT v2 with token-prefix targets, no early stop

- **Date:** 2026-10-02. **Change vs E1:** `build.py --prefix-token` (each field starts with `Marked token: "X".`, [decision](/decisions/token-prefix-targets.md)); AV-SFT without the early stop, 3 epochs, checkpoints every 500 steps; AR-SFT 3 epochs.
- **Commands:** `python -m nla.hrm.build --input splits/{ar,av}_sft_explained.parquet --stage {ar,av}_sft --prefix-token --output {ar,av}_sft_tok.parquet`; `python -m nla.hrm.train_ar_sft --train-parquet ar_sft_tok.parquet --epochs 3 --output ckpt/ar_sft_tok`; `python -u -m nla.hrm.train_av_sft --train-parquet av_sft_tok.parquet --norm-stats-json norm_stats.json --target-format-rate 2 --epochs 3 --eval-every 500 --save-every 500 --output ckpt/av_sft_tok`.
- **NLL gap (300 AR-SFT-bucket rows, `--prefix-token`):** step 500 +0.0077; step 2000 +0.0165 (real 1.9749, shuffled 1.9914); final +0.0273 (real 2.0033, shuffled 2.0306). Still rising at the end of training. The marked-token-span-only NLL line was not yet available when these were run (TODO if re-run).
- **Generation eval (`eval --limit 100 --max-new-tokens 300`, clean iid and ood):** format rate 59% iid / 60% ood, all 81 of 200 malformed completions lack the closing tag (truncated at 300 tokens; mean length 193 vs 187 words, so the limit is too tight); non-last positions: quote correct 22% iid (n=87), 9% ood (n=88), quoted 100%; FVE sum / L / H iid -2.89 / -2.83 / -8.03, ood -2.54 / -3.13 / -7.53; cos to gold L/H 0.86 to 0.87, `|sum_hat|/|sum|` 1.14 to 1.26; `lh_jaccard` 0.19.
- **Conclusion:** token reading has started in-distribution (22% versus 0% in E2/E3; about 5% is chance) and is weak out of distribution; FVE is strongly negative as expected for an uncalibrated SFT-only reconstructor. Use these as the SFT-only baseline for E5.

# E5: RL run 3 from the token-prefix SFT checkpoints

- **Date:** launched 2026-10-02 (overnight). **Hypothesis:** with the token available to the reconstructor, correct token reading is rewarded, so RL should raise quote accuracy and calibrate the AR (FVE towards positive).
- **Change vs E3:** starts from E4; `--kl-beta 0.05` (was 0.01) and `--policy-lr 1e-5` (was 2e-5) to curb drift (KL reached 0.5 in E3); `--max-new-tokens 400` (was 300) because 40% of E4 completions were truncated.
- **Command:** `python -u -m nla.hrm.train_rl --rl-parquet rl.parquet --eval-parquet eval_iid_clean.parquet --av-sft-ckpt ckpt/av_sft_tok --ar-sft-ckpt ckpt/ar_sft_tok --log-reward --batch-size 16 --group-size 8 --policy-lr 1e-5 --kl-beta 0.05 --max-new-tokens 400 --steps 650 --save-every 50 --eval-every 50 --output ckpt/rl_tok`. Expected 6.5 to 7.5 h (about 35 to 42 s/step, unmeasured). CHECK: confirm the command actually used.
- **Watch rules:** abort if malformed stays above about 30/128 for several prints, or KL above 0.3 early; FVE should recover towards 0 within about 25 steps.
- **Planned evaluation:** `eval` on `ckpt/rl_tok/final` with `--max-new-tokens 400`, `--limit 100`, clean iid and ood, plus `--shuffle-vectors` as the control, `--run-judge --mean-from rl.parquet`; compare quote accuracy on non-last positions (E4 baseline: 22% iid / 9% ood) and FVE.
- **Results:** TODO.
