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
| E5 | RL run 3 on E4 (`ckpt/rl_tok`) | iid 0.23 / 0.21 / 0.25; ood 0.03 / -0.02 / 0.19 (n=100); non-last: iid 0.15 / 0.14 / 0.15, ood -0.07 / -0.12 / 0.10 | iid 19% (n=87); ood 8% (n=88) | iid -0.15 / -0.04, ood 0.18 / -1.48 (n=100) | not measured | done; within-dataset shuffle pending |
| E6 | SFT with token-span loss weight 8 (`ckpt/av_sft_tokw`, AR from E4) | iid -2.03 / -2.64 / -8.10; ood -2.25 / -3.19 / -8.40 (n=100; SFT-only, uncalibrated AR) | iid 49% (n=87); ood 30% (n=88) | not run | all-token +0.0392; token-span +0.7533 | done |
| E7 | RL on E6 (`ckpt/rl_tokw`), 1300 steps | TODO | TODO | TODO | TODO | running (launched 2026-10-03) |

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
- **Training log (first step):** 41.7 s/step (ETA 7.5 h), malformed 25/128, `fve_sum` -3.5 / -3.1 / -8.4 at step 1 (mid-run and final values: TODO from `rl_tok.log`).
- **Held-out eval (`eval --limit 100 --max-new-tokens 400`, `ckpt/rl_tok/final`, clean iid and ood):**
  - format rate 99% iid / 100% ood (the truncation fix worked);
  - FVE sum / L / H: iid 0.228 / 0.205 / 0.245, ood 0.029 / -0.022 / 0.189; non-last positions only: iid 0.153 / 0.137 / 0.154 (n=87), ood -0.070 / -0.124 / 0.098 (n=88);
  - geometry: cos to gold sum 0.79 iid / 0.74 ood, L 0.965 / 0.955, H 0.987 / 0.986; `|sum_hat|/|sum|` 0.82; `lh_jaccard` 0.18;
  - marked-token quote (non-last): correct 19% iid, 8% ood (E4 SFT-only baseline 22% / 9%: RL did not improve token reading); all positions: 30% iid / 19% ood of the 99% to 100% that quote one;
  - grounding in the prompt text (new metric, approximate): 11% of 535 (iid) and 16% of 505 (ood) quoted spans occur verbatim in the context; 3% of 238 (iid) and 6% of 212 (ood) capitalised names occur in the context.
- **Samples (5 non-last rows):** coarse properties right (fiction versus question, Nordic language), details invented (stock names such as Leo, Lily, Anne, Starfall; a tourism article read as a question about photographing mosques); wrong tokens are close to the true ones in kind (`.` read as `,` or `:`, ` desire` read as `hope`), so the verbalizer may read the token class and not its identity (five anecdotes only).
- **Conclusion:** first positive held-out FVE, mostly in-distribution; token reading unchanged by RL; explanations confabulate specifics. Open: how much of the FVE is explained by knowing only the dataset or the true token (`nla.hrm.baselines`), and whether it depends on the vector (shuffled control, ideally shuffling within a dataset).
- **Group-mean baselines** (`nla.hrm.baselines --limit 100`, same 100 rows per split as the eval; FVE sum / L / H; train rows = base rows outside every eval file, 28248):

| predictor | iid all rows | iid non-last (n=87) | ood all rows | ood non-last (n=88) |
|---|---|---|---|---|
| global mean | 0.046 / 0.015 / 0.087 | 0.002 / -0.015 / 0.015 | -0.057 / -0.124 / 0.103 | -0.118 / -0.184 / 0.040 |
| per dataset | 0.084 / 0.060 / 0.106 | 0.036 / 0.023 / 0.035 | same as global (dataset unseen) | same as global |
| per dataset x last-pos | 0.158 / 0.136 / 0.171 | 0.056 / 0.043 / 0.052 | same as global | same as global |
| per TRUE marked token | 0.280 / 0.173 / 0.469 | 0.226 / 0.124 / 0.402 | 0.028 / -0.085 / 0.270 | -0.067 / -0.182 / 0.179 |
| **E5** | 0.228 / 0.205 / 0.245 | 0.153 / 0.137 / 0.154 | 0.029 / -0.022 / 0.189 | -0.070 / -0.124 / 0.098 |

  Reading (paired, same rows, n=100, no confidence intervals): E5 beats the per-dataset x last-position baseline by about 0.07 (all rows) and 0.10 (non-last) on every term, so the explanation carries information beyond source and position; roughly a third of the all-rows FVE is what that baseline already gets. Knowing the true token would give z_H an FVE of 0.40 to 0.47 against E5's 0.15 to 0.25 (z_L: E5 is at or above the true-token baseline), so token reading is where the headroom is, mainly for H. On ood E5's sum FVE equals the true-token baseline although it quotes the right token only 8% of the time (other coarse features carry the information).
- **Shuffled-vector control** (`eval --shuffle-vectors`, plain shuffle: another random row's vector injected, scored against the original gold; n=100 per split, same rows): FVE sum / L / H iid -0.249 / -0.212 / -0.233 (non-last, n=87: -0.285 / -0.232 / -0.299), ood -0.283 / -0.290 / -0.147 (non-last -0.340 / -0.347 / -0.201), versus normal iid 0.228 / 0.205 / 0.245 and ood 0.029 / -0.022 / 0.189; non-last marked-token correct 2% iid, 0% ood (versus 19% / 8%); grounding 5% / 3% iid, 14% / 5% ood (versus 11% / 3%, 16% / 6%); cos to gold unchanged in kind (L 0.95, H 0.98, sum 0.64). Conclusion: the FVE gain and the token reading both depend on the specific vector (they vanish to below the global mean and to chance); the within-dataset shuffle, which also removes dataset-level information, has not been run.
- **Token-type split of the non-last quote accuracy (from the dump):** iid punctuation or space 2/13 (15%), word piece 14/72 (19%); ood punctuation or space 2/8 (25%), word piece 5/80 (6%). Word-piece identities are read well above chance (a guess among thousands of pieces would almost never be right), contradicting the earlier guess that only the token class is read; small counts.
- **Judge (`eval --run-judge --mean-from rl.parquet --reuse-generations`, n=100 per split, bf16):** sum-patch KL at the position iid 4.60 / ood 5.43 nats (noise floor 0.004 / 0.002; E3: 6.11 / 6.75); fraction of KL recovered vs mean ablation iid -0.152 / ood +0.178 (E3: -1.50 / -0.13), i.e. about the level of the mean vector (mean-ablation KL implied: about 4.0 iid, 6.6 ood); z_H-only patch KL iid 2.50 / ood 2.74, recovered -0.038 / -1.478 (E3: 0.09 / -4.20). Stored-gold drift warnings (max KL 0.09 to 0.14 in a few rows) are bf16 rounding. Per-row ratios are noisy and heavy-tailed.
- **Conclusion (judge):** absolute disturbance of Mimir's output is lower than E3 but the reconstruction is not clearly better than the mean vector: the explanations are vector-dependent (shuffle control) yet not functionally faithful. The reward (normalised MSE) is dominated by high-variance directions, while Mimir's output depends on token and context; reading the token (19%) is the likely limit. Candidate next experiments: AV-SFT with a higher loss weight on the token span (`train_av_sft --prefix-weight`), structured and shorter targets, and possibly a judge-aligned reward term on a subset of rollouts.
- **Next:** (1b) within-dataset shuffled control: `eval --shuffle-vectors --shuffle-within-dataset`; (1c) done (judge, above); (2) `eval --shuffle-vectors` on `rl_tok/final`; (3) judge with `--mean-from rl.parquet`; (4) consider structured, shorter targets restricted to vector-determined facets ([v2 plan](/plans/data-collection-v2.md)).

# E6: AV-SFT with a higher loss weight on the marked-token span

- **Date:** 2026-10-03. **Hypothesis:** the token span is about 6 of ~250 target tokens, so at weight 1 it barely moves the loss; weighting it should improve token reading. **Change vs E4:** `train_av_sft --prefix-weight 8` (everything else as E4; the AR is E4's `ckpt/ar_sft_tok`, unchanged since its input format is the same).
- **Command (CHECK the flags actually used):** `python -u -m nla.hrm.train_av_sft --train-parquet av_sft_tok.parquet --norm-stats-json norm_stats.json --target-format-rate 2 --epochs 3 --prefix-weight 8 --eval-every 500 --save-every 500 --output ckpt/av_sft_tokw`.
- **Artifacts:** `ckpt/av_sft_tokw`.
- **Teacher-forced NLL (`nll_check --prefix-token`, 300 AR-SFT-bucket rows) versus E4's final checkpoint:** all-token gap (shuffled minus real) +0.0392 (real 1.9993, shuffled 2.0385) versus +0.0273 (real 2.0033, shuffled 2.0306); token-span-only NLL real 0.2880 / shuffled 1.0413, gap +0.7533, versus real 0.3263 / shuffled 0.7838, gap +0.4575. The span includes near-deterministic template pieces (`Marked token: "`, closing quote and period), so the token content carries most of the gap.
- **Generation eval (`eval --av-ckpt ckpt/av_sft_tokw --ar-ckpt ckpt/ar_sft_tok --max-new-tokens 400 --limit 100`, clean iid and ood):** format rate 100% / 100%; non-last positions: marked token correct 49% iid (n=87) and 30% ood (n=88), quoted 99%; all positions: 56% iid / 38% ood of the quoted ones are correct (E4: 22% / 9% non-last; E5: 19% / 8%); FVE sum / L / H iid -2.031 / -2.641 / -8.102 (non-last -1.877 / -2.712 / -8.415), ood -2.247 / -3.187 / -8.397 (non-last -2.214 / -3.312 / -8.600), strongly negative as for every SFT-only checkpoint (uncalibrated AR scale, `|sum_hat|/|sum|` 1.11 / 1.12; cos to gold sum 0.38 / 0.34, L 0.85 / 0.82, H 0.86 / 0.85); `lh_jaccard` 0.19; grounding 12% of 511 quoted spans and 2% of 245 names (iid), 13% of 500 and 4% of 362 (ood).
- **Conclusion:** the weighting increases vector dependence of the token (span gap 1.6x, all-token gap 1.4x) without hurting the rest of the explanation, and more than doubles generated token accuracy (22% to 49% iid, 9% to 30% ood) before any RL. Confabulation of details is unchanged (the targets were not changed).
- **Next:** E7 (RL on top, below).

# E7: RL run on the weighted-token SFT checkpoints, 1300 steps

- **Date:** planned 2026-10-03 (about 15 h expected at about 41 s/step, unmeasured for this run). **Hypothesis:** with token reading at 49% / 30% after SFT, RL (which also calibrates the AR) should give positive held-out FVE and improve functional fidelity (judge) more than E5; the longer run plus a checkpoint every 100 steps shows where it stops helping and whether RL erodes the token reading (E5: 22% to 19%).
- **Change vs E5:** AV from E6 (`ckpt/av_sft_tokw`), 1300 steps instead of 650, checkpoints and eval every 100 steps; all other settings equal (AR from E4, `--log-reward`, 16 x 8, lr 1e-5, kl_beta 0.05, 400 tokens).
- **Command:** `python -u -m nla.hrm.train_rl --rl-parquet rl.parquet --eval-parquet eval_iid_clean.parquet --av-sft-ckpt ckpt/av_sft_tokw --ar-sft-ckpt ckpt/ar_sft_tok --log-reward --batch-size 16 --group-size 8 --policy-lr 1e-5 --kl-beta 0.05 --max-new-tokens 400 --steps 1300 --save-every 100 --eval-every 100 --output ckpt/rl_tokw`. CHECK that this is what was launched.
- **Watch rules:** as E5 (malformed above about 30/128 for several prints, KL above about 0.3 early; FVE recovers towards 0 within about 25 steps). No resume; to continue after a crash, start from a saved step with `--av-sft-ckpt ckpt/rl_tokw/step_N --ar-sft-ckpt ckpt/rl_tokw/step_N` (the KL anchor moves to that checkpoint).
- **Planned evaluation:** `eval --limit 100 --max-new-tokens 400` on `step_300`, `step_700`, `step_1000` and `final` (quote accuracy on non-last positions, FVE, grounding), then for the chosen checkpoint the full suite: plain and within-dataset shuffle controls, `nla.hrm.baselines`, and the judge with `--mean-from rl.parquet`.
- **Launch log (steps 1 to 47):** 52.9 s for step 1, then about 37 s/step (ETA about 13 h); malformed 7/128 at step 1, 0 to 1/128 from step 15 (one spike of 6/128 at step 25); training-rollout `fve_sum` / L / H -4.03 / -3.42 / -9.60 at step 1, about 0 at step 15 to 20, 0.07 to 0.32 (sum) at steps 25 to 45; reward 7.9 to about 10; KL only 0.002 to 0.008 (E5's first steps were higher), so early gains are mostly AR calibration; CJK hits 0 to 2/128 from step 1 (present in the SFT model).
- **Results:** TODO.
