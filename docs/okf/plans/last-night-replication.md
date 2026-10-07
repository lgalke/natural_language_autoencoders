---
type: Plan
title: "Last night before the talk: replicate the split pipeline, two cheap diagnostics"
description: What is still worth running on the GPU the night before the talk (2026-10-07 to 10-08), with the exact commands, the decision rules and the list of runs that are not worth it.
tags: [plan, replication, split-av, talk]
timestamp: 2026-10-07
---

# Context

The pipeline is finished: joint verbalizer E1 to E8, split verbalizer E9, continuation E10, bootstrap intervals and plots (all in [experiments](/experiments.md)). One night of GPU (about 10 to 12 h) is left before the talk is finalised.

What is robust (n=100 rows per split, intervals over rows only):
- the vectors carry the information: linear probe 0.99 from z_H, 0.81 from z_L; the shuffle controls collapse the effects;
- the judge removes about 40 to 50% of the output KL versus the mean vector, row-specifically (shuffled reconstructions are worse than the mean);
- the joint verbalizer's FVE equals the cross-stream ridge baseline (its text carries the shared part of the streams); swapping the streams changes nothing;
- the split verbalizer matches the joint one on iid and shows a clean token dissociation: L call 0.14 (iid) / 0.05 (ood) versus H call 0.40 to 0.47 / 0.30 (non-last).

What is NOT robust: any FVE difference of about 0.05 to 0.1 between models or checkpoints (E10 minus E9 was -0.085 on ood, as large as split minus joint). The main open weakness is that every comparison is one training run.

# Runs for tonight, in this order

All commands run on the cluster in the repo root after `git pull origin hrm`.

**1. SFT diagnostic with a stronger token-span weight (about 16 min SFT + 5 min eval).** Same split data and epochs as the first split SFT; only the weight changes (8 to 32). Question: is the L call at chance because of the training signal (it rises) or because this verbalizer cannot extract the token from z_L although the probe reads 0.81 (it stays flat)?

```bash
python -u -m nla.hrm.train_av_sft --train-parquet av_sft_split.parquet \
    --norm-stats-json norm_stats.json --target-format-rate 2 --epochs 2 \
    --prefix-weight 32 --eval-every 100000 --output ckpt/av_split_sft_w32

python -m nla.hrm.eval --eval-parquet iid=eval_iid_clean.parquet ood=eval_ood.parquet \
    --av-ckpt ckpt/av_split_sft_w32 --ar-ckpt ckpt/rl_tokw/final \
    --norm-stats-json norm_stats.json --limit 100 --max-new-tokens 400 --split-av \
    --dump-samples samples_split_sft_w32.jsonl --output eval_split_sft_w32.json
```

Reference (split SFT at weight 8, non-last): iid L 10.3% / H 36.0%, ood L 3.4% / H 19.5%.

**2. Replicate the split pipeline end to end (the overnight job, about 6.5 h).** Fresh SFT with the same recipe as run 1, a quick check of the SFT replicate, then 500 RL steps with the E9 command. No code changes; different output directories.

```bash
python -u -m nla.hrm.train_av_sft --train-parquet av_sft_split.parquet \
    --norm-stats-json norm_stats.json --target-format-rate 2 --epochs 2 \
    --prefix-weight 8 --eval-every 100000 --output ckpt/av_split_sft_r2

python -m nla.hrm.eval --eval-parquet iid=eval_iid_clean.parquet ood=eval_ood.parquet \
    --av-ckpt ckpt/av_split_sft_r2 --ar-ckpt ckpt/rl_tokw/final \
    --norm-stats-json norm_stats.json --limit 100 --max-new-tokens 400 --split-av \
    --dump-samples samples_split_sft_r2.jsonl --output eval_split_sft_r2.json

python -u -m nla.hrm.train_rl --rl-parquet rl.parquet \
    --av-sft-ckpt ckpt/av_split_sft_r2 --ar-sft-ckpt ckpt/rl_tokw/final --split-av \
    --log-reward --batch-size 16 --group-size 8 --policy-lr 1e-5 --kl-beta 0.05 \
    --w-comp 1.0 --max-new-tokens 300 --steps 500 --save-every 100 \
    --output ckpt/rl_split_r2 2>&1 | tee rl_split_r2.log
```

The SFT replicate check compares per-field token accuracy with run 1 (above). The replicate's own random seed is not fixed, so it is an independent draw of the SFT shuffle and of the RL sampling.

**3. If time allows, alongside the RL (memory permitting): the text-only context baseline.** `nla/hrm/train_context_baseline.py` predicts z_L and z_H from the prompt text with the same architecture and was never run (expect small bugs; read it and its `--help` first; drop it if it is not running within an hour). It gives the FVE that any description of the context can reach, which answers the "how much is just context inference" confound. Check `nvidia-smi` first: a 3 am out-of-memory error must not kill the RL job.

**4. Optional, only if the model downloads: a capacity probe.** AV-SFT with `--verbalizer-model Qwen/Qwen2.5-3B-Instruct` (E6 recipe: `--target-format-rate 2 --epochs 3 --prefix-weight 8 --eval-every 100000`) on `av_sft_tok.parquet`, then `nla.hrm.nll_check --verbalizer-model Qwen/Qwen2.5-3B-Instruct --prefix-token --av-ckpt ... --parquet splits/ar_sft_explained.parquet --sidecar-from av_sft_tok.parquet --limit 300`. Reference (1.5B, E6): token-span NLL gap +0.7533, all-token gap +0.0392, token accuracy 49% / 30%.

# Not worth running

- More RL from E9: E10 (600 more steps) lowered the held-out FVE.
- A per-field token-bonus RL run: the joint model's token bonus (E8) raised the training metric but not the held-out one; half a day with no time to evaluate it.
- Penalty rewards on quotes or names (E8: gamed through the blind spot), best-of-N (no gain over greedy), a Mimir copy or a 7B verbalizer (cannot be validated before Friday), new data (v2 plan).

# Morning checklist (about 45 min of GPU, then analysis)

1. `eval --split-av` on `ckpt/rl_split_r2/final` with `--run-judge --mean-from rl.parquet --judge-shuffle roll`, `--dump-samples samples_split_r2_final.jsonl`, `--output eval_split_r2_final.json`; commit the dump.
2. `python -m nla.hrm.bootstrap --dump E7=samples_tokw_final.jsonl E9=samples_split_final.jsonl R2=samples_split_r2_final.jsonl --compare E9 R2 --compare E7 R2 --out results.json`; judge intervals with `--judge NAME=eval_*.json` on the cluster; figures with `nla.hrm.plots`.
3. Decision rule: if the replicate shows the same iid parity with the joint model and the same token dissociation (L call below about 0.2, H call above about 0.3, iid non-last), report the split result as replicated and quote the E9 versus R2 spread as the run-to-run variance; otherwise report it as one run and say so.
4. Log every result in [experiments](/experiments.md), push, then finalise the slides.

# Without a GPU, during the night

Slide outline and results page from the experiment log and the figures in `docs/okf/figures/`, with "say / do not say" notes; refresh the comparison figures once the dumps arrive.

# Added: mini experiment on "H is easier to decode than L" (probe profile, no RL, minutes)

Where the hypothesis stands: the probe reads the token at the extraction position from z_H at 0.991 and from z_L at 0.808; the split verbalizer reads it from z_H at 40 to 47% and from z_L at 5 to 14%; z_H is better predicted from z_L (R^2 0.34) than the reverse (0.22). Open: is it "z_H is a more linearly informative summary of everything", or "z_L encodes DIFFERENT things (for example other positions)", and is the L information absent or only nonlinearly encoded?

`probe_check` now has a profile mode (tests in `tests/hrm/test_probe_profile.py`, planted-structure check on synthetic data):

```bash
# which information does each stream hold? token at position + K, K = -3 .. +3 (positive K = later tokens, visible to a bidirectional state)
python -m nla.hrm.probe_check --base base.parquet --non-last --offsets -3 -2 -1 0 1 2 3 --mlp-hidden 512 2>&1 | tee probe_profile.txt
# data hunger versus information gap: learning curve at the extraction position
python -m nla.hrm.probe_check --base base.parquet --non-last --offsets 0 --train-sizes 500 2000 8000 --mlp-hidden 512 2>&1 | tee probe_curve.txt
```

How to read it (accuracies carry a 95% binomial interval; treat gaps below about 0.03 as ties):
- z_H above z_L at every offset, same margin: z_H is simply the more linearly informative state.
- z_L above z_H at some offsets (for example -1, -2): the streams hold different things (L local, H the current token and context); this is the interesting outcome.
- The MLP closes the L gap at offset 0: the token is in z_L but nonlinearly encoded, so "harder to read" is about accessibility, which also explains why a small LoRA verbalizer with an affine adapter fails on it.
- The learning curve: if z_L at 8000 rows approaches z_H, the gap is data hunger; if it stays flat, it is an information gap.
- Caveats: top-200 classes per offset, no shuffled-label control (the majority baseline is printed), the last prompt position is dropped with `--non-last`.
