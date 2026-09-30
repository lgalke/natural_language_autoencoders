---
type: Decision
title: Mixed reasoning and instruction prompts, group-level splits, one whole dataset held out
description: Corpus sources, position sampling, bucket fractions and the out-of-distribution holdout.
tags: [data, splits]
timestamp: 2026-09-30
---

# Corpus

`build_prompt_corpus.py` default sources: GSM-Symbolic (test, `question`), BBH causal_judgement (test, `input`), MuSR murder_mysteries (`narrative`), SimpleStories (`story`), and Danish DynaWord (`text`), sampled with streaming and a shuffle buffer, at most 2000 rows per source by default. ProofWriter is disabled (not found on the Hub under the guessed id). Local HRMMix corpora can be added with `--local-jsonl`; HRMMix itself was not available at the time of writing.

SimpleStories was added as a generic, freely available text source after the initial reasoning-only mix failed to load. It is narrative text, not reasoning prompts.

# Positions

Up to 6 positions per prompt (`--positions-per-prompt`), sampled with a per-(seed, doc_id) keyed RNG from token index 3 onwards (skipping the BOS and start of the chat template), plus always the last prompt position (the "answer position"). Prompts longer than 2048 rendered tokens are skipped, not truncated.

# Splits

- `eval_ood`: every row whose dataset equals `--holdout-dataset` (default MuSR), never used for training.
- The remaining rows are split by `world` (one group per prompt unless the corpus supplies clusters) into `av_sft` 30%, `ar_sft` 30%, `rl` 30%, `eval_iid` 10%.
- `judge_subset`: a random 50% of (eval_iid + eval_ood) rows, for the heavier Mimir judge.

# Why grouped

All positions of one prompt must stay in one bucket; otherwise context leaks between SFT, RL and evaluation.

# Caveat

MuSR narratives are long and distributionally unlike the rest, so `eval_ood` is a strong shift (long narrative versus short prompts), not a mild one.
