# Handover: HRM verbalizer (`nla/hrm`, branch `hrm`)

You are picking up a research project mid-run, working directly on the GPU cluster. This file is model-agnostic
and deliberately explicit. Read it fully, then do the first steps under "Start here".

## What this is

A natural-language autoencoder that verbalizes the two recurrent states (fast **L**, slow **H**) of the
hierarchical reasoning model DFM-Mimir-v1.5, to test whether the L and H explanations differ and are faithful.
Verbalizer/reconstructor = frozen Qwen2.5-1.5B-Instruct + two LoRA adapters. Code: `nla/hrm/`. The rest of the repo
(`nla/` outside `hrm/`, Miles/SGLang) is upstream NLA and is NOT used by this work.

## Read in this order

1. `CLAUDE.md`: repo invariants, and the section "HRM extension" (load-bearing rules for `nla/hrm/`).
2. `docs/okf/index.md`, then `docs/okf/open-questions.md` and `docs/okf/observations/pitfalls.md`.
3. `docs/hrm.md`: design and the step-by-step run commands (full run).
4. `docs/okf/paper/methods-notes.md` only when writing up.

## Rules of engagement (important)

- Ask the user before anything expensive or irreversible: training runs (an RL run is over an hour), deleting or
  overwriting `ckpt/`, `splits/`, or any `*.parquet`.
- **Never re-run `nla.hrm.split` on the existing data.** It reshuffles the buckets; checkpoints were trained on the
  current ones and eval rows could leak into training.
- Never print, log or commit secrets. Keys live in `.env` (git-ignored) or environment variables; check presence only.
- Do not commit data, checkpoints or reports (they are git-ignored; keep it that way). Commit and push only when the
  user asks. Small commits, one concern each.
- Do not change defaults (loss weights, reward, hyperparameters) silently. Propose the change, say why, and record the
  outcome in `docs/okf/` afterwards.
- If something contradicts these docs, trust the code (`--help` on every script) and tell the user.
- Run long jobs in the background with output to a log file (`nohup ... > log.txt 2>&1 &` or tmux) and poll the log;
  do not block on them.

## Environment

- Repo on the cluster: `/work/dfm/lukasgp/natural_language_autoencoders`. Pipeline artifacts live in the repo root
  (git-ignored): `corpus.jsonl`, `base.parquet`, `splits/`, `norm_stats.json`, `av_sft.parquet`, `ar_sft.parquet`,
  `rl.parquet`, `eval_iid.parquet`, `eval_ood.parquet`, `judge_subset.parquet`, `ckpt/{ar_sft,av_sft,rl}`.
- One ~95 GB GPU. All scripts default to CUDA + bf16 when available. Mimir needs `transformers>=5.13`.
- Explanation LLM (only needed to regenerate SFT data): `nla.hrm.openai_compat.UCloudGLMProvider`, key in
  `UCLOUD_API_KEY` or `./.env`. Not needed for evaluation.
- `python -m nla.hrm.preflight` checks GPU, package versions, which artifacts exist, and whether keys are set.
- Lint and tests: `ruff check nla/hrm tests/hrm && python -m pytest tests/hrm -q` (12 tests, no GPU needed).

## State of the project

Done: data extraction, split, explanations (GLM-5.3, 9191 of 9324 AV-SFT rows kept), both SFT stages, one
200-step RL run (`ckpt/rl/final`, about 76 min, stable, KL to post-SFT about 0.04, malformed completions down to 0 of 64).

**Not known yet** (the central open question): whether RL improved reconstruction beyond the mean predictor, whether
the L and H fields differ, and the patch-back judge numbers. The RL log alone cannot tell: the reward is dominated by the
malformed-completion penalty and absolute losses are about 1e-4 (see `docs/okf/observations/rl-run-1.md`).

Untested code: `nla/hrm/train_context_baseline.py`, `nla/hrm/eval_cross.py` (never executed end to end; expect small bugs).

## Start here

1. `python -m nla.hrm.preflight` and `git status`; confirm branch `hrm` and a clean tree.
2. Quick evaluation of the RL checkpoint (about a minute, 10 random rows per split):
   ```bash
   python -m nla.hrm.eval --eval-parquet iid=eval_iid.parquet ood=eval_ood.parquet \
       --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final --norm-stats-json norm_stats.json \
       --limit 10 --dump-samples samples_rl.jsonl --run-judge --output eval_rl.json
   ```
3. Same for the post-SFT checkpoints for comparison: `--av-ckpt ckpt/av_sft --ar-ckpt ckpt/ar_sft`,
   `--dump-samples samples_sft.jsonl`, `--output eval_sft.json`.
4. Read 10 to 20 entries of `samples_rl.jsonl` (fields: `context_marked`, `L_field`, `H_field`, `fve`). Judge whether the
   text is about the prompt, whether L and H differ, and whether fields are boilerplate or near-duplicates
   (`lh_jaccard` close to 1). Report this to the user in plain language before doing anything else.
5. If the quick numbers look sane, run the full evaluation without `--limit`, then the remaining steps:
   - text-only baseline and cross-reconstruction (step 11 in `docs/hrm.md`); fix bugs you hit and describe them;
   - full-corpus cancellation numbers: `python -m nla.hrm.diagnostics --input base.parquet`.
6. Only after the evaluation is interpreted, propose next experiments (candidates: `--log-reward`, a milder failure
   reward, higher `--w-sum`, longer RL). Get approval before launching them.

## Record what you learn

After each meaningful result or decision, update `docs/okf/` (OKF v0.1: every concept file needs frontmatter with a
`type`; index.md lists entries; log.md has dated entries; absolute links like `/decisions/x.md`):
- results and surprises: a file in `docs/okf/observations/`; link it from `index.md` and add a line to `log.md`;
- resolved or new unknowns: edit `docs/okf/open-questions.md`;
- numbers for the paper: replace the matching TODO in `docs/okf/paper/methods-notes.md`. Never report smoke-test numbers
  as results; state sample sizes.

## Top gotchas (details in `docs/okf/observations/pitfalls.md`)

- `apply_chat_template(tokenize=True)` returns a `BatchEncoding`; index `["input_ids"]`.
- Always `add_special_tokens=False` for Mimir prompts (the template already contains `<bos>`).
- OOM: lower `--micro-batch-size` (RL) or `--max-batch-tokens` (extraction); do not build full-vocabulary logits when only
  hidden states are needed (`logits_to_keep=1`).
- `--sanity` showing real == shuffled is usually malformed completions, not broken injection; check the `malformed=` count.
- PEFT saves adapters under `<output>/<adapter_name>/`; load with `nla/hrm/model.py:load_sft_adapter` / `load_rl_checkpoint`.
- FVE is the meaningful metric; raw MSE is about 1e-3 even for a trivial predictor (shared-scalar normalization).
- `nla/hrm/injection_token_cache_hrm.yaml` is machine-local and git-ignored; markers are also stored in every parquet sidecar.

## Report back with

What you ran (exact commands), the numbers (FVE sum/L/H, format rate, `lh_jaccard`, judge KL and fraction recovered,
with n), two or three representative explanations quoted verbatim, anything that looked wrong, and what you recommend
next. Keep it short; put detail in `docs/okf/`.
