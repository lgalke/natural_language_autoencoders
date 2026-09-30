---
type: Observation
title: Engineering pitfalls hit during the build, with causes and fixes
description: A log of failures worth knowing about when reproducing or extending the pipeline.
tags: [pitfalls, debugging, reproducibility]
timestamp: 2026-09-30
---

# Tokenizer and chat template

- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding`** (dict-like) on transformers 5.13, not a list. Iterating it yields the keys, so a search for a marker id silently finds 0 matches. `nla/hrm` indexes `["input_ids"]` explicitly. `nla.schema.compute_canonical_neighbors` in the base repo has the same bare-list pattern and was not changed.
- **Double BOS.** The Mimir template writes `<bos>` as text; encoding the rendered string with default special tokens gives `[2, 2, ...]`. Always `add_special_tokens=False`, and assert a single BOS.
- **Markers must be verified in context.** A CJK character that is one token alone can tokenize differently next to `<`/`>`. The first marker pair passed the in-isolation check but was absent from the real prompt; candidates are now verified against the template. See [two markers](/decisions/two-injection-markers.md).

# Memory

- **stage0 OOM (105 GiB allocation).** Extraction called the full causal LM, so the LM head built `[batch, seq, 262144]` logits for every position, which only hid behind short prompts until longer ones (SimpleStories, MuSR) appeared. Fix: `logits_to_keep=1`, length-sorted batching under a token budget (`--max-batch-tokens`), and skipping (not truncating) prompts over `--max-prompt-tokens`.
- **RL OOM (95 GB).** The policy step ran all 64 rollouts at once with full-vocabulary logits for every position, for both policy and reference, with gradients; the AR update did the same for about 128 texts. Fix: `--micro-batch-size` chunks with gradient accumulation (gradients match the full batch to about 1e-8, verified), response-only logits via `logits_to_keep`, and no LM-head logits in hidden-state-only passes.

# Defaults and environment

- **CPU by default.** Early versions of every script defaulted to `--device cpu`, float32, so a 1.5B model trained on CPU with 0% GPU use. Defaults are now CUDA and bf16 when available.
- **`norm_stats.py` slow.** The injection-scale step ran Qwen-1.5B in float32 on CPU with no device flag.

# Data and providers

- **Corpus builder.** MuSR's text column is `narrative`, not `context`; ProofWriter was not found; full non-streaming loads of large datasets are slow. Sources that fail are now skipped with a warning.
- **Authentication.** The default Anthropic provider cannot use an OpenRouter or UCloud key and does not read `.env`. See [explanation teacher](/decisions/explanation-teacher.md).

# Pipeline bookkeeping

- **`judge_subset.parquet` had no sidecar.** `split.py` wrote the file but not its `.nla_meta.yaml`, so `build.py` failed to read it. Fixed in `split.py`; for splits made earlier, copy the `eval_iid` sidecar and fix `dataset_id` and `row_count`.
- **`eval_iid.parquet` missing at eval time.** The early small example in the docs only built `eval_ood`; build all of `eval_iid`, `eval_ood`, `judge_subset` with `--stage rl`.
- **Never re-run `split.py` on a dataset that already has trained checkpoints**: the world shuffle is reseeded per run and buckets change, so eval rows could leak into training.
- **Generated marker cache committed by accident.** `nla/hrm/injection_token_cache_hrm.yaml` is machine-local and now git-ignored.

# PEFT

- `save_pretrained(path, selected_adapters=["av"])` writes `path/av/adapter_config.json`; only an adapter named `"default"` saves flat. Loading uses `nla/hrm/model.py:load_sft_adapter`.
- Freshly initialised LoRA adapters are an exact no-op (B = 0), so base and adapter logits agree before training; do not treat that as a wiring bug.

# Sanity-check false alarms

- `--sanity` reported real == shuffled when (a) nearly all completions were malformed (every one gets the same failure reward), or (b) the batch had 2 rows and a random permutation was the identity. The shuffle is now a roll by one. Check the malformed count before suspecting injection.
