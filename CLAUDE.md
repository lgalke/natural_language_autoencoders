# NLA — instructions for Claude / AI assistants

## Constraints

- **This is an open-source repo.** Only standard libs: `pathlib.Path`,
  `pyarrow`, `transformers`, `datasets`, `httpx`, `pyyaml`, `numpy`, `orjson`,
  `safetensors`, the public `anthropic` SDK, and whatever Miles/SGLang pull in.
  No private/internal dependencies.
- **Miles is upstream, not ours.** Don't edit files under `miles/` — extend via
  subclassing (`NLAFSDPActor`) and the `--*-path` function-pointer args. The
  two upstream patches we depend on (`--custom-actor-cls-path`,
  `--force-use-critic`) are documented in `docs/design.md` §2.
- Miles uses argparse; match that for CLIs in `nla/`.
- Storage and completion-provider backends are pluggable via import-path
  strings (`--storage-cls`, `--provider-cls`). The shipped implementations are
  `LocalStorage` and `AnthropicProvider`. Cloud storage / other LLM APIs are
  bring-your-own — don't hardcode bucket paths or vendor SDKs into `nla/`.

## Key invariants (do not break these)

- **Data-gen NEVER normalizes** — all parquets store raw vectors
  (`norm="none"`). `stage3_build` asserts input `norm == "none"`. Normalization
  happens at injection time (`injection_scale`) and at loss time (`mse_scale`),
  both read from the sidecar.
- **Stage-1 split is DOCUMENT-level** — partition by unique `doc_id`, all rows
  from the same doc go to the same bucket. Never split positions from one doc
  across `av_sft` / `ar_sft` / `rl`.
- **Stage-0 `_MIN_POSITION = 50`** — need enough left-context for the
  activation to be meaningful. Earlier positions decode to noise.
- **Critic extraction is suffix-anchored** — no scan, no marker token. The
  critic prompt template ends with `... <summary>`; training extracts at
  `tokens[-1]`. `critic_suffix_ids` in the sidecar is for sanity-checking only.
- **Per-doc keyed RNG** — same `(seed, doc_id)` → same sampled positions
  regardless of chunk boundaries, slice ordering, or process count. This is
  what makes multi-GPU stage-0 sharding bit-reproducible.
- **Injection hook scans for the token ID inside the hook** (`inputs[0]`), not
  from precomputed positions. Miles reorders samples twice before the forward
  pass; any precomputed index is wrong by construction.
- **`cp_size == 1` only.** Context-parallel splits each sample across ranks
  and breaks the neighbor check. NLA sequences are short; CP buys nothing.
- **Sidecar is the contract.** Token IDs, prompt templates, `injection_scale`,
  `mse_scale`, `d_model` — all loaded from `nla_meta.yaml` and asserted
  against the live tokenizer at startup. Never hardcode them.

## Debugging

If injection silently fails the actor sees the literal CJK marker char and
free-associates Chinese. Grep generated text for CJK — that's the loudest
smoke test for the entire injection path. See `docs/inference.md`
§ "Debugging: injection-failure smell" for the cause checklist.

## HRM extension (`nla/hrm/`, `hrm` branch)

Full design: `docs/hrm.md`. A verbalizer for Mimir's (HRM-Text) L/H recurrent
states, built as its **own single-process PyTorch+PEFT trainer** — it does
**not** go through Miles/SGLang (Mimir has no SGLang backend, and its
PrefixLM mask isn't representable by flash-attention). It reuses this repo's
datagen/sidecar/injection *machinery*, not the training stack. Everything
below is `nla/hrm/`-specific; the invariants above still apply to the rest
of `nla/`.

- **Two-stream sidecar, not one.** `nla/hrm/sidecar.py:HrmDatasetMeta`/
  `HrmTokenMeta` — a DIFFERENT schema from `nla.schema.NLATokenMeta` (which
  assumes one injection marker; the HRM prompt needs two, `[L]`/`[H]`). Don't
  try to load an HRM sidecar with `nla.datagen.sidecar.read_sidecar` or
  vice versa — `kind` differs (`nla_hrm_dataset` vs `nla_dataset`) and
  `deserialize_sidecar` asserts on it.
- **Shared-scalar normalization, never per-stream.**
  `nla/hrm/recon.py:shared_normalize(z_L, z_H)` divides BOTH streams (and
  their sum) by `sqrt(||z_L||^2 + ||z_H||^2)` so `normalize(z_L) +
  normalize(z_H) == normalize(z_L+z_H)` exactly. Normalizing each stream by
  its own norm breaks this — the sum-reconstruction loss term would stop
  meaning "did you predict what H reads."
- **Two injection markers, verified IN CONTEXT.**
  `nla/hrm/injection.py:find_two_injection_tokens` picks two chars that are
  single-token *in isolation*, but BPE can merge them differently once
  embedded in `<concept>{c}</concept>` — every candidate is re-verified
  against the actual template (`compute_canonical_neighbors_two`) before
  being cached, not just tokenized alone. Two separate calls to
  `nla.injection.inject_at_marked_positions` (one per marker), not one call
  with interleaved vectors.
- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding`** (dict-
  like) on this repo's transformers version, not a bare list/tensor.
  `nla/hrm/model.py:render_av_prompt` and `nla/hrm/injection.py` both index
  `["input_ids"]` explicitly — if you see "marker not found" despite the
  char being right, check this first (iterating a `BatchEncoding` yields its
  KEYS, not token ids, so a naive `for tid in ids` loop silently finds 0
  matches). `nla.schema.compute_canonical_neighbors` has the same
  bare-`ids` pattern; worth checking if this repo's pinned transformers
  version ever changes.
- **SFT is format-only, RL is where L/H differentiate.** AV-SFT/AR-SFT train
  on Claude explanations that are NOT stream-specific (two independent
  samples of the same context, randomly assigned to the L/H fields — see
  `build.py`). Don't expect SFT checkpoints to show any L-vs-H content
  difference; that's `train_rl.py`'s job, the only stage where the reward
  reads the REAL z_L vs z_H.
- **The Mimir judge is eval-only, never in the RL loop.** `judge.py`'s
  patch-back KL is the primary faithfulness metric, but it loads Mimir
  itself (an extra ~1B-param forward pass) — `train_rl.py` never imports
  `nla.hrm.mimir` at all. Run `judge.py` / `eval.py --run-judge`
  periodically/standalone instead.
- **`judge.py` needs `prompt_ids`**, not a re-tokenization of
  `context_marked`'s decoded text (lossy round-trip). Only `stage0_hrm.py`
  writes it and only `build.py --stage rl` carries it through — av_sft/
  ar_sft parquets don't have it (Mimir is never touched again after
  extraction for those stages).
- **Split AV (`nla/hrm/split_av.py`, `--split-av` in `train_rl`/`eval`).** Optional design where each stream is
  verbalized in its OWN call (the other stream's vector zeroed, one field written, prompt tagged). Same template,
  markers, adapters, AR and judge; the two fields are joined by `split_av.join_split` before the unchanged reward.
  Don't mix split and joint checkpoints: a split AV must be evaluated with `eval --split-av`.
- **PEFT's nested-adapter-directory save.** `PeftModel.save_pretrained(path,
  selected_adapters=["av"])` writes to `path/av/adapter_config.json`, not
  flat at `path/` (only an adapter literally named `"default"` saves flat).
  `nla/hrm/model.py:load_sft_adapter`/`load_rl_checkpoint` already handle
  this — don't `model.load_adapter(ckpt_dir, ...)` directly on a
  `train_*_sft.py` `--output` dir.
