# HRM verbalizer: a two-stream NLA for Mimir's L/H recurrence

**Goal:** verbalize the internal states of [DFM-Mimir](https://huggingface.co/danish-foundation-models/DFM-Mimir-v1.5)
(`hrm_text`, an HRM-Text model — see the sibling repos `hrm-text-rlvr` and
`hrm-interp`), specifically to find out whether its fast **L** stream and
slow **H** stream store different things. Built as `nla/hrm/`, a package on
the `hrm` branch, reusing the rest of this repo's datagen/sidecar/injection
machinery but with its own single-process trainer instead of Miles/SGLang.

## Why not Miles

The rest of this repo assumes: one activation vector per row, a verbalizer
whose `hidden_size` matches the target's, an AR that's "the base model
truncated at layer K", and rollout served through SGLang. None of that fits
here:

- Mimir has **no SGLang or vLLM backend** — `hrm_text` isn't a supported
  architecture there.
- Its PrefixLM attention mask needs a 4-D mask overlay that flash-attention
  can't represent (`HrmTextPreTrainedModel._check_and_adjust_attn_implementation`),
  which rules out the packed-sequence training path Miles is built around.
- The hook site (below) is **two** vectors per position, not one, and "the
  base model truncated at layer K" has no meaning for a model whose 16
  "layers" are the *same* weights applied up to 8 times recursively.
- The verbalizer is a **different model** (Qwen2.5-1.5B-Instruct) from the
  target (Mimir), whereas the original NLA verbalizes a model's own
  activations back through itself.

So `nla/hrm/` is a from-scratch, single-process PyTorch + PEFT trainer. It
reuses: `nla/datagen/stage1_split.py`'s doc-level-split idea (re-implemented
per-`world` in `split.py`), `nla/datagen/injection_tokens.py` /
`nla/schema.py`'s injection-token machinery, `nla/injection.py`'s
`inject_at_marked_positions` (unchanged), `nla/datagen/stage2_api_explain.py`
(extended, not forked — see below), and the sidecar-YAML pattern (its own
schema, `nla/hrm/sidecar.py`, since the row shape differs).

## The hook site

Mimir-v1.5: `d_model=1536`, `H_cycles=2`, `L_cycles=3` — the recurrence is

```
z_L = 0                      # z_L_init, frozen
z_H = embed(tokens) * scale
for h in 1..H_cycles:
    for l in 1..L_cycles:
        z_L = L_module(z_L + z_H)      # "L_out@h.l"
    z_H = H_module(z_H + z_L)          # "H_out@h"
```

i.e. the schedule is **L L L H L L L H**. We hook **just before the second H
application**:

```
z_L = L_out@2.3     (the 6th L call — last L application of H-cycle 2)
z_H = H_out@1       (the 1st H application)
s   = H_in@2 = z_L + z_H     (what H's 2nd application actually reads)
```

`nla/hrm/mimir.py:HrmStreamCapture` implements this with forward hooks on
`model.model.L_module` / `H_module` (each a full `HrmTextStack`, so this
matches HRM-Interp's `CycleStateCollector` convention of hooking the stack,
not individual layers) and asserts `z_L + z_H == H_in@2` exactly on every
forward pass — free correctness check, since that's literally how
`HrmTextModel.forward` computes it.

Both `z_L` and `z_H` have norm ≈ `sqrt(1536) ≈ 39.2` (each stack ends in an
unweighted RMSNorm). On real Mimir forward passes we've measured
`cos(z_L, z_H) ≈ -0.8` — **strong cancellation**: the two streams write in
substantially opposing directions before H reads their sum. This directly
motivated the reconstruction-loss weighting below (see `norm_stats.py` /
`diagnostics.py`).

## Verbalizer / reconstructor

One frozen **Qwen2.5-1.5B-Instruct**, two PEFT LoRA adapters (`av`, `ar`;
`train_rl.py` adds a third, frozen `av_ref` snapshot for the KL penalty).

**Deviation from the original NLA**: AV and AR are a *different* model from
the target (Qwen, not a copy of Mimir), and the AR is the **full-depth**
verbalizer plus an affine head — not truncated to a layer K, which has no
well-defined meaning for a recurrent target. The two-stream input/field
structure is entirely this extension's own.

- **Injection**: two learnable per-stream affine maps `inj_L`, `inj_H`
  (`nla/hrm/model.py:InjectionAdapter`, `d_mimir -> d_verbalizer`), near-
  identity init, scale initialized **large** — near the verbalizer's own
  ambient residual-stream scale (the 75th-percentile hidden-state norm,
  `norm_stats.py:fit_injection_scale_p75`), not its token-embedding scale.
  Two DISTINCT single-token CJK marker chars (`[L]`/`[H]`, picked by
  `nla/hrm/injection.py:find_two_injection_tokens`, verified in the actual
  template context — not just in isolation, since BPE merges at the
  `<concept>{c}</concept>` boundary can shift a char that's single-token
  alone into a multi-token or wrong-ID sequence in context) — TWO calls to
  `nla.injection.inject_at_marked_positions` (unchanged), one per stream.
- **Reconstruction**: `head_L`, `head_H` (`ReconHeads`), affine maps from the
  AR's final hidden state (`d_verbalizer -> d_mimir`), same near-identity
  init pattern, small scale.
- **AV output format**: `<explanation>\nL: ...\nH: ...\n</explanation>`
  (`nla/hrm/recon.py:parse_fields`). The L head only ever sees the L field's
  text, the H head only the H field's.

## Normalization (shared-scalar, not per-stream)

`nla/hrm/recon.py:shared_normalize(z_L, z_H)` divides **both** streams, and
`s = z_L + z_H`, by the **same** per-token scalar
`norm = sqrt(||z_L||^2 + ||z_H||^2)`:

```
normalize(z_L) + normalize(z_H) == normalize(z_L + z_H)   # exact
```

**Never normalize each stream by its own norm** — that breaks additivity, and
additivity is what makes the sum term below a meaningful "did you predict
what H reads" signal rather than two independent, incomparable directions.

## Loss / reward

Plain MSE (no whitening), matching the original NLA:

```
loss = w_sum * MSE(ẑ_L+ẑ_H, s) + w_comp * 0.5*(MSE(ẑ_L,z_L) + MSE(ẑ_H,z_H))
reward = -loss   (or -log(loss), NLA_LOG_MSE_REWARD-style — --log-reward)
```

on shared-normalized vectors. Defaults `w_sum=1.0, w_comp=0.25` — revisit
after `diagnostics.py`'s cancellation numbers on your corpus (strong
cancellation argues for weighting the sum term even more heavily, since the
individual-stream terms alone could reward "plausible-looking but
functionally irrelevant" content that the model doesn't actually act on).
FVE is reported per term relative to `norm_stats.py`'s train-set
mean-predictor baseline. A malformed completion (unparseable, or an empty
field) gets `recon.failed_reward` — strictly worse than the mean predictor.

## SFT is format-only

There is no stream-specific teacher: Claude (`explain.py`, `--samples-per-row
2`) writes TWO independently-sampled ~100-word, 4-5-feature explanations of
the **same** context, randomly assigned to the L/H fields
(`build.py:wrap_lh_explanation`). AV-SFT teaches the format and that
generation should condition on the injected vectors at all — nothing about
what makes L different from H. AR-SFT similarly pairs `api_explanation_0`
with the L head / `z_L` and `api_explanation_1` with the H head / `z_H`,
arbitrarily but fixed. **Differentiation between the streams is entirely
RL's job** (`train_rl.py`) — it's the only stage where the reward can
actually tell L-shaped content from H-shaped content, because only there
does the AR score against the REAL z_L/z_H.

## RL

A self-written GRPO-style loop (`train_rl.py`), not Miles/TRL:

1. Sample B rows × G rollouts, generate at temperature 1.
2. Score with the CURRENT `ar`+heads (no_grad) — reward = `-loss`/`-log(loss)`.
3. Group-normalize advantages within each row's G rollouts.
4. Policy step: token-mean REINFORCE·advantage + β·k2-KL toward the frozen
   post-SFT `av_ref` snapshot. Updates `av` LoRA + `inj_L`/`inj_H`.
5. AR step: a SEPARATE (WITH grad) AR forward + `recon_loss` on this batch's
   well-formed rollouts → supervised update of `ar` LoRA + heads — **online**,
   same rationale as the original NLA's simultaneous AV/AR training: a frozen
   critic would get gamed as the AV improves.
6. Sanity checks (`--sanity`): real vs. shuffled-across-batch vectors should
   give different mean reward (the loudest end-to-end check that injection
   is wired correctly); every batch is grepped for literal CJK marker chars
   leaking into generated text (the same injection-failure smell the rest of
   NLA relies on — if injection silently fails, the model sees `㈎`/`㈏` as
   literal text).

The heavier Mimir patch-back judge (below) is **eval-only** — never in this
loop; it's `judge.py` / `eval.py --run-judge`, run periodically/standalone.

## Corpus

Mixed reasoning + instruction prompts (`build_prompt_corpus.py`): the public
HRMMix sources (GSM-Symbolic, ProofWriter, BBH, MuSR) plus DA/EN instruction
prompts, with HRMMix's full local corpus swappable in via `--local-jsonl`
once available. `split.py` holds out one WHOLE source dataset entirely
(`--holdout-dataset`, default `musr`) as `eval_ood`, on top of a random
`world`-grouped split for `av_sft`/`ar_sft`/`rl`/`eval_iid`.

Prompts are rendered with the Gemma-4 chat template, `enable_thinking=False`,
**`add_special_tokens=False`** (the template already renders `<bos>` as
text — see `hrm-interp/prompt_templates.md`'s "Double BOS on re-encode"), and
the WHOLE rendered prompt is `token_type_ids=1` (bidirectional) — Mimir was
pretrained only on (instruction, response) pairs with the instruction
attended to bidirectionally end-to-end, so there's no narrower "just the user
turn" bidirectional span to carve out.

## The judge (`judge.py`)

The **primary faithfulness metric**: patch `H_in@2` at the extraction
position with `ẑ_L+ẑ_H` and measure `KL(clean‖patched)` — at the patched
position, at the last prompt position, and (with `--mean-s-json`) as a
"fraction of KL recovered" against a mean-ablation baseline. A **secondary**
patch replaces `H_out@1` with `ẑ_H` alone and lets cycle 2 run normally — z_H
persists through cycle 2 (re-added at every L step), so this is the causal
test of the H component specifically; the primary patch, being on the sum,
is blind to the L/H split. Both patches assert the TRUE vector reproduces
clean logits exactly (`KL < 1e-4`) as a correctness check on the patch
mechanism itself before trusting anything else.

`judge.py` re-renders each row's EXACT original Mimir prompt from the
`prompt_ids` column (not a re-tokenization of `context_marked`'s decoded
text — lossy) and re-extracts, so it also catches renderer/hook drift
between extraction time and judge time.

## Files

```
nla/hrm/
  mimir.py                  Mimir load/render/HrmStreamCapture — shared extraction+judge contract
  injection.py               two-marker injection-token selection (verbalizer tokenizer)
  sidecar.py                 HrmDatasetMeta/HrmTokenMeta (two-stream sidecar schema)
  build_prompt_corpus.py     HF + local-jsonl -> {prompt,dataset,world,doc_id} JSONL
  stage0_hrm.py               corpus -> base.parquet (z_L, z_H, prompt_ids, ...)
  diagnostics.py              cancellation / cos(z_L,z_H) stats
  split.py                    base.parquet -> av_sft/ar_sft/rl/eval_iid/eval_ood/judge_subset
  norm_stats.py                mean-predictor MSE baselines + injection-scale-p75, TRAIN buckets only
  explain.py                  Claude explanations (wraps stage2_api_explain.explain_table)
  build.py                    -> final av_sft/ar_sft/rl training parquets
  model.py                    load_verbalizer, InjectionAdapter, ReconHeads, build_inputs_embeds
  recon.py                    shared_normalize, recon_loss, parse_fields, reward
  train_ar_sft.py            AR-SFT (format-only warm-up)
  train_av_sft.py            AV-SFT (format-only warm-up)
  train_rl.py                  the GRPO-style loop
  train_context_baseline.py   text-only baseline (no injected vector) — the bar the AV must beat
  judge.py                    Mimir patch-back KL (primary faithfulness metric)
  eval.py                     FVE / format-rate / field-monitoring report (+ optional judge)
  eval_cross.py                cross-reconstruction FVE matrix (L/H/joint x z_L/z_H)
  testing.py                  FakeCompletionProvider — no-network smoke-test provider
```

## Commands (small end-to-end example)

```bash
python -m nla.hrm.build_prompt_corpus --output corpus.jsonl
python -m nla.hrm.stage0_hrm --corpus corpus.jsonl --output base.parquet --device cuda
python -m nla.hrm.diagnostics --input base.parquet
python -m nla.hrm.split --base base.parquet --output-dir splits/
python -m nla.hrm.norm_stats --train-parquet splits/av_sft.parquet --train-parquet splits/ar_sft.parquet \
    --train-parquet splits/rl.parquet --output norm_stats.json
python -m nla.hrm.explain --input splits/av_sft.parquet --output splits/av_sft_explained.parquet
python -m nla.hrm.explain --input splits/ar_sft.parquet --output splits/ar_sft_explained.parquet
python -m nla.hrm.build --input splits/ar_sft_explained.parquet --stage ar_sft --output ar_sft.parquet
python -m nla.hrm.build --input splits/av_sft_explained.parquet --stage av_sft --output av_sft.parquet
python -m nla.hrm.build --input splits/rl.parquet --stage rl --output rl.parquet
for s in eval_iid eval_ood judge_subset; do
  python -m nla.hrm.build --input splits/$s.parquet --stage rl --output $s.parquet
done

python -m nla.hrm.train_ar_sft --train-parquet ar_sft.parquet --output ckpt/ar_sft
python -m nla.hrm.train_av_sft --train-parquet av_sft.parquet --norm-stats-json norm_stats.json --output ckpt/av_sft
python -m nla.hrm.train_rl --rl-parquet rl.parquet --av-sft-ckpt ckpt/av_sft --ar-sft-ckpt ckpt/ar_sft \
    --sanity --output ckpt/rl

python -m nla.hrm.judge --eval-parquet eval_ood.parquet --reconstructions recon.json --output judge.json
python -m nla.hrm.eval --eval-parquet iid=eval_iid.parquet ood=eval_ood.parquet --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final \
    --run-judge --output eval_report.json
```

## Full run, step by step

Run everything from the repo root on a GPU machine with `pip install -e ".[hrm]"`
(needs `transformers>=5.13`). Scripts default to CUDA + bf16 when available.
Work in one directory, e.g. `RUN=runs/v1; mkdir -p $RUN && cd $RUN` (paths below
are relative to it). `configs/hrm/datagen.sh` chains steps 1-8 if you prefer.

**0. Explanation provider + key.** Step 6 calls an LLM. Options
(`--provider-cls`): `nla.datagen.providers.AnthropicProvider` (needs
`ANTHROPIC_API_KEY`), `nla.hrm.openai_compat.UCloudGLMProvider` (GLM-5.3, low
reasoning effort; key in `UCLOUD_API_KEY`, exported or in `./.env` in the
directory you run from), or any OpenAI-compatible endpoint via
`nla.hrm.openai_compat.OpenAICompatProvider` + `--provider-kwargs`.
`.env` is git-ignored. Try step 6 on a slice first and check the `DROPPED` count.

**1. Prompt corpus.** Mixed reasoning + instruction prompts. Sources that fail to
load are skipped with a warning (`--strict` to fail instead); add `--local-jsonl`
for HRMMix once you have it. Check the per-source counts it prints.
```bash
python -m nla.hrm.build_prompt_corpus --max-per-source 2000 --output corpus.jsonl
```

**2. Extract Mimir's L/H states** (needs the GPU; Mimir is loaded here only).
Prompts are length-sorted and batched by token budget; prompts over
`--max-prompt-tokens` are skipped, not truncated.
```bash
python -m nla.hrm.stage0_hrm --corpus corpus.jsonl --output base.parquet \
    --positions-per-prompt 6 --max-batch-tokens 16384
```

**3. Diagnostics.** Reports cancellation and cos(z_L, z_H). Strong cancellation
argues for keeping `--w-sum` high in RL.
```bash
python -m nla.hrm.diagnostics --input base.parquet
```

**4. Split.** Document/`world`-level buckets plus a whole held-out source
(`--holdout-dataset`, default `musr`) as `eval_ood`; also writes `judge_subset.parquet`.
```bash
python -m nla.hrm.split --base base.parquet --output-dir splits/
```

**5. Norm stats** (train buckets only): mean-predictor baselines for FVE and the
injection-scale init. Use `--skip-injection-scale` to fall back to 5.0.
```bash
python -m nla.hrm.norm_stats --train-parquet splits/av_sft.parquet \
    --train-parquet splits/ar_sft.parquet --train-parquet splits/rl.parquet \
    --output norm_stats.json
```

**6. Explanations** (two independent explanations per row, for the L/H SFT fields).
```bash
P="--provider-cls nla.hrm.openai_compat.UCloudGLMProvider"
python -m nla.hrm.explain --input splits/av_sft.parquet --output splits/av_sft_explained.parquet $P
python -m nla.hrm.explain --input splits/ar_sft.parquet --output splits/ar_sft_explained.parquet $P
```
Skim ~10 explanation pairs before continuing: ~100 words, 4-5 features, and the
two per row should differ.

**7. Build training/eval parquets.** `--stage rl` is used for every split that has
no explanations (rl, eval_iid, eval_ood, judge_subset).
```bash
python -m nla.hrm.build --input splits/ar_sft_explained.parquet --stage ar_sft --output ar_sft.parquet
python -m nla.hrm.build --input splits/av_sft_explained.parquet --stage av_sft --output av_sft.parquet
for s in rl eval_iid eval_ood judge_subset; do
  python -m nla.hrm.build --input splits/$s.parquet --stage rl --output $s.parquet
done
```

**8. SFT warm-ups** (format only: they teach the `L:/H:` format, not L-vs-H content).
AV-SFT stops early once `--target-format-rate` (0.99) is reached on eval.
```bash
python -m nla.hrm.train_ar_sft --train-parquet ar_sft.parquet --output ckpt/ar_sft
python -m nla.hrm.train_av_sft --train-parquet av_sft.parquet \
    --norm-stats-json norm_stats.json --output ckpt/av_sft
```
Check the AV-SFT `[eval] format_rate` line reached the target before moving on.

**9. RL** (where L/H differentiation can actually emerge). `--sanity` runs once
before training: real vs. shuffled vectors should score differently (equal scores
with many malformed completions are not necessarily a bug, see the `malformed=`
count in the step logs). Watch `malformed`, `cjk_leak` and `kl` in the step lines.
```bash
python -m nla.hrm.train_rl --rl-parquet rl.parquet --eval-parquet eval_iid.parquet \
    --av-sft-ckpt ckpt/av_sft --ar-sft-ckpt ckpt/ar_sft \
    --batch-size 32 --group-size 8 --max-new-tokens 300 --steps 500 \
    --sanity --output ckpt/rl
```
`--micro-batch-size` (default 8) is the memory knob: rollouts per forward/backward pass
(results are identical, it only changes peak memory). If you still hit OOM, lower it, and/or
lower `--batch-size`/`--group-size`. `configs/hrm/rl.sh` wraps these with the same defaults.

**10. Evaluate** (report + Mimir patch-back judge; loads Mimir, so run it on the GPU).
Pass the same checkpoint dir for AV and AR when evaluating an RL checkpoint.
```bash
python -m nla.hrm.eval --eval-parquet iid=eval_iid.parquet ood=eval_ood.parquet \
    --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final \
    --norm-stats-json norm_stats.json --run-judge --output eval_report.json
```
Run it on the post-SFT checkpoints too (`--av-ckpt ckpt/av_sft --ar-ckpt ckpt/ar_sft`)
to see what RL added.

**Inspecting outputs.** RL appends a few rollouts every `--samples-every` steps (default 10) to
`ckpt/rl/samples.jsonl`, and logs `fve_sum/fve_L/fve_H` per step when `norm_stats.json` sits next to
`rl.parquet` (or via `--norm-stats-json`). For a whole split, add `--dump-samples samples.jsonl` to step 10
(one line per row: context, completion, L/H fields, per-row MSE/FVE). To run on a NEW prompt:
```bash
python -m nla.hrm.infer --prompt "Your text here" --positions -1 \
    --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final --sidecar-from rl.parquet \
    --norm-stats-json norm_stats.json --judge
```
`--positions` are token indices into the rendered prompt (negative = from the end); `--show-tokens`
prints them. `--sidecar-from` is any built parquet (it supplies the injection markers/templates).

**11. Baseline and cross-reconstruction.** The text-only baseline is what the AV's
explanations must beat; the matrix tests whether the L/H fields carry stream-specific content.
```bash
python -m nla.hrm.train_context_baseline --train-parquet rl.parquet --output ckpt/baseline
python -m nla.hrm.eval_cross --probe-train-parquet rl.parquet --probe-test-parquet eval_ood.parquet \
    --av-ckpt ckpt/rl/final --ar-ckpt ckpt/rl/final --output cross.json
```
`train_context_baseline.py` and `eval_cross.py` have not been run end-to-end yet;
expect to debug them on first use.

See `CLAUDE.md`'s "HRM extension" section for the load-bearing invariants.

Decisions, observations, open questions and notes for a paper's methods section are in the
[OKF](https://okf.md) knowledge bundle at `docs/okf/` (start at `docs/okf/index.md`).
