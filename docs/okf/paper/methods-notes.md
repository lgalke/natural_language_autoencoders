---
type: Paper Notes
title: Methods notes for a research paper (draft prose, parameters, caveats)
description: Structured text for a methods section; every result is a TODO placeholder and every unverified item is flagged.
tags: [paper, methods, draft]
timestamp: 2026-09-30
---

# How to use this page

Sections 1 to 8 are written to be adapted into a methods section. Statements of fact come from the code as committed on the `hrm` branch. Items marked **TODO** are not available yet; items marked **CHECK** should be confirmed against the actual run configuration before publication. Design rationale lives in the linked decision pages. Do not cite numbers from smoke tests.

# 1. Problem and approach

Natural-language autoencoders (NLAs) map an activation vector to text and back: a verbalizer (AV) reads the vector and writes a description, and a reconstructor (AR) reads the description and predicts the vector; low reconstruction error implies that the text preserves the information in the vector. We adapt this to a hierarchical recurrent model (HRM) whose layers are applied recursively, which yields two distinct states per position, a fast low-level (L) state and a slow high-level (H) state. We ask whether verbalizations of the two states differ, and whether they are faithful to what the model subsequently uses.

Deviations from the original NLA: the verbalizer and reconstructor are a different, off-the-shelf model (Qwen2.5-1.5B-Instruct) rather than copies of the target; the reconstructor is full depth with an affine head instead of a truncated copy of the target; input and output are two-stream with field-structured text; faithfulness is additionally measured by patching reconstructions back into the target. See [model decision](/decisions/verbalizer-model.md).

# 2. Target model and activations

**Target.** DFM-Mimir-v1.5 (HRM-Text architecture, `hrm_text` in `transformers>=5.13`): hidden size d = 1536, 16 transformer blocks per stack (one L stack and one H stack with separate weights), H_cycles = 2, L_cycles = 3, vocabulary 262144, Gemma-4 tokenizer and chat template, PrefixLM attention. The forward pass is

    z_H^(0) = E(x) * s_emb ;  z_L^(0) = 0
    for h = 1..2:   for l = 1..3:  z_L <- L(z_L + z_H) ;   z_H <- H(z_H + z_L)

with the schedule L L L H L L L H, where each stack ends in an RMSNorm without a learned gain (so every stack output has norm sqrt(d) = 39.19).

**Hook.** We read the pair (z_L, z_H) immediately before the second application of H: z_L is the output of the sixth L call, z_H the output of the first H call, and their sum is exactly H's input. Captured with forward hooks on the two stacks; each extraction asserts the call counts (6 L, 2 H) and the sum identity. See [hook site](/decisions/hook-site.md).

**Prompts and positions.** Single-turn prompts rendered with the Gemma-4 chat template (thinking disabled, no extra BOS); the whole rendered prompt is one bidirectional PrefixLM block. For each prompt we sample up to 6 token positions (from token index 3 onward, keyed RNG on (seed, prompt id)) plus always the last prompt token. Activations are stored raw (float32). See [rendering](/decisions/prefixlm-rendering.md) and [corpus](/decisions/corpus-and-splits.md).

**Corpus (CHECK exact mix used).** Public sources: GSM-Symbolic, BBH (causal judgement), MuSR (murder mysteries), SimpleStories, and Danish DynaWord text, at most 2000 prompts per source; prompts over 2048 rendered tokens skipped. **TODO:** counts per source, number of activation rows per split (only known: 9324 rows in the AV-SFT bucket before explanation filtering), HRMMix if it becomes available.

**Splits.** Group-level (by prompt) partition into AV-SFT 30%, AR-SFT 30%, RL 30%, in-distribution eval 10%; the MuSR dataset is held out entirely as an out-of-distribution eval set; a random half of the eval rows forms a fixed subset for the (expensive) patch-back judge.

# 3. Verbalizer and reconstructor

**Backbone.** Qwen2.5-1.5B-Instruct (d = 1536), frozen, with two LoRA adapters (rank 32, alpha 64, all attention and MLP projections): AV and AR.

**Input.** The AV prompt contains two slots, `[L] <concept>•</concept>` and `[H] <concept>•</concept>`, each a single reserved CJK token. The embedding at each slot is replaced by `W_s z_s + b_s` (s in {L, H}), with learned affine maps initialised as a scaled identity (scale from the 75th percentile of Qwen's last-layer hidden-state norm, or 5.0 when skipped; **CHECK which**). See [markers](/decisions/two-injection-markers.md).

**Output.** `<explanation>\nL: ...\nH: ...\n</explanation>`, about 100 words (4 to 5 features) per field. Completions that do not parse, or with an empty field, receive a fixed failure reward.

**Reconstruction.** Each field is wrapped in a fixed template that ends in a summary token; the AR's final hidden state at that token goes through a per-stream affine head (`head_L` for the L field, `head_H` for the H field) producing z_L_hat and z_H_hat. The L head never sees the H field and vice versa.

# 4. Objective

With the shared scale c = sqrt(||z_L||^2 + ||z_H||^2) per position, let n(v) = v / c. Because the same c is used for z_L, z_H and s = z_L + z_H, n(z_L) + n(z_H) = n(s) exactly. The loss is

    L = w_sum * MSE(n(z_L_hat) + n(z_H_hat), n(s))  +  w_comp * 0.5 * [ MSE(n(z_L_hat), n(z_L)) + MSE(n(z_H_hat), n(z_H)) ]

with MSE the mean over the d coordinates, w_sum = 1.0, w_comp = 0.25. The sum term ties explanations to what H actually reads; the per-stream terms ask what each stream stores. We observed strong anticorrelation between the streams (**TODO:** full-corpus cos and cancellation statistics; smoke-test value only, about -0.8), so the sum term guards against rewarding content that cancels. Reward = -L (optionally -log L). See [normalization](/decisions/shared-scalar-normalization.md) and [loss](/decisions/loss-and-reward.md).

**FVE.** Reported per term as 1 - MSE / MSE_mean, where MSE_mean is the error of the constant train-set mean predictor in the same normalized space. Note that absolute MSE values are tiny (order 1e-4 to 1e-3) for any predictor because of the unit-norm scaling; only FVE is interpretable.

# 5. Training

**Explanation data.** For each activation row, two independent explanations of the prompt with the target token marked were sampled at temperature 1 from GLM-5.3 (low reasoning effort); they were randomly assigned to the L and H fields. No teacher sees the activations, so SFT data contains no stream-specific information by construction. 9191 of 9324 AV-SFT rows were retained after format filtering. See [teacher](/decisions/explanation-teacher.md), [SFT](/decisions/sft-is-format-only.md).

**SFT (format-only).** AR-SFT: next-vector regression with direction-only MSE (both prediction and target rescaled to norm sqrt(d)), one row per field. AV-SFT: next-token cross-entropy on the response only, with both injection maps trained; stop when the greedy format rate on held-out rows reaches 0.99. AdamW, lr 1e-4, **CHECK** batch sizes (defaults 16 and 8) and epochs (default 1).

**RL.** Group-normalized REINFORCE (GRPO-style, no clipping, one update per rollout batch) with a k2 KL penalty to the frozen post-SFT AV: per step 8 prompts x 8 rollouts at temperature 1, max 300 new tokens, policy lr 1e-5, KL coefficient 0.01, advantages (r - mean) / std within each group. The AR and affine heads are trained simultaneously and online on the well-formed rollouts (lr 1e-4) using the same loss, so the reward model tracks the AV. First run: 200 steps, single GPU, bf16, 76 minutes. See [RL design](/decisions/rl-design.md).

# 6. Evaluation

1. **Reconstruction FVE** (sum, L, H) on in-distribution and held-out-dataset rows, split by last-prompt-position versus other positions.
2. **Patch-back judge (primary faithfulness).** Replace the true H input (z_L + z_H) at the extraction position with the reconstructed sum and measure KL(clean || patched) of the next-token distribution at that position and at the last prompt position; report the fraction of KL recovered relative to replacing with the train-set mean. Secondary: replace z_H alone at the output of the first H call and let the second cycle run (causal test of the H component). The true vector must reproduce clean logits (KL < 1e-4; asserted). See [judge](/decisions/judge-design.md).
3. **Text-only baseline.** Same architecture and loss, SFT only, predicting z_L and z_H directly from the marked prompt text without seeing any activation. Explanations are informative only insofar as they beat it.
4. **Cross-reconstruction matrix.** Freeze the AV; train fresh, budget-matched probes to map L field, H field, or both concatenated onto z_L and z_H; a diagonal (L to z_L, H to z_H) clearly above the off-diagonal indicates stream-specific content.
5. **Field monitoring.** Token-overlap (Jaccard) between the two fields, field lengths, and malformed rate, to detect duplicated or collapsed fields. Duplication would itself be a reportable result.

**Results: TODO.** Suggested table: rows = {post-SFT, post-RL, text-only baseline}, columns = {FVE sum, FVE L, FVE H, judge KL, fraction of KL recovered} for in-distribution and held-out sets. Suggested figure: 2 x 3 cross-reconstruction heatmap; histogram of per-position cancellation and Jaccard overlap.

# 7. Implementation details worth reporting

- Single GPU, PyTorch + PEFT, HF `generate` with `inputs_embeds` for rollouts; Mimir is used only for extraction and the judge and is never loaded during training.
- bf16 base weights; LoRA adapters and heads believed to be kept in float32 (PEFT default; **CHECK**); RL micro-batched (8 rollouts per pass) with gradient accumulation.
- Prompts right-padded in batches; we verified that hidden states at the hook are identical (difference below 4e-6) with and without padding, and that disabling the PrefixLM mask changes them substantially ([PrefixLM](/decisions/prefixlm-rendering.md)).
- Code: `nla/hrm/` on the `hrm` branch; commands in `docs/hrm.md`.

# 8. Limitations and threats to validity

- The verbalizer is a different model from the target: explanations describe what Qwen can read out of an injected vector after training, not a native description. Faithfulness therefore rests on the patch-back judge, not on the text itself.
- SFT targets come from an LLM that never saw the activations; the text distribution reflects prompt context, and the model could in principle reconstruct from context alone. The text-only baseline and the judge are the controls for this.
- Reward scale: the malformed-completion penalty is much larger than typical reconstruction differences ([RL run 1](/observations/rl-run-1.md)); conclusions about what RL learned require the FVE evaluation.
- One seed, one checkpoint of the target, one 200-step run so far; no variance estimates. **TODO.**
- PrefixLM makes activations at a position depend on later tokens, so they are not next-token predictors in the usual sense; explanations describe the state in the context of the whole prompt.
- The data collection is ad hoc (uniform position sampling that includes the shared chat-template tail, prompt positions only, no response positions, uncontrolled source mix; see [data collection v2](/plans/data-collection-v2.md)); statements about L/H dynamics during generation are out of scope for v1 data.
- The corpus is a public stand-in for the intended reasoning mixture; MuSR as out-of-distribution set is a strong shift (long narratives).
- Open items are tracked in [open questions](/open-questions.md).

# 9. Ready-to-adapt paragraph (draft)

"We train a natural-language autoencoder over the two recurrent states of a hierarchical reasoning model. For every sampled prompt position we extract the low-level state z_L and high-level state z_H immediately before the second high-level update, where their sum is exactly the input of that update. A frozen Qwen2.5-1.5B-Instruct with a verbalizer adapter receives both vectors through learned per-stream affine maps injected at two reserved token slots and generates a field-structured explanation; a second adapter with per-stream linear heads reconstructs each state from its own field. Both adapters are warmed up on stream-neutral LLM-written explanations (format only) and then trained with a GRPO-style objective whose reward is the negative reconstruction loss computed in a space normalized by one scalar shared by z_L, z_H and their sum, which keeps the sum exactly additive. We evaluate reconstruction (fraction of variance explained against a mean predictor), faithfulness by patching the reconstructions back into the target model and measuring the KL divergence of its output distribution, and stream specificity by a cross-reconstruction matrix, against a text-only baseline. [TODO: results]"

# Citations

[1] Fraser-Taliente et al., "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations", Transformer Circuits, 2026. [link](https://transformer-circuits.pub/2026/nla/index.html)
[2] Mimir technical report. [link](https://arxiv.org/abs/2608.13517)
[3] HRM-Text. [link](https://sapientinc.github.io/HRM-Text/assets/HRM_Text.pdf)
[4] SimpleStories dataset. [link](https://huggingface.co/datasets/SimpleStories/SimpleStories)
