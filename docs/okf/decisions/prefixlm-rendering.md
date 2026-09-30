---
type: Decision
title: Render prompts with the Gemma-4 chat template and treat the whole prompt as one bidirectional block
description: add_special_tokens=False, enable_thinking=False, token_type_ids=1 on every real prompt token, right-padded batches.
tags: [mimir, prefixlm, extraction]
timestamp: 2026-09-30
---

# Decision

- Single-turn prompts rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`.
- Tokenized with `add_special_tokens=False`: the template already emits `<bos>` as text, and encoding with the default would double it (`[2, 2, ...]`). Every prompt is asserted to contain exactly one BOS, at position 0.
- `token_type_ids` equals the attention mask: every real prompt token is bidirectional (1), padding is 0.
- Batches are right-padded; positions are sampled only inside the real length.

# Why the whole prompt is one block

Mimir was pretrained only on (instruction, response) pairs with the instruction attended to bidirectionally end to end. There is no narrower "user turn only" bidirectional span. The interpretability and RLVR sibling repos use the same convention. Without the prefix mask accuracy collapses (ARC probe 0.71 bidirectional versus 0.12 causal, from the sibling repos).

# Consequence for the method

A prompt position's activation can depend on tokens after it. The explainer prompt therefore says so, and describes the ⟦marked⟧ token in the context of the whole prompt, not as a next-token prediction. See [explanation teacher](/decisions/explanation-teacher.md).

# Verified: batching with padding is safe

Hidden states at the hook for prompts run in a padded batch equal those of the same prompts run alone: max absolute difference 0 to 3.8e-6 (CPU, float32, SDPA; prompts of 16, 34 and 10 tokens in one batch). As a sensitivity control, disabling the prefix mask (all `token_type_ids` = 0) changes `z_L` by about 24, so the test would detect a mask problem. Not yet checked: bf16 on GPU, where small differences are expected.
