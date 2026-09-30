---
type: Decision
title: Explanation teacher is GLM-5.3 (UCloud) at low reasoning effort via an OpenAI-compatible provider
description: SFT targets come from a hosted GLM instead of Claude; stage2 was generalised to multiple samples per row.
tags: [data, sft, providers]
timestamp: 2026-09-30
---

# Decision

- SFT explanations were generated with `zai-org/GLM-5.3` hosted on UCloud (`https://ai.cloud.sdu.dk/v1`), through `nla.hrm.openai_compat.UCloudGLMProvider`: OpenAI-compatible chat/completions over httpx, low reasoning effort, 2000 max tokens, key from `UCLOUD_API_KEY` (environment or a git-ignored `.env`).
- The default `AnthropicProvider` was not usable: it reads only `ANTHROPIC_API_KEY` and talks to Anthropic, so an OpenRouter or UCloud key fails authentication.
- `nla/datagen/stage2_api_explain.py` gained `--text-column` and `--samples-per-row` (both backward compatible) and its chunk loop became a reusable `explain_table()`.

# Explainer prompt

`nla/hrm/explain.py:DEFAULT_HRM_INSTRUCTION`: describe the ⟦marked⟧ token's state in the context of the whole prompt (stating that the model reads bidirectionally), 4 to 5 features, about 100 words, closing `</analysis>` tag required.

# Observed yield

9324 rows in, 9191 out for `av_sft`; 133 (1.4%) dropped because fewer than 2 of the 2 completions survived extraction. Likely causes (not investigated): output truncated before the closing tag, or empty replies. Reasoning models spend `max_tokens` on hidden reasoning first, which is why the default effort is low.

# Caveat for the paper

The SFT teacher is not the original method's Claude model; explanation style may differ. Explanations were only spot-checked, if at all; see [open questions](/open-questions.md).
