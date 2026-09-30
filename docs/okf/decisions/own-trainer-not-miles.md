---
type: Decision
title: Build a standalone PyTorch+PEFT trainer instead of using Miles/SGLang
description: nla/hrm/ is a single-process trainer; only datagen, sidecar and injection machinery are reused from the rest of the repo.
tags: [architecture, infrastructure]
timestamp: 2026-09-30
---

# Decision

The HRM verbalizer lives in `nla/hrm/` and trains with plain PyTorch + PEFT in one process. It does not go through Miles (Ray/FSDP) or SGLang.

# Why

- Mimir (`hrm_text`) has no SGLang or vLLM backend, so the RL rollout path used by the rest of this repo does not exist for it.
- Mimir's PrefixLM mask is applied as a 4-D overlay that flash-attention cannot represent, which rules out packed-sequence training. See [PrefixLM mask](/decisions/prefixlm-rendering.md).
- The hook site yields two vectors per position; the existing NLA assumes one vector per row.
- The existing AR is "the base model truncated at layer K". This has no meaning for a recurrent model whose 16 blocks are reused up to 8 times.
- Mimir needs `transformers>=5.13`; pinning that for all of `nla/` could conflict with Miles/SGLang's pinned versions.

# What is reused

`nla.injection.inject_at_marked_positions` (unchanged), the sidecar-YAML pattern, `nla/datagen/stage2_api_explain.py` (extended, see [explanation teacher](/decisions/explanation-teacher.md)), `nla.schema.normalize_activation`, `nla.datagen.injection_tokens.compute_critic_suffix_ids`.

# Alternatives considered

- Extending Miles with an HRM actor: rejected, it would need new SGLang support for a model we do not control.
- A larger Qwen with Miles: still needs two-vector injection and LoRA-on-FSDP, and keeps Mimir in a separate environment.

# Consequences

- No multi-GPU training path. Single GPU only; torchrun DDP was left for later.
- Rollouts use HF `generate` with `inputs_embeds`, which is much slower than SGLang. Measured ~23 s per RL step for 64 rollouts of up to 300 tokens; see [RL run 1](/observations/rl-run-1.md).
