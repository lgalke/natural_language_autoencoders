---
okf_version: "0.1"
---

# Headline (hypothesis, strong evidence, replication pending)
* [H holds WHAT, L holds WHERE](/observations/h-what-l-where.md) - probes and the split verbalizer agree: z_H reads the token, z_L the position; each verbalizer call is at chance on the other call's fact

# Overview
* [HRM verbalizer](/overview.md) - goal, method in one paragraph, current status

# Decisions
* [Standalone trainer, not Miles](/decisions/own-trainer-not-miles.md) - why nla/hrm has its own PyTorch+PEFT loop
* [Verbalizer model](/decisions/verbalizer-model.md) - Qwen2.5-1.5B, LoRA adapters, full-depth AR, affine maps
* [Hook site](/decisions/hook-site.md) - z_L and z_H just before the second H application
* [Prompt rendering and PrefixLM](/decisions/prefixlm-rendering.md) - chat template, bidirectional block, padding verified
* [Shared-scalar normalization](/decisions/shared-scalar-normalization.md) - keeps z_L + z_H additive
* [Loss and reward](/decisions/loss-and-reward.md) - sum anchor plus per-stream MSE, failure reward
* [Two injection markers](/decisions/two-injection-markers.md) - [L]/[H] slots, verified in context
* [SFT is format-only](/decisions/sft-is-format-only.md) - why no stream-specific teacher exists
* [RL design](/decisions/rl-design.md) - GRPO-style loop, online AR, KL to post-SFT
* [Judge design](/decisions/judge-design.md) - patch-back KL, eval-only
* [Token-prefix targets](/decisions/token-prefix-targets.md) - optional supervised token target in the SFT data
* [Corpus and splits](/decisions/corpus-and-splits.md) - sources, positions, OOD holdout
* [Explanation teacher](/decisions/explanation-teacher.md) - GLM-5.3 via an OpenAI-compatible provider

# Plans
* [Data collection v2](/plans/data-collection-v2.md) - proposed changes to extraction, sampling and mix (not implemented)
* [Last night before the talk](/plans/last-night-replication.md) - replicate the split pipeline, two cheap diagnostics, commands and decision rules

# Observations
* [H what, L where (headline)](/observations/h-what-l-where.md) - the division of labour between the streams, evidence and open checks
* [Cancellation of L and H](/observations/cancellation.md) - cos about -0.8 on a small sample
* [RL run 1](/observations/rl-run-1.md) - what the 200-step log does and does not show
* [RL run 2 (log-reward)](/observations/rl-run-2-log-reward.md) - FVE positive, KL large, samples look unfaithful (preliminary)
* [Token probe](/observations/token-probe.md) - marked token is 99% linearly decodable from z_H, 81% from z_L
* [Engineering pitfalls](/observations/pitfalls.md) - failures, causes and fixes

# Experiments
* [Experiment log](/experiments.md) - every training run and evaluation with commands, numbers (with n) and status

# Status
* [Open questions](/open-questions.md) - missing measurements and untested code

# Paper
* [Methods notes](/paper/methods-notes.md) - reusable prose, parameters and caveats for a methods section
* [Talk outline](/paper/talk-outline.md) - slide-by-slide structure with sourced numbers, say / do not say, pending items
* [One slide: H what, L where](/paper/slide-h-what-l-where.md) - the probe table, the verbalizer table and three examples, copy-ready

# History
* [Log](/log.md) - dated record of changes to this bundle and the code
