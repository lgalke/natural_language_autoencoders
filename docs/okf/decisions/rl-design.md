---
type: Decision
title: Hand-written GRPO-style loop with an online AR and KL to the post-SFT AV
description: Group-normalized REINFORCE on the AV, simultaneous supervised training of the AR, k2 KL penalty to a frozen post-SFT snapshot.
tags: [rl, training]
timestamp: 2026-09-30
---

# Algorithm (per step)

1. Sample B activation rows, generate G rollouts each at temperature 1 (defaults B=8, G=8, max 300 new tokens).
2. Parse each completion into (L, H) fields. Score with the current AR and heads under no_grad: reward = -loss, or the failure reward for malformed output.
3. Advantages: reward minus the group mean, divided by the group std (clamped at 1e-4).
4. Policy step: token-mean REINFORCE weighted by the advantage, plus `kl_beta * k2` KL to `av_ref`. `av_ref` is a frozen copy of the `av` adapter taken right after SFT. Updates the `av` LoRA and both injection adapters.
5. AR step: a separate forward pass with gradients on the well-formed rollouts, `recon_loss`, updating the `ar` LoRA and heads.

# Decisions inside this

- The AR trains online on the AV's rollouts, as in the original NLA: a frozen AR would be gamed as the AV improves.
- KL target is the post-SFT AV, not the adapter-off base model (the base cannot read the injected vectors).
- The k2 estimator (0.5 * log-ratio squared per token, averaged over response tokens), following the original repo's finding that k1 has zero expected gradient as a direct loss term.
- Steps 4 and 5 use separate forward/backward passes and separate optimizers (AdamW, policy lr 1e-5, AR lr 1e-4, kl_beta 0.01); no shared graph.
- Memory: both steps run in `--micro-batch-size` chunks with gradient accumulation, and logits are computed only for response positions; see [OOM observations](/observations/pitfalls.md).

# Built-in sanity checks

- `--sanity`: real versus shuffled (rolled by one) gold vectors should give different mean reward.
- Every logged step greps completions for CJK characters (the injection-failure smell). The regex also matches ordinary Chinese, so isolated hits can be benign language drift.
- Per-step FVE when `norm_stats.json` is available.

# Not included

No PPO clipping and one optimizer step per rollout batch (on-policy). No Mimir forward pass in the loop; see [judge design](/decisions/judge-design.md).
