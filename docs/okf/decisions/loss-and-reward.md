---
type: Decision
title: Plain MSE with a sum anchor plus per-stream terms; reward is minus the loss
description: loss = w_sum*MSE(sum) + w_comp*0.5*(MSE_L + MSE_H), no whitening; malformed completions get a fixed worse-than-mean reward.
tags: [loss, reward, rl]
timestamp: 2026-09-30
---

# Decision

On shared-normalized vectors ([normalization](/decisions/shared-scalar-normalization.md)):

```
loss = w_sum * MSE(z_L_hat + z_H_hat, s) + w_comp * 0.5 * (MSE(z_L_hat, z_L) + MSE(z_H_hat, z_H))
reward = -loss          (optionally -log(loss) with --log-reward)
```

Defaults: `w_sum = 1.0`, `w_comp = 0.25`. FVE per term is reported relative to the train-set mean-predictor MSE (`norm_stats.py`).

# Why the sum anchor

We want to learn what each stream stores (fast L versus slow H memory). But if the streams partly cancel, the per-stream terms alone could push explanations towards content that H never reads. The heavily weighted sum term keeps explanations tied to what the model actually reads.

# Whitening: considered and dropped

An earlier design whitened each space on the training data. The final decision is plain MSE (as in the original NLA), with no whitening.

# Malformed completions

An unparseable completion, or one with an empty L or H field, gets `failed_reward`, derived from a fixed MSE of 2.0 scaled by `(w_sum + w_comp)`: -2.5 with the defaults. This is deliberately worse than the mean predictor.

# Known weakness

With the default weights, the failure penalty (-2.5) is orders of magnitude larger than differences between well-formed completions (about 1e-5). Early in training the gradient mostly says "be well-formed". See [RL run 1](/observations/rl-run-1.md) and [open questions](/open-questions.md) for the options (`--log-reward`, a milder penalty).
