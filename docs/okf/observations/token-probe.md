---
type: Observation
title: "The marked token is almost fully linearly decodable from the vectors (probe ceiling)"
description: Linear probe on z_L, z_H, sum and concat predicting the token at the extraction position, on held-out prompts.
tags: [probe, information, token-identity]
timestamp: 2026-10-02
---

# Measurement

`nla.hrm.probe_check` on the real `base.parquet` (21585 rows; top-200 tokens cover 66% of rows; train 17276, test 4309; split grouped by prompt `world`): multinomial linear probe predicting the identity of the token at the extraction position.

| features | test accuracy |
|---|---|
| majority class | 0.285 (chance 0.005) |
| z_L | 0.808 |
| z_H | 0.991 |
| sum | 0.959 |
| concat | 0.979 |

# Reading

- Token identity is almost perfectly linearly readable from z_H and well from z_L, so a verbalizer failing to quote the marked token (0% correct in [RL run 2](/observations/rl-run-2-log-reward.md)) is a training problem, not an information limit.
- Token identity is much more recoverable from the slow state than from the fast one (0.99 versus 0.81): a first concrete L-versus-H difference, from a probe and not from verbalizations.
- Caveats: top-200 tokens only; no random-vector or shuffled-label control was run; a linear probe on 1536 dimensions with 200 classes can find structure that a verbalizer would need long training to read; the last prompt position (always the same template token) inflates the majority baseline.

# Consequence

See [token-prefix targets](/decisions/token-prefix-targets.md).
