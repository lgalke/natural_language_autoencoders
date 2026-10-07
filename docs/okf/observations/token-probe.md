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

# Update 2026-10-07: offset profile (is H easier to decode than L, and what do they hold?)

`probe_check --non-last --offsets ... --mlp-hidden 512` on `base.parquet` (top-200 tokens per offset covering about 60% of rows, about 3200 test rows per offset, interval about +-0.017; split by prompt group; user paste, the output for offset -3 was cut off). Token at position + K, linear probe / MLP(512) test accuracy (majority baseline about 0.08, chance 0.005):

| offset | z_L | z_H | sum | concat |
|---|---|---|---|---|
| -2 | 0.470 / 0.488 | 0.550 / 0.569 | 0.520 / 0.537 | 0.532 / 0.536 |
| -1 | 0.609 / 0.616 | 0.764 / 0.774 | 0.700 / 0.712 | 0.724 / 0.729 |
| 0 | 0.739 / 0.744 | 0.989 / 0.988 | 0.947 / 0.956 | 0.973 / 0.976 |
| +1 | 0.584 / 0.588 | 0.734 / 0.733 | 0.673 / 0.676 | 0.701 / 0.700 |
| +2 | 0.430 / 0.449 | 0.444 / 0.446 | 0.437 / 0.459 | 0.442 / 0.461 |
| +3 | 0.356 / 0.376 | 0.348 / 0.356 | 0.351 / 0.375 | 0.360 / 0.370 |

Reading:
- **H is easier to decode (confirmed, robust to the interval):** z_H beats z_L at offsets -2 to +1, by 0.25 at the extraction position (0.989 vs 0.739; the 0.808 quoted above included the always-identical last positions), 0.155 at -1, 0.15 at +1, 0.08 at -2; at +2 and +3 the two are tied (0.43 to 0.44 and 0.35).
- **Same profile, different sharpness:** both streams peak at the current token and fall off symmetrically (bidirectional states also hold the next tokens, down to about 0.35 at +3). No offset favours z_L, so for TOKENS z_L is a blurrier copy of the same neighbourhood information, not a different kind of information.
- **The MLP does not close the gap** (offset 0: 0.744 vs 0.739): the L deficit is not a linear-accessibility effect.
- The sum is worse than z_H alone (0.947 vs 0.989): z_L adds noise for token identity; the concat (0.973) does not beat z_H either.
- Consequence for the verbalizer: the split verbalizer reaches 40 to 47% (H call) and 5 to 14% (L call) of a probe ceiling of 0.99 and 0.74; previous and next token are linearly readable at 0.76 / 0.73 (z_H) and 0.61 / 0.58 (z_L), so facts about the neighbours are a feasible verbalizer target.
- Open: tokens are only one thing a stream can hold; the split verbalizer's genre-keyword rates were higher for the L call than for the H call. Probes for source identity and position (`--target dataset`, `--target relpos`, added in the next commit) test whether z_L holds coarse context that z_H does not.

## Learning curve at the extraction position (user paste 2026-10-07; `--non-last --offsets 0 --train-sizes 500 2000 8000 --mlp-hidden 512`, same 3224 test rows, top-200 tokens)

| train rows | z_L linear / MLP | z_H linear / MLP | sum | concat |
|---|---|---|---|---|
| 500 | 0.336 / 0.323 | 0.822 / 0.820 | 0.591 | 0.729 |
| 2000 | 0.486 / 0.478 | 0.956 / 0.955 | 0.816 | 0.900 |
| 8000 | 0.680 / 0.681 | 0.986 / 0.986 | 0.930 | 0.966 |
| 12929 (all) | 0.739 / 0.744 | 0.989 / 0.988 | 0.947 | 0.973 |

- **z_H saturates at about 2000 to 8000 rows; z_L is still climbing at 13k rows** by roughly +0.17 to +0.19 per 4x more data (0.486 to 0.680 for 2000 to 8000; +0.059 for the last 1.6x). So the L deficit is mostly SAMPLE EFFICIENCY, not missing information: the token is in z_L, but a probe needs far more examples to read it (at 500 rows the gap is 0.49, at 2000 0.47, at 8000 0.31, at all rows 0.25). A rough extrapolation, only if the log-linear trend continued (a guess), would put z_L at 0.9 around 40k rows and at the z_H level around 90k.
- The MLP never helps (all rows: 0.744), so the difficulty is not nonlinearity either.
- **For the verbalizer:** the split SFT gives each call about 9k examples; z_L is in the data-hungry regime there, which fits the L call's weak token reading (5 to 14% vs probe 0.74), though the verbalizer is far below the probe even for the H call, so extraction through the adapter and LoRA costs a lot on top. Larger or more varied data (the v2 plan) is the lever that should help z_L most.

## Source identity (user paste 2026-10-07; `--non-last --target dataset`, 5 classes, 27160 rows, 5425 test rows, majority 0.368)

z_L 0.999 / MLP 0.999, z_H 0.992 / 0.994, sum 0.999, concat 0.998 (intervals +-0.001 to 0.002). Coarse context (which source a prompt comes from) is decodable almost perfectly from both streams; z_L is marginally higher (0.007, outside the intervals) and is the only target so far where z_L is not below z_H, but both are at the ceiling, so this cannot show a division of labour. It is consistent with the split verbalizer's genre-keyword rates (L call at or above H call) and says that source identity is not what makes z_L hard to read. A target with headroom (position in the prompt, `--target relpos`; finer-grained context) is needed.
