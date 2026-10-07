---
type: Observation
title: "Headline hypothesis: H holds WHAT (token), L holds WHERE (position); the split verbalizer reproduces it"
description: The probes and the split verbalizer agree on a division of labour between the two streams: z_H is the sharper readout of the current token and its neighbours, z_L of the position in the prompt; each split-verbalizer call is at chance on the other call's fact. Status, evidence, what would falsify it, what is missing.
tags: [headline, hypothesis, l-versus-h, probe, split-av, position]
timestamp: 2026-10-08
---

# The hypothesis

**z_H (the slow state after the first H update) encodes WHAT is at the position (the token and its neighbours); z_L (the fast state after the L cycles) encodes WHERE the position is in the prompt.** A verbalizer that sees one stream per call states each stream's fact, and only that one.

# Evidence so far (all n=100 rows or the probe test sets; intervals are over rows only)

| level | what was measured | z_L / L call | z_H / H call |
|---|---|---|---|
| probe | token at the extraction position, linear (27k rows, non-last, top-200) | 0.739 | **0.989** |
| probe | token at -1 / +1 | 0.609 / 0.584 | **0.764 / 0.734** |
| probe | position fifth of the prompt, linear / MLP (chance 0.20) | **0.669 / 0.711** | 0.607 / 0.642 |
| probe | source identity (5 classes) | 0.999 | 0.992 (tie at the ceiling) |
| probe learning curve | token: 500 / 2000 / 8000 / 12929 rows | 0.34 / 0.49 / 0.68 / 0.74 | 0.82 / 0.96 / 0.99 / 0.99 |
| split verbalizer (E11), non-last | marked token correct, iid / ood (chance about 5%) | 9% / 4.5% (chance) | **43% / 24%** |
| split verbalizer (E11), non-last | position fact correct, iid / ood (chance about 20%) | **45% / 34%** | 18% / 15% (chance) |

Sources: [token probe](/observations/token-probe.md), [experiments](/experiments.md) (probe profile, E11). The joint verbalizer cannot show this: its L and H texts carry only the shared part of the streams ([experiments](/experiments.md), cross-stream baselines and swap control).

# How strong it is

- Probe level: large margins (0.25 for the token at offset 0, 0.06 to 0.07 for position, intervals about 0.013 to 0.017), consistent across offsets, not closed by an MLP. The token deficit of z_L is mostly sample efficiency (learning curve), not absent information. Position is the only probed target where z_L wins; source identity is a tie at the ceiling.
- Verbalizer level: a double dissociation, each call at chance on the other's fact, from ONE SFT run with ONE seed, evaluated on 87 / 88 non-last rows, no intervals yet.

# What would falsify or weaken it

- The SFT replicate (with `--facts position`) does not reproduce the pattern.
- The bootstrap intervals on the non-last numbers overlap (n=87 / 88).
- A better-controlled position probe: relpos is confounded with prompt length and source (length distributions differ), no shuffled-label control, five bins only.
- The pattern disappears with another fact pair (for example previous token for L, or absolute position).

# Missing

Intervals on E11 (needs the dump), the replicate SFT (and RL), a position probe controlled within source and length, other fact pairs (neighbour tokens, absolute position), and RL on the fact-structured targets (planned overnight 2026-10-08, `ckpt/rl_split_pos`).

# Why it matters

It is the first answer to the project's question (do the two streams store different things?) that does not depend on the free-text explanations: the probes show it in the vectors, and the split verbalizer shows that the difference is verbalizable.
