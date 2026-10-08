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

- **Replicate (independent SFT run, 2026-10-09):** the token half replicates on both splits (H call minus L call +0.241 [+0.149, +0.345] iid, +0.193 [+0.102, +0.284] ood); the position half replicates on iid (L call minus H call +0.207 [+0.092, +0.333]) but NOT on ood (+0.034 [-0.068, +0.136], against +0.193 in the first run). Interaction +0.448 [+0.299, +0.609] iid, +0.227 [+0.080, +0.375] ood (first run +0.598, +0.386). In the replicate the L call is also above chance on the token on iid (0.195). Figure: `docs/okf/figures/dissociation_{iid,ood}_nonlast_E11_vs_replicate.png`. So: "H reads WHAT" is replicated; "L holds WHERE" is supported on iid by the probe (z_L 0.669 vs 0.607) and by two runs, and weaker on ood.

- **After RL (E12, RL from the position-fact SFT, 500 steps):** the dissociation survives: interaction +0.540 [+0.379, +0.701] iid and +0.398 [+0.250, +0.545] ood (non-last, 87 / 88 rows). RL raised the L call's token reading on iid from 0.093 to 0.198 (still well below the H call's 0.419); the H call stays at chance on position. Figures: `docs/okf/figures/dissociation_{iid,ood}_nonlast_E12.png`. Same SFT run as E11, so this is not a replicate.

- **E11 intervals (5000 row resamples, non-last):** the interaction (position L minus H) minus (token L minus H) is +0.598 [+0.437, +0.759] on iid and +0.386 [+0.239, +0.534] on ood; each contrast alone also excludes zero (token H minus L +0.333 [+0.230, +0.437] iid; position L minus H +0.264 [+0.138, +0.391] iid). Figures: `docs/okf/figures/dissociation_{iid,ood}_nonlast_E11.png`. These intervals cover which rows were drawn, not which SFT run: still one run.

- Probe level: large margins (0.25 for the token at offset 0, 0.06 to 0.07 for position, intervals about 0.013 to 0.017), consistent across offsets, not closed by an MLP. The token deficit of z_L is mostly sample efficiency (learning curve), not absent information. Position is the only probed target where z_L wins; source identity is a tie at the ceiling.
- Verbalizer level: a double dissociation, each call at chance on the other's fact, from ONE SFT run with ONE seed, evaluated on 87 / 88 non-last rows (row intervals above).

# A supporting observation: the L call also states the genre better

Crude keyword rule (the source's genre word anywhere in the field), n=100 per split: E11 iid L call 0.78 vs H call 0.55 (paired +0.23 [+0.12, +0.34]), ood 0.77 vs 0.34 (+0.43 [+0.31, +0.55]); E12 and the earlier split SFT without facts (a separate SFT run) point the same way (details in [full examples](/paper/slide-examples-full.md)). The source probe is a tie at the ceiling, so this is extraction, not content. Suggests the refinement "L holds where and what kind of document; H holds the local token", to be tested with a real genre/classification probe.

# What would falsify or weaken it

- (Replicate run 2026-10-09: the pattern is reproduced for the token on both splits and for the position on iid, not for the position on ood.)
- A better-controlled position probe: relpos is confounded with prompt length and source (length distributions differ), no shuffled-label control, five bins only.
- The pattern disappears with another fact pair (for example previous token for L, or absolute position).

# Missing

The replicate SFT (and RL), a position probe controlled within source and length, other fact pairs (neighbour tokens, absolute position), and RL on the fact-structured targets (planned overnight 2026-10-08, `ckpt/rl_split_pos`).

# Why it matters

It is the first answer to the project's question (do the two streams store different things?) that does not depend on the free-text explanations: the probes show it in the vectors, and the split verbalizer shows that the difference is verbalizable.
