---
type: Talk Slide
title: "One slide: H holds WHAT, L holds WHERE (probe table, verbalizer table, three examples)"
description: Copy-ready tables and examples for the headline slide, with the exact numbers, sample sizes, intervals, the selection rule for the examples and the caveats to put on the slide.
tags: [talk, slide, headline, probe, split-av]
timestamp: 2026-10-08
---

# Slide title

**Mimir's two recurrent states hold different things: z_H holds WHAT (the token), z_L holds WHERE (the position)**

# Table 1: linear probes on the stored vectors (test accuracy)

Held-out prompts (split by prompt group), non-last positions, about 3,200 test rows for the token targets and 5,400 for the others, 95% binomial interval about +-0.015.

| target | chance (majority) | z_L | z_H |
|---|---|---|---|
| token at the position | 0.083 | 0.739 | **0.989** |
| previous token | 0.082 | 0.609 | **0.764** |
| next token | 0.086 | 0.584 | **0.734** |
| position: which fifth of the prompt | 0.199 | **0.669** | 0.607 |
| source (5 datasets) | 0.368 | 0.999 | 0.992 |

Notes for the slide: an MLP probe gives the same ordering (position 0.711 vs 0.642); z_L's token deficit is sample efficiency (z_L 0.34 / 0.49 / 0.68 / 0.74 for 500 / 2k / 8k / 13k training rows; z_H 0.82 / 0.96 / 0.99 / 0.99). Caveats: top-200 tokens only; position is not controlled for prompt length and source; no shuffled-label control.

# Table 2: the verbalizer, one stream per call (accuracy of the stated fact, non-last rows)

The L call sees only z_L, the H call only z_H; both are trained to write the same two facts (`Marked token: "X". Position: K of 5.`). 95% intervals over rows; n=87 (iid) / 88 (ood). E11 = after SFT, E12 = after 500 RL steps from E11 (same SFT run).

| model, split | token, L call | token, H call | position, L call | position, H call | interaction |
|---|---|---|---|---|---|
| chance | 0.05 | 0.05 | 0.20 | 0.20 | 0 |
| E11 iid | 0.093 [0.035, 0.161] | **0.430** [0.329, 0.535] | **0.448** [0.345, 0.552] | 0.184 [0.103, 0.264] | +0.598 [0.437, 0.759] |
| E11 ood | 0.045 [0.011, 0.091] | **0.239** [0.159, 0.330] | **0.341** [0.250, 0.443] | 0.148 [0.080, 0.227] | +0.386 [0.239, 0.534] |
| E12 iid | 0.198 [0.116, 0.287] | **0.419** [0.314, 0.529] | **0.483** [0.379, 0.586] | 0.161 [0.092, 0.241] | +0.540 [0.379, 0.701] |
| E12 ood | 0.034 [0.000, 0.080] | **0.284** [0.193, 0.375] | **0.364** [0.273, 0.466] | 0.216 [0.136, 0.307] | +0.398 [0.250, 0.545] |

Interaction = (position L-call minus H-call) minus (token L-call minus H-call), paired over the same rows. Figures: `docs/okf/figures/dissociation_{iid,ood}_nonlast_{E11,E12}.png`. Caveats: one SFT run (the independent replicate has not been run; E12 is the same run after RL); intervals cover rows, not training runs; the rest of each explanation is invented prose (show only the fact line).

# Three examples (E11, iid, non-last)

Selection rule: rows where the L call states the position right and the token wrong and the H call states the token right and the position wrong. This full pattern occurs in 11 of 87 iid rows (13%; 6 of 88 ood, 7%); the reverse pattern (L token right and position wrong, H position right and token wrong) occurs in 0 of 87 (0 of 88 ood); E12: 9 of 87 and 7 of 88, reverse 0. Examples = the first rows of a seed-1 shuffle of the 11 matching iid rows, one per pattern-matching dataset where possible, then the next; the pattern is therefore not typical of all rows, but it is the pattern the table counts.

1. SimpleStories, token 15 of 123 (true fifth 1): prompt `…a girl discovered a car made ⟦of⟧ flowers and vines…`
   - L call: `Marked token: "the". Position: 1 of 5.` (position right, token wrong)
   - H call: `Marked token: "of". Position: 2 of 5.` (token right, position wrong)
2. GSM-symbolic, token 48 of 104 (true fifth 3): prompt `…a pair of swimming leggings for $9 more than the jersey cost, and ⟦a⟧ pair of cleats that were originally $78…`
   - L call: `Marked token: "the". Position: 3 of 5.`
   - H call: `Marked token: "a". Position: 1 of 5.`
3. SimpleStories, token 184 of 194 (true fifth 5): prompt `…leaving behind a sparkle in the air, a reminder ⟦of⟧ their colorful adventure.`
   - L call: `Marked token: "the". Position: 5 of 5.`
   - H call: `Marked token: "of". Position: 1 of 5.`

Say on the slide: the facts come out as the probes predict; the surrounding prose is invented (for example the H call for example 1 goes on about "a bag of popcorn" and a list of purchases, for a story about a car made of flowers), so do not show it.

# Figure instead of the tables

`docs/okf/figures/h_what_l_where.png` (and `.svg`, vector, better for slides): left, linear probes on z_L (blue) and z_H (orange) for five targets with majority baselines; right, the split verbalizer's per-call accuracy (E11, non-last rows, 95% CI over rows) on the marked token and the position fact, iid and ood, with chance. Regenerate: `python -m nla.hrm.bootstrap --dump E11=samples_split_sft_pos.jsonl --out results.json && python -m nla.hrm.plots hwhat --results results.json --model E11 --out docs/okf/figures/h_what_l_where.png`. The probe values in the left panel are the constants in `nla/hrm/plots.py` (`PROBES`, from [token probe](/observations/token-probe.md)); the error bars there are binomial intervals over the probe test rows.

# Copy-paste block for the slides (plain text)

Example 1 (SimpleStories, token 15 of 123)
Prompt: ...a girl discovered a car made [of] flowers and vines...
Truth: token "of", position 1 of 5
L call (sees only z_L): Marked token: "the". Position: 1 of 5.    position right, token wrong
H call (sees only z_H): Marked token: "of". Position: 2 of 5.     token right, position wrong

Example 2 (GSM-symbolic, token 48 of 104)
Prompt: ...a pair of swimming leggings for $9 more than the jersey cost, and [a] pair of cleats...
Truth: token "a", position 3 of 5
L call (sees only z_L): Marked token: "the". Position: 3 of 5.    position right, token wrong
H call (sees only z_H): Marked token: "a". Position: 1 of 5.      token right, position wrong

Example 3 (SimpleStories, token 184 of 194)
Prompt: ...a sparkle in the air, a reminder [of] their colorful adventure.
Truth: token "of", position 5 of 5
L call (sees only z_L): Marked token: "the". Position: 5 of 5.    position right, token wrong
H call (sees only z_H): Marked token: "of". Position: 1 of 5.     token right, position wrong

Footnote for the slide: examples are rows where this full pattern occurs (13% of iid rows; the reverse pattern occurs in 0%); the rest of each explanation is invented prose and is not shown.

Table form (paste into a slide table):

| Prompt (marked token in brackets) | Truth | L call (z_L only) | H call (z_H only) |
|---|---|---|---|
| ...a girl discovered a car made [of] flowers and vines... | "of", 1 of 5 | "the", 1 of 5 | "of", 2 of 5 |
| ...swimming leggings for $9 more than the jersey cost, and [a] pair of cleats... | "a", 3 of 5 | "the", 3 of 5 | "a", 1 of 5 |
| ...a sparkle in the air, a reminder [of] their colorful adventure. | "of", 5 of 5 | "the", 5 of 5 | "of", 1 of 5 |
