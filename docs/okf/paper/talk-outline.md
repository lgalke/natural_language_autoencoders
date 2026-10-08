---
type: Talk Outline
title: "Talk outline: what do Mimir's L and H streams store? (slide by slide, with sourced numbers)"
description: Draft structure for the talk of 2026-10-09 with the numbers to put on each slide, the figure files, what to say and what not to say, and which numbers are still pending.
tags: [talk, slides, results]
timestamp: 2026-10-08
---

All numbers are from [experiments](/experiments.md) unless stated. n=100 rows per split (87 / 88 non-last) for verbalizer numbers; "CI" = 95% interval over rows only (it ignores training-run variance, which is about 0.05 to 0.1 FVE on ood). Marked **PENDING** = not yet available when this was written.

# Storyline in one sentence

We built a small open verbalizer for Mimir's two recurrent states, showed that it reads real information from them (and where it does not), and found that the two streams hold different things: **z_H holds WHAT (the token and its neighbours), z_L holds WHERE (the position)**, visible in linear probes and, when each stream is verbalized in its own call, in what the verbalizer can state.

# Slides

1. **Question.** Hierarchical reasoning models (Mimir, HRM-Text: L fast cycles, H slow cycles, schedule L L L H L L L H). Do the two streams store different things? Idea: natural-language autoencoder (verbalizer reads a vector, reconstructor writes it back).
2. **What we read.** Hook just before the second H application: z_L (L output after the 6th L call), z_H (H output after the first H call); H reads their sum. z_L and z_H are strongly anti-correlated (mean cos -0.85) by a constant offset; the sum holds only 15% of the squared norm; around the mean only 22 to 34% of one stream is linearly predictable from the other (R^2 0.218 / 0.337), so most variation is stream-specific.
3. **Method.** Frozen Qwen2.5-1.5B-Instruct + LoRA (r=32) verbalizer; per-stream affine injection adapters; two-slot prompt; reconstructor with two heads; shared-scalar normalisation (keeps z_L + z_H additive); reward = -log MSE; SFT on GLM-5.3 explanations (about 9.2k rows), then GRPO-style RL. Evaluation: FVE vs mean predictor, patch-back judge in Mimir, linear probes, shuffle controls. State the deviations from the original NLA (small foreign verbalizer, LoRA, full-depth reconstructor, two streams, small data).
4. **The vectors are read.** Linear probe: token at the extraction position 0.989 from z_H, 0.739 from z_L (non-last, chance 0.005). Shuffled vectors collapse FVE below the mean and token reading to 0 to 2% (E5). Figure: `docs/okf/figures/token_progress.png` (token accuracy by design iteration: 0% to 49% / 30% iid / ood, chance about 5%).
5. **Functional fidelity (judge).** Replacing Mimir's H input with the reconstruction instead of the true vector: KL to the clean output 3.81 (iid) / 4.82 (ood) nats vs 7.17 / 7.93 for the mean vector; ratio of means +0.47 / +0.39; better than the mean in 81% / 77% of rows; another row's reconstruction is worse than the mean (KL 8.5 to 8.8). Say "recovers about half of the effect, specific to each row", not "faithful". (E7 joint; E9 split gives +0.41 / +0.44.) Intervals (95%, rows, share of the mean-ablation KL removed): E7 iid 0.469 [0.367, 0.564], ood 0.391 [0.303, 0.475]; E9 split iid 0.407 [0.304, 0.507], ood 0.438 [0.358, 0.508] (not distinguishable); another row's reconstruction iid -0.225 [-0.370, -0.091] (worse than the mean vector), ood -0.092 [-0.215, 0.015] (no better than the mean vector).
6. **What the explanations are (honest slide).** Right: genre/register, language, the marked token for function words (iid 92%, ood 81%). Wrong: content words (iid 30%, ood 12%), invented details (14% of quoted spans and 1 to 3% of names appear in the prompt), and on a new source the genre falls back to the training genre (a murder-mystery prompt described as a children's story). Example: `examples.md` / `show_examples` (pick by seed, show one right and one wrong).
7. **Joint verbalizer: L and H are not separated (negative).** FVE per stream equals what the other TRUE stream predicts linearly (iid all rows L 0.183 / H 0.332 vs ridge 0.144 / 0.283; ood H 0.264 vs 0.271); swapping the streams between the slots changes nothing (FVE within +-0.03); L and H texts differ in style (a classifier separates them at 91%) but not in content; the two teacher samples are exchangeable, so this is not a data artefact. Conclusion: a joint prompt cannot show it.
8. **Probes: H = what, L = where.** Table (linear probe test accuracy; offsets are token positions relative to the extraction position):

   | target | z_L | z_H |
   |---|---|---|
   | token, offset 0 | 0.739 | **0.989** |
   | token, offset -1 / +1 | 0.609 / 0.584 | **0.764 / 0.734** |
   | position fifth of the prompt (chance 0.20) | **0.669** | 0.607 |
   | source identity (5 classes) | 0.999 | 0.992 (tie) |

   Learning curve (token, 500 / 2000 / 8000 / 12929 rows): z_L 0.34 / 0.49 / 0.68 / 0.74; z_H 0.82 / 0.96 / 0.99 / 0.99: z_L's token is data-hungry, not absent; an MLP never helps. Caveats on the slide: top-200 tokens, relpos confounded with length and source, no shuffled-label control.
9. **Split verbalizer: one stream per call.** Design: the L call sees only z_L (the other slot zeroed) and writes the L field; the H call likewise. Same AR, reward and judge. It matches the joint model on iid (paired FVE difference -0.005 [-0.033, 0.024]) and in the judge (+0.41 / +0.44) while the two calls are independent by construction. The ood FVE advantage seen once (+0.10) is NOT robust (a second checkpoint of the same pipeline, E10, was 0.085 lower): do not claim it.
10. **Headline: a double dissociation (E11).** Both calls are trained to state the same two facts (marked token, position fifth). Non-last rows, iid / ood: marked token L call 9% / 4.5% (chance 5%), H call 43% / 24%; position L call 45% / 34%, H call 18% / 15% (chance 20%). Each call is at chance on the other call's fact. Intervals (95%, rows): iid token L 0.093 [0.035, 0.161] vs H 0.430 [0.329, 0.535], position L 0.448 [0.345, 0.552] vs H 0.184 [0.103, 0.264]; the interaction contrast (position L-H minus token L-H) is +0.598 [+0.437, +0.759] iid and +0.386 [+0.239, +0.534] ood. Figures: `docs/okf/figures/dissociation_{iid,ood}_nonlast_E11.png`. After RL (E12, same SFT run): interaction +0.540 [+0.379, +0.701] iid, +0.398 [+0.250, +0.545] ood; figures `dissociation_{iid,ood}_nonlast_E12.png`. Independent replicate SFT (2026-10-09): token half replicates on both splits (H call minus L call +0.24 iid, +0.19 ood, intervals exclude zero), position half on iid (+0.21 [+0.09, +0.33]) but not on ood (+0.03 [-0.07, +0.14]); interaction +0.448 [+0.299, +0.609] iid, +0.227 [+0.080, +0.375] ood. Figures `dissociation_{iid,ood}_nonlast_E11_vs_replicate.png`. Wording for the slide: "H holds what: replicated; L holds where: replicated on iid, weaker on ood". Status: one SFT run, n=87 / 88; say it as "reproduces the probe pattern in the verbalizer", replicated or not depending on the replicate.
11. **What did not work (credibility).** Verifiable rewards on quotes and names (E8): the model quoted 3.6 times less but invented the same amount in lowercase (rare words not in the prompt 44.5 vs 44.3 per row): reward gamed through its blind spot; best-of-8 selection by reconstruction: no gain over greedy (token 48% vs 51%), 8 times the compute; 600 more RL steps (E10) lowered held-out FVE (-0.05 iid, -0.09 ood). Lesson: checkpoint-to-checkpoint variance (about 0.05 to 0.1 FVE) exceeds the row intervals; small FVE differences between models are not claims.
12. **Limitations and confounds.** One run per condition; n=100; intervals over rows only; small verbalizer + LoRA vs the original design; the teacher describes the context, not a stream; probes are linear and top-200 tokens; the position probe is not controlled for length and source. The text-only context baseline and a larger verbalizer (3B) were not run.
13. **Next.** More and more varied data (the probe learning curve predicts z_L benefits most); other fact pairs (neighbour tokens, absolute position); a fact-structured verbalizer trained on programmatic facts instead of teacher prose; replicate seeds; larger verbalizer or a Mimir copy; controlled position probe.

# Say / do not say

- Say: "reads real, row-specific information"; "recovers about half of the output effect"; "z_H reads the token, z_L the position (probe and split verbalizer)"; "the joint verbalizer's text carries only the shared part".
- Do not say: "faithful" or "explains the model"; "L and H texts differ in content" for the joint model; that the split model is better than the joint one on FVE; that E8 or best-of-N helped; any single-run FVE difference below about 0.1 as a finding.

# Figures (regenerate with `nla.hrm.bootstrap` then `nla.hrm.plots`)

In the repo: `docs/okf/figures/token_progress.png`, `fve_{iid,ood}_all_E7_vs_split.png`, `token_{iid,ood}_nonlast_E7_vs_split.png`. To make: `plots dissociation` (E11), `plots judge` (needs the eval JSON), `plots bars --metrics tok_L tok_H pos_L pos_H` for the replicate. All figures use the same palette and 95% row CIs; keep the model order fixed across figures.

# Still missing when this was written

The judge for E12 (`ckpt/rl_split_pos`); the iid `final` FVE line of E7 is available (0.235 [0.174, 0.296]).
