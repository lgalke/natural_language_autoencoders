---
type: Talk Outline
title: "Likely questions after the talk, with short honest answers and the numbers behind them"
description: Prepared answers for the split-verbalizer talk (H holds WHAT, L holds WHERE): validity of the claim, method, baselines and controls, generalisation, HRM-specific interpretation, and the hostile questions; each answer says what is measured, what is not, and what to say when it is not known.
tags: [talk, qa, limitations]
timestamp: 2026-10-09
---

Numbers are from [experiments](/experiments.md): n=87 (iid) / 88 (ood) non-last rows for the facts, n=100 per split for FVE and the judge, intervals are 95% over rows only. "Run" = an independent SFT run.

# Is the finding real?

1. **Is the L/H difference an artefact of your design (zeroed input, chosen facts)?** The same direction appears without any verbalizer: linear probes on the raw vectors read the token from z_H better (0.989 vs 0.739; previous / next token 0.764 / 0.734 vs 0.609 / 0.584) and the position fifth from z_L better (0.669 vs 0.607; MLP 0.711 vs 0.642); an MLP does not close the gap. The split verbalizer reproduces it: token H call minus L call +0.33 iid / +0.19 ood, position L call minus H call +0.26 / +0.19 (first run). Both calls get identical supervision (the same two programmatic facts).
2. **Only one run, 87 rows?** Two independent SFT runs (more if the extra seeds finished). Token half (H reads it better) replicates on both splits (+0.24 iid, +0.19 ood in the replicate); position half (L states it better) replicates on iid (+0.21) but not on ood (+0.03 [-0.07, +0.14] vs +0.19 in the first run). Say exactly that. Intervals cover rows, not runs.
3. **Isn't "position" confounded with prompt length and source?** Yes: the position target is the fifth of the rendered prompt; length distributions differ by source and I did not control for it. Source identity alone cannot explain the probe result (both streams read the source at 0.99+). Planned controls (not run): absolute-position bins, probes within each source, shuffled-label control. The token result is not affected by this.
4. **z_L is about -z_H (cos -0.85), so aren't they one signal?** The anti-correlation is mostly a constant offset. Around the mean only 22 to 34% of one stream is linearly predictable from the other (R^2 0.218 / 0.337), and the sum holds only 15% of the squared norm; a joint verbalizer's L and H texts do carry only the shared part (FVE equals the cross-stream baseline, swapping the streams changes nothing), which is exactly why the split design was needed.
5. **Why would the slow H state hold the token and the fast L state the position?** I do not know and would not read "slow / fast" semantics into it: z_H here is the H output right after its first update (it has just read the L state), z_L the L output after the sixth L call, at one hook site only. Whether the pattern changes at other cycles is untested.

# Method and validity

6. **Is the verbalizer faithful or guessing from priors?** Two separate statements. The facts (token, position) and the genre are stated above chance and match the probes. The surrounding prose is not reliable: only 14% of quoted spans and 1 to 3% of capitalised names occur in the prompt; characters, numbers and plots are invented. Functionally, patching Mimir's second H input with the reconstruction removes about 40 to 50% of the output KL versus the mean vector (iid 0.47 [0.37, 0.56], ood 0.39 [0.30, 0.48] for the joint model; split 0.41 [0.30, 0.51] / 0.44 [0.36, 0.51]) and another row's reconstruction is worse than the mean vector on iid (-0.22 [-0.37, -0.09]).
7. **How do you evaluate the texts, is there an LLM judge?** No. The facts are programmatic and exact-match; genre is a crude keyword rule (L call 0.78 vs H call 0.55 iid, 0.77 vs 0.34 ood in E11); fidelity is the patch-back KL in Mimir; reconstruction is FVE.
8. **What did RL do? Isn't RL supposed to carve out the difference?** SFT with identical targets is the controlled place to read the stream difference. RL raised the reconstruction (FVE +0.12 iid, +0.18 ood over the SFT) but neither sharpened nor erased the dissociation (interaction +0.60 after SFT, +0.54 after RL), moved both calls towards a few stock texts, and 600 more RL steps lowered the held-out FVE (-0.05 iid, -0.09 ood). A per-call fact reward in RL is the experiment that would test reward-driven carving; not run.
9. **Reward hacking?** Yes, we saw it: a penalty on quoted spans and names (E8) made the model quote 3.6 times less while inventing the same amount in lowercase; best-of-8 selection did not help. The headline does not depend on RL.
10. **Why a small Qwen with LoRA, not a bigger model or a copy of Mimir?** Cost and time; the probe learning curve says z_L's token needs about 10 times more data than z_H's (z_L 0.34 / 0.49 / 0.68 / 0.74 for 500 / 2k / 8k / 13k rows; z_H 0.82 / 0.96 / 0.99 / 0.99), so data is the first suspect, capacity not tested (a 3B probe was planned, not run).

# Numbers and baselines

11. **The FVE is low.** Yes: 0.23 to 0.26 iid, 0.04 to 0.15 ood, about what the other true stream explains linearly (iid L 0.144, H 0.283). FVE = 1 - MSE(reconstruction) / MSE(mean vector) on vectors normalised by sqrt(||z_L||^2 + ||z_H||^2). It is supporting evidence, not the claim; differences of 0.05 to 0.1 between models or checkpoints are within run-to-run variation (E10 vs E9 was -0.085 on ood).
12. **Why does the split model not beat the joint one?** It does not: iid parity (paired FVE difference -0.005 [-0.033, 0.024]); the ood gap seen once (+0.10) was not robust (another checkpoint of the same pipeline was only +0.017 above the joint model). The split design buys attribution to one stream, not reconstruction.

# Scope and generalisation

13. **Other models, tasks, longer contexts?** One model (DFM-Mimir-v1.5), one hook site, short prompts from gsm-symbolic, SimpleStories, Danish instructions, BBH, with MuSR held out. Not tested: other checkpoints, long reasoning traces, generated text.
14. **What about the generation-start position, where the next word is predicted?** The verbalizer does not read it: the H-call text is nearly identical across different prompts (3 distinct openings in 25 rows in E11, the same invented arithmetic problem for a GSM prompt, a Danish legal text, a story and a murder mystery); the L call gets language or genre right in several cases (Danish legal text, story) and wrong for MuSR. Do not claim anything about the model's answer plan.
15. **Is this causal, does the model USE these distinctions?** No: everything is decodability. The only functional test is the patch-back KL with reconstructions. A single-stream ablation in Mimir (replace only z_L or only z_H by its mean) is a quick next experiment, not run.
16. **Can you steer the model by editing the text?** Not tested.

# Hostile ones

17. **Isn't this probing with extra steps?** Partly, and I would say so: the probes carry the core result. What the verbalizer adds is (a) an open pipeline, (b) evidence that the information is accessible in language under identical supervision for both streams, (c) negative results that matter: the joint prompt cannot show the difference, surface rewards get gamed, more RL hurts, the generation-start position collapses to stock text.
18. **Why trust examples you picked?** The examples are selected: rows where the full pattern occurs (13% of iid rows; the reverse pattern 0%; with the extra condition that both calls name the right genre, 3%). The tables, not the examples, carry the claim.
19. **What would change your mind?** The position half failing on more runs, the probe gap vanishing in a within-source / length-controlled probe, or the token half disappearing at other hook sites.

# If you do not know

Say "not tested" plus what the test would be (one of: controlled position probe, stream ablation, other hook sites, 3B verbalizer, more seeds, per-call fact reward). The honest list of what was not run is in [experiments](/experiments.md) and the limitations slide of the [talk outline](/paper/talk-outline.md).
