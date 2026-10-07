---
type: Project
title: HRM verbalizer - a two-stream natural-language autoencoder for Mimir
description: Goal, hypothesis and current status of the nla/hrm extension on the hrm branch.
tags: [overview, hrm, status]
timestamp: 2026-09-30
---

# Goal

Verbalize the internal states of DFM-Mimir (an HRM-Text model with a recurrent L/H structure, schedule L L L H L L L H) as natural language, and find out whether the fast L stream and the slow H stream store different things. Ideally the verbalizations of the two streams differ, and the difference is faithful to what the model uses.

# Method in one paragraph

Extract `z_L` and `z_H` just before Mimir's second H application ([hook site](/decisions/hook-site.md)). A frozen Qwen2.5-1.5B-Instruct with a verbalizer adapter reads both vectors (injected as two soft tokens) and writes `L: ... H: ...`; a reconstructor adapter maps each field back to its stream ([model](/decisions/verbalizer-model.md)). The pair is trained by format-only SFT ([SFT](/decisions/sft-is-format-only.md)) and then by RL on the reconstruction loss ([loss](/decisions/loss-and-reward.md), [RL](/decisions/rl-design.md)). Faithfulness is judged by patching reconstructions back into Mimir ([judge](/decisions/judge-design.md)).

# Headline (2026-10-08)

Hypothesis with strong but unreplicated evidence: **z_H holds WHAT (the token and its neighbours), z_L holds WHERE (the position in the prompt)**. Linear probes (z_H 0.989 vs z_L 0.739 for the token, z_L 0.669 vs z_H 0.607 for the position) and a verbalizer that sees one stream per call (each call at chance on the other call's fact) agree. Details, evidence table and falsification criteria: [H what, L where](/observations/h-what-l-where.md). The joint verbalizer's texts do not show it (they carry the shared part only).

# Status (2026-09-30)

- Pipeline implemented and smoke-tested end to end; one full pipeline run and one 200-step RL run completed on a single GPU ([RL run 1](/observations/rl-run-1.md)).
- No result yet on whether the fields are stream-specific ([open questions](/open-questions.md)).

# Where things are

- Code: `nla/hrm/`. Run instructions: `docs/hrm.md` (small example and full step-by-step). Invariants for contributors: `CLAUDE.md`, section "HRM extension".
- Paper-oriented prose: [methods notes](/paper/methods-notes.md).
