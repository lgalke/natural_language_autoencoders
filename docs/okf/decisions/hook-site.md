---
type: Decision
title: Hook just before Mimir's second H application; store z_L and z_H separately
description: z_L = L_out@2.3, z_H = H_out@1, and their sum is exactly what H's second application reads.
tags: [model, extraction, hrm]
timestamp: 2026-09-30
---

# Mimir's recurrence

Mimir-v1.5: d_model 1536, H_cycles 2, L_cycles 3, 16 blocks per stack, each stack ends in an unweighted RMSNorm. Schedule: L L L H L L L H.

```
z_L = 0 ; z_H = embed(tokens) * embedding_scale
for h in 1..2:
    repeat 3 times: z_L = L(z_L + z_H)
    z_H = H(z_H + z_L)
```

# Decision

Capture at the point just before the second H application:

- `z_L` = output of the 6th L call (L_out@2.3)
- `z_H` = output of the 1st H call (H_out@1)
- `s = z_L + z_H` = H's second-cycle input (H_in@2)

Hooks are forward hooks on `model.model.L_module` / `H_module` (whole stacks), matching HRM-Interp's `CycleStateCollector`. Each extraction asserts 6 L and 2 H firings and that `z_L + z_H` equals the recorded H_in@2.

# Why

- It is the last point at which the two streams are still separate before H combines them, so it supports both a sum-level and a per-stream analysis.
- z_H persists through cycle 2 (it is re-added at every L step), which makes the z_H-only patch a causal test of the H component; see [judge](/decisions/judge-design.md).

# Known property

Both streams have norm about sqrt(1536) = 39.19 (RMSNorm), and on the small smoke corpus they are strongly anti-correlated; see [cancellation](/observations/cancellation.md).
