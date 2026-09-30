---
type: Observation
title: z_L and z_H are strongly anti-correlated at the hook (small-sample measurement)
description: cos(z_L, z_H) about -0.8 and cancellation ratio about 0.29 on a 3-prompt smoke corpus; full-corpus numbers still to be recorded.
tags: [diagnostics, hrm, cancellation]
timestamp: 2026-09-30
---

# Measurement

`python -m nla.hrm.diagnostics` on the smoke extraction (3 prompts, 12 rows, CPU, float32, Mimir-v1.5):

- cancellation ratio `||z_L + z_H|| / (||z_L|| + ||z_H||)`: mean 0.288
- `cos(z_L, z_H)`: mean -0.81
- every vector has norm 39.1918, which is sqrt(1536): the unweighted RMSNorm at the end of each stack fixes the norm.

# Reading

The two streams write in largely opposing directions before H reads their sum. A reconstruction loss on the individual streams alone can therefore reward content that H never sees, which is why the loss keeps a heavy [sum anchor](/decisions/loss-and-reward.md). It also means the sum s has a much smaller norm than either stream, which matters for [normalization](/decisions/shared-scalar-normalization.md).

# Status and caveats

- n = 12 rows from 3 prompts: this is a plumbing-level observation, not a statistic. Do not quote it in a paper.
- The full-corpus diagnostics (`base.parquet.diagnostics.json`, by dataset and by last-position versus other) were run on the real extraction but are not recorded here: TODO(results).
