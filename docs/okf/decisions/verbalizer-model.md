---
type: Decision
title: Qwen2.5-1.5B-Instruct as verbalizer and reconstructor, full-depth AR, learned affine adapters
description: A frozen Qwen base with two LoRA adapters; injection and reconstruction go through learned per-stream affine maps.
tags: [model, architecture]
timestamp: 2026-09-30
---

# Decision

- One frozen `Qwen/Qwen2.5-1.5B-Instruct` base, two PEFT LoRA adapters: `av` (verbalizer) and `ar` (reconstructor). LoRA r=32, alpha=64, on q/k/v/o/gate/up/down projections.
- Faithfulness to Mimir is enforced by using Mimir as a frozen judge (patch-back), never as a trainable component.
- Injection: one learnable affine map per stream (`inj_L`, `inj_H`, Linear d_mimir -> d_verbalizer with bias), near-identity initialization scaled by an initial scale.
- Reconstruction: `head_L`, `head_H`, affine maps from the AR's final hidden state to d_mimir.
- The AR is full depth (no truncation to a layer K) plus the affine head.

# Why

- Qwen2.5-1.5B has hidden size 1536, equal to Mimir's, so no dimensionality change is required. The affine maps are kept anyway for scale correction and flexibility.
- A copy of Mimir as verbalizer was the paper-faithful option but has no fast generation backend, cannot use flash-attention, needs `transformers>=5.13` in the training environment, and is weak at free-form English. A larger Qwen would need a learned dimensionality adapter and more compute.

# Deviation from the original NLA

The verbalizer is a different model from the target, and the AR is not truncated. The two-stream input and the field-structured output are extensions. State this when describing the method.

# Injection scale initialisation

The initial scale is large, following the original heuristic of matching the verbalizer's ambient residual norm: the 75th-percentile last-layer hidden-state norm of Qwen over a text sample (`norm_stats.py`), with a fallback of 5.0 when skipped. It is deliberately not initialised at Qwen's token-embedding norm.

# Open points

- Whether the run used the p75 scale or the 5.0 fallback is not recorded in this bundle; see [open questions](/open-questions.md).
