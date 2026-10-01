---
type: Decision
title: "Optional token-prefix targets: prepend the marked token to every SFT field"
description: "build.py --prefix-token adds a Marked-token prefix (taken from the stored context, no LLM call) to the L and H fields of the SFT data."
tags: [sft, data, faithfulness]
timestamp: 2026-10-02
---

# Decision

`nla.hrm.build --prefix-token` (av_sft and ar_sft stages) prepends `Marked token: "X". ` to each of the two fields, taking X from the ⟦marked⟧ token in the stored context (whitespace-only tokens are shown JSON-escaped). Recorded in the dataset sidecar as `build_options: {prefix_token: true}`. `nll_check --prefix-token` scores against the same targets.

# Why

- The [probe](/observations/token-probe.md) shows the token is almost perfectly decodable from the vectors, while the verbalizer quoted a wrong token every time ([RL run 2](/observations/rl-run-2-log-reward.md)).
- Most of an LLM-written explanation (names, plot details, wording) cannot be determined from the vector, so most of the SFT loss carries no vector signal; a short, deterministic, vector-determined target raises the signal.
- It gives a clean faithfulness measurement: `eval` prints how often the quoted token equals the real one.

# Trade-offs

- It changes what an explanation is (it states the token explicitly) and gives the reconstructor the token for free, so reconstruction FVE is no longer a pure measure of what the free text carries. Keep the no-prefix pipeline as the comparison.
- The prefix is identical in the L and H fields, so it does not by itself differentiate the streams.
