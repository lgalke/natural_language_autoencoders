---
type: Plan
title: "Data collection v2: proposed changes to the extraction corpus and position sampling"
description: Ranked proposals for a better activation dataset than the ad-hoc v1 pipeline; none implemented yet, to be prioritised after the data_report numbers.
tags: [data, plan, extraction, sampling]
timestamp: 2026-10-02
---

# Status

Proposal only. The v1 data (see [corpus and splits](/decisions/corpus-and-splits.md)) came from a corpus script that grew out of "get something loading": a fixed number of prompts per public source, positions sampled uniformly inside the rendered prompt, last prompt position always added. Nothing about the mix was designed. Every item below changes the data, so it means new splits, new `norm_stats`, and new SFT and RL runs: none of the existing checkpoints or numbers carry over.

# Measure first

`python -m nla.hrm.data_report --base base.parquet` (CPU, seconds) reports rows and prompts per dataset, prompt lengths, the share of last-position rows, the share of rows that sit inside the shared chat-template tail, how concentrated the marked tokens are, the position distribution within prompts, and near-duplicate prompts. 

## v1 numbers (`data_report` on the real `base.parquet`)

32592 rows from 5432 prompts (6 rows per prompt: 5 sampled + the last position).

| dataset | rows | share | prompts | mean / max length (tokens) |
|---|---|---|---|---|
| gsm-symbolic | 12000 | 36.8% | 2000 | 72 / 159 |
| simplestories | 12000 | 36.8% | 2000 | 291 / 728 |
| da_instruct (DynaWord) | 5970 | 18.3% | 995 | 375 / 2044 |
| musr (held-out OOD) | 1500 | 4.6% | 250 | 1175 / 1526 |
| bbh | 1122 | 3.4% | 187 | 236 / 574 |

- Last-prompt-position rows: 16.7% (exactly one of six per prompt), all the same marked token (id 107, the newline). The shared chat-template tail is 5 tokens; rows inside it are 20.3% in total, so only 3.6% beyond the last position.
- 4756 distinct marked tokens; the top-8 cover 34.5% of rows (id 107 alone 18.8%, then punctuation and function-word ids at 1% to 4%).
- Position within the prompt, quintiles: 15%, 17%, 17%, 17%, 33% (the last quintile includes the tail).
- Near-duplicate prompts (same first 200 characters with digits collapsed): 742 of 5432 (13.7%) in 270 groups.
- Two sources (GSM-Symbolic, SimpleStories) are 74% of the rows; the reasoning sources BBH and MuSR together are 8%. DynaWord kept 995 of 2000 prompts because the 2048-token length filter drops about half of its long documents (so it is biased towards short documents).

## What the numbers change

- The template tail beyond the last position is a small problem (3.6% of rows); the large one is the **last prompt position** (16.7%, a single identical token). Proposal 1 below should focus on capping the last position.
- **Metric consequence:** a verbalizer that always quotes the newline would be right on 17% to 19% of rows, so overall marked-token quote accuracy is inflated. `eval` now prints the non-last-position numbers separately; read quote accuracy there.
- The mix is dominated by two unlike sources, and templated GSM-Symbolic variants are likely the bulk of the near-duplicates; whether they leak into the in-distribution eval is measured by `data_report --splits-dir splits/` (**TODO: record the leakage numbers**).

Indirect evidence already in hand: in the [token probe](/observations/token-probe.md) the single most common marked token was 28.5% of rows, which suggests that template tokens and last-position rows are a large share of v1.

# Proposals, ranked by expected value for the research question

1. **Cap the last position (and drop the rest of the template tail from random sampling).**
   - Every prompt ends with the same few template tokens (`<turn|>`, newline, `<|turn>`, `model`, newline; 5 tokens in the smoke measurement). The sampler draws uniformly from token index 3 up to the end, so it can land there, and for short prompts that is a large fraction of the rows (50% on the toy prompts).
   - Rows on these tokens say almost nothing about the particular prompt and inflate the share of one identical marked token, which also makes "marked token" targets and quote accuracy easier to game.
   - Change: exclude the common tail from random sampling in `stage0_hrm.py`; keep the last prompt position as a separate, capped number of rows per prompt (the "about to answer" state is scientifically interesting; it just should not be 15% to 50% of the data).
   - Effort: small. Risk: low.

2. **Add response positions (causal region).**
   - v1 samples only the bidirectional prompt. During generation Mimir's tokens are causal and the answer is being produced; this is plausibly where the L and H streams differ most, and v1 has no data from there. Without it the project cannot say anything about the dynamics of generation.
   - Change: render prompt plus a response (Mimir's own sampled response, or gold answers teacher-forced), mark the prompt with `token_type_ids = 1` and the response with 0, capture at response positions. Judge and `infer.py` must reproduce this exactly (store the full ids and the type ids, not only the prompt ids). The verbalizer prompt and the explainer prompt need a position-kind notion (the explainer must no longer say the model reads the whole text bidirectionally for these rows).
   - Effort: large; treat as a separate phase. Decide the response source first (own samples versus gold) since it changes what the activations mean.

3. **Control the mix, and deduplicate.**
   - Set explicit per-source weights and balance by rows (not prompts), so long prompts do not dominate through length and short ones are not swamped.
   - Near-duplicates: GSM-Symbolic problems are templated variants of the same question. If variants land in different buckets (the split is per prompt), the in-distribution eval is optimistic. Group templated sources by template or original id before splitting, and drop exact and near duplicates.
   - Check what the Danish DynaWord `text` column is as a "user message": raw documents are not instructions, and the length filter keeps only the short ones.
   - Effort: medium (corpus script and `split.py`). Risk: low.

4. **Scale up.**
   - The original NLA used hundreds of thousands of rows for SFT; v1's AV-SFT bucket has about 9k. Extraction is cheap (more prompts per source, more positions per prompt). The cost is the two LLM explanations per row, so the affordable size depends on LLM throughput. Options: one longer explanation per row split into fields, shorter structured explanations, or the [token-prefix targets](/decisions/token-prefix-targets.md), which need no LLM at all.

5. **Use HRMMix when available.** It is the intended reasoning mixture; the public sources (GSM-Symbolic, BBH, MuSR, SimpleStories, DynaWord) are stand-ins. Everything above should be implemented so that a local JSONL (`--local-jsonl`) goes through the same sampling and splitting.

# Also worth fixing in the same pass

- Position sampling by token content (for example a minimum share of content words) if the data report shows punctuation and function words dominate.
- Record per-row provenance needed later: the position kind (prompt interior, last prompt position, response), the source dataset, and the template group for dedupe.
- Keep the held-out source (currently MuSR) entirely out of every other bucket, as in v1.

# Decision needed before any of this

Whether the research question requires item 2 (response positions). If it does, items 1 and 3 should be built in the same pass, to avoid collecting three versions of the data.
