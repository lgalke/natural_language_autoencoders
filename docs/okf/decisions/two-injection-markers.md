---
type: Decision
title: Two distinct injection markers, verified in template context, injected with two calls
description: The AV prompt contains an [L] and an [H] slot, each a different single-token CJK character.
tags: [injection, tokenizer]
timestamp: 2026-09-30
---

# Decision

The verbalizer prompt has two slots, `<concept>{inj_L}</concept>` and `<concept>{inj_H}</concept>`, each filled by a different single-token CJK character. Injection is two separate calls to the unchanged `inject_at_marked_positions`, one per marker.

# Why two distinct markers

`nla.schema.compute_canonical_neighbors` asserts that its marker occurs exactly once in the template. A repeated marker would need that assert changed and an interleaved-vector convention; two markers reuse the existing machinery as-is.

# Verification in context

`find_two_injection_tokens` first picks characters that are single-token in isolation, then re-verifies every candidate pair inside the real template. See [BPE pitfall](/observations/pitfalls.md): a character that tokenizes to one token alone can change once it sits next to `<` or `>`.

# Storage convention

The stored parquet prompt contains the literals `<INJECT_L>` and `<INJECT_H>`; the real characters are substituted at load time. The sidecar (`HrmTokenMeta`) records ids and neighbour ids for both markers; nothing is hardcoded in training code.
