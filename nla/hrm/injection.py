"""Two-marker injection token selection for the HRM verbalizer's prompt.

The AV prompt needs to inject TWO vectors per sample (z_L into an [L] slot,
z_H into an [H] slot). `nla.schema.compute_canonical_neighbors` and
`nla.datagen.injection_tokens.find_injection_token` are built around a
single marker that appears exactly once in the template (`nla/schema.py`
asserts this) — reusing them for two markers would mean special-casing that
assert, so instead we pick TWO distinct single-token CJK marker chars
(`inj_L`, `inj_H`) and inject them with TWO separate calls to
`nla.injection.inject_at_marked_positions` (unchanged, called twice — see
`nla/hrm/model.py:build_inputs_embeds`), each scanning for its own marker id.

Cached separately from `nla/datagen/injection_token_cache.yaml` (different
value shape — a pair, not a single char) so the two systems' caches never
collide or get misread as each other's format.
"""

from pathlib import Path
from typing import Any

import yaml

from nla.datagen.injection_tokens import _INJECTION_RANGE, _tokenize_one

_CACHE_PATH = Path(__file__).parent / "injection_token_cache_hrm.yaml"


def _load_cache() -> dict[str, dict[str, Any]]:
    if not _CACHE_PATH.exists():
        return {}
    loaded = yaml.safe_load(_CACHE_PATH.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    _CACHE_PATH.write_text(yaml.safe_dump(cache, allow_unicode=True, sort_keys=True))


def find_two_injection_tokens(tokenizer: Any, actor_template: str) -> tuple[str, int, str, int]:
    """Auto-pick two DISTINCT single-token CJK chars for the [L]/[H] markers,
    verified to STAY single-token AND single-occurrence when embedded in
    `actor_template` (context can change BPE merges at the `<concept>{c}</concept>`
    boundary even for a char that tokenizes alone to one token — this is why
    verification happens against the real template, not just the bare char,
    same rationale as `nla.schema.compute_canonical_neighbors`).

    Cached (re-verified against the live tokenizer on cache hit — same drift
    protection as `find_injection_token`)."""
    key = tokenizer.name_or_path
    cache = _load_cache()
    if key in cache:
        entry = cache[key]
        ids_l = _tokenize_one(tokenizer, entry["char_L"])
        ids_h = _tokenize_one(tokenizer, entry["char_H"])
        assert len(ids_l) == 1 and ids_l[0] == entry["id_L"], (
            f"cached L-marker for {key!r} no longer valid: {entry['char_L']!r} -> {ids_l} "
            f"(cached id={entry['id_L']}). Delete the cache entry and rerun."
        )
        assert len(ids_h) == 1 and ids_h[0] == entry["id_H"], (
            f"cached H-marker for {key!r} no longer valid: {entry['char_H']!r} -> {ids_h} "
            f"(cached id={entry['id_H']}). Delete the cache entry and rerun."
        )
        compute_canonical_neighbors_two(
            tokenizer, actor_template, entry["char_L"], entry["id_L"], entry["char_H"], entry["id_H"]
        )
        return entry["char_L"], entry["id_L"], entry["char_H"], entry["id_H"]

    lo, hi = _INJECTION_RANGE
    # Single-token-in-isolation candidates, in codepoint order.
    candidates: list[tuple[str, int]] = []
    for codepoint in range(lo, hi + 1):
        char = chr(codepoint)
        ids = _tokenize_one(tokenizer, char)
        if len(ids) == 1:
            candidates.append((char, ids[0]))

    for i, (char_l, id_l) in enumerate(candidates):
        for char_h, id_h in candidates[i + 1 :]:
            try:
                compute_canonical_neighbors_two(tokenizer, actor_template, char_l, id_l, char_h, id_h)
            except AssertionError:
                continue
            cache[key] = {"char_L": char_l, "id_L": id_l, "char_H": char_h, "id_H": id_h}
            _save_cache(cache)
            return char_l, id_l, char_h, id_h

    raise AssertionError(
        f"no pair of the {len(candidates)} single-token-in-isolation CJK chars in "
        f"U+{lo:04X}-U+{hi:04X} survives as two single-occurrence markers in the actor "
        f"template for tokenizer {key!r}. Widen _INJECTION_RANGE or hand-pick chars."
    )


def compute_canonical_neighbors_two(
    tokenizer: Any,
    actor_template: str,
    inj_l_char: str,
    inj_l_id: int,
    inj_h_char: str,
    inj_h_id: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Like `nla.schema.compute_canonical_neighbors` but for a template with
    two distinct `{inj_L}`/`{inj_H}` placeholders. Returns
    ((left_L, right_L), (left_H, right_H))."""
    content = actor_template.format(inj_L=inj_l_char, inj_H=inj_h_char)
    # NOTE: apply_chat_template(tokenize=True) returns a BatchEncoding (dict-
    # like) on transformers>=5.13, not a bare list of ids — indexing
    # ["input_ids"] is required, or `for tid in ids` silently iterates dict
    # KEYS instead of token ids (0 matches, wrong-looking "marker not found"
    # error). nla.schema.compute_canonical_neighbors has the same bare-`ids`
    # pattern; worth checking there too if this repo's pinned transformers
    # version ever moves to one where apply_chat_template returns dicts.
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
    )["input_ids"]

    def _one(target_id: int, name: str) -> tuple[int, int]:
        matches = [i for i, tid in enumerate(ids) if tid == target_id]
        assert len(matches) == 1, (
            f"{name} marker id {target_id} appears {len(matches)}x in canonical prompt "
            f"(expected 1). Template: {content!r}"
        )
        p = matches[0]
        assert 0 < p < len(ids) - 1, f"{name} marker at position {p} is at edge of sequence"
        return ids[p - 1], ids[p + 1]

    return _one(inj_l_id, "L"), _one(inj_h_id, "H")
