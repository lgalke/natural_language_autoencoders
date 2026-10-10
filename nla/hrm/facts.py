"""Programmatic facts about the marked position (no LLM calls): the sentences that open every L/H field in the fact-structured
SFT data and that `eval` scores against the truth."""

import re

_POS_SENTENCE_RE = re.compile(r"Position:\s*\d\s*of\s*5", re.I)


def position_bin(position: int, prompt_len: int) -> int:
    """1..5: which fifth of the rendered prompt the extraction position is in (same bins as `probe_check --target relpos`)."""
    return min(int(5 * position / prompt_len), 4) + 1


def position_sentence(position: int, prompt_len: int) -> str:
    return f"Position: {position_bin(position, prompt_len)} of 5."


def has_position_fact(text: str) -> bool:
    return _POS_SENTENCE_RE.search(text) is not None


def field_prefix(token_prefix_text: str, position: int | None, prompt_len: int | None, position_fact: bool) -> str:
    """The fact sentences that precede a field's prose: `Marked token: "X". ` (given) then optionally `Position: K of 5. `."""
    out = token_prefix_text
    if position_fact:
        assert position is not None and prompt_len, "--position-fact needs the position and prompt_len columns"
        out += position_sentence(position, prompt_len) + " "
    return out
