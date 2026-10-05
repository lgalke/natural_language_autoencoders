"""Verifiable reward terms for RL, computed from the explanation text and the prompt (no model call).

The AV only sees the injected vectors, never the prompt text, so it can only produce
a correct marked-token quote or a prompt-grounded quote/name by actually reading the vector;
copying is impossible. Both terms reuse the eval-time checks (`eval.quote_match`, `eval.grounding`).

  token_correct   the first quoted `Marked token: "X"` equals the real marked token
  ungrounded      quoted spans (>= 3 chars) plus mid-sentence capitalised names NOT found in the prompt

Limitation: invented LOWERCASE nouns (owl, squirrel, cave) are not detected, only quotes and names.
"""

from nla.hrm.eval import grounding, quote_match

UNGROUNDED_CAP = 10  # the penalty saturates at this many ungrounded items


def verifiable_terms(parsed: tuple[str, str], context_marked: str | None) -> dict:
    g = grounding(parsed, context_marked)
    ungrounded = (g["quoted_total"] - g["quoted_ok"] + g["caps_total"] - g["caps_ok"]) if g else 0
    return {"token_correct": quote_match(parsed, context_marked) is True, "ungrounded": ungrounded}


def verifiable_bonus(terms: dict, w_token: float, w_ground: float) -> float:
    """+w_token for a correct marked-token quote, minus w_ground * min(ungrounded, cap) / cap (so in [-w_ground, 0])."""
    return w_token * float(terms["token_correct"]) - w_ground * min(terms["ungrounded"], UNGROUNDED_CAP) / UNGROUNDED_CAP
