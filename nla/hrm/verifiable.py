"""Verifiable reward terms for RL, computed from the explanation text and the prompt (no model call).

The AV only sees the injected vectors, never the prompt text, so it can only produce
a correct marked-token quote or a prompt-grounded quote/name by actually reading the vector;
copying is impossible. Both terms reuse the eval-time checks (`eval.quote_match`, `eval.grounding`).

  token_correct   the first quoted `Marked token: "X"` equals the real marked token
  ungrounded      quoted spans (>= 3 chars) plus mid-sentence capitalised names NOT found in the prompt

Limitation: invented LOWERCASE nouns (owl, squirrel, cave) are not detected, only quotes and names.
"""

from nla.hrm.eval import grounding, position_match_fields, quote_match, quote_match_fields
from nla.hrm.recon import parse_single

UNGROUNDED_CAP = 10  # the penalty saturates at this many ungrounded items


def verifiable_terms(parsed: tuple[str, str], context_marked: str | None) -> dict:
    g = grounding(parsed, context_marked)
    ungrounded = (g["quoted_total"] - g["quoted_ok"] + g["caps_total"] - g["caps_ok"]) if g else 0
    return {"token_correct": quote_match(parsed, context_marked) is True, "ungrounded": ungrounded}


def verifiable_bonus(terms: dict, w_token: float, w_ground: float) -> float:
    """+w_token for a correct marked-token quote, minus w_ground * min(ungrounded, cap) / cap (so in [-w_ground, 0])."""
    return w_token * float(terms["token_correct"]) - w_ground * min(terms["ungrounded"], UNGROUNDED_CAP) / UNGROUNDED_CAP


def split_fact_terms(l_completion: str, h_completion: str, row: dict) -> dict | None:
    """Split AV: which facts each CALL stated correctly (marked token, position fifth). None for the last prompt position
    (both facts are trivial there, so no bonus). A malformed call gets no credit."""
    if row.get("position") is not None and row.get("prompt_len") and row["position"] >= row["prompt_len"] - 1:
        return None
    l_text, h_text = parse_single(l_completion, "L"), parse_single(h_completion, "H")
    fields = (l_text or "", h_text or "")
    tok = quote_match_fields(fields, row.get("context_marked"))
    pos = position_match_fields(fields, row)
    return {"L_tok": l_text is not None and tok[0] is True, "H_tok": h_text is not None and tok[1] is True,
            "L_pos": l_text is not None and pos[0] is True, "H_pos": h_text is not None and pos[1] is True}


def split_call_rewards(rewards: list[float], texts_L: list[str], texts_H: list[str], rows: list[dict],
                       group_size: int, w_fact: float) -> tuple[list[float], list[float], dict]:
    """Per-call rewards for the split AV: the shared pair reward plus w_fact for EACH fact (token, position) that the
    call itself stated correctly, for BOTH calls and BOTH facts (each call gets a fair chance at each fact: the point is
    to test whether a reward can extract what SFT could not). Returns (rewards_L, rewards_H, mean rates of the facts)."""
    rl, rh = list(rewards), list(rewards)
    counts = {"L_tok": 0, "H_tok": 0, "L_pos": 0, "H_pos": 0}
    n = 0
    for i, (tl, th) in enumerate(zip(texts_L, texts_H, strict=True)):
        t = split_fact_terms(tl, th, rows[i // group_size])
        if t is None:
            continue
        n += 1
        for k in counts:
            counts[k] += t[k]
        rl[i] += w_fact * (t["L_tok"] + t["L_pos"])
        rh[i] += w_fact * (t["H_tok"] + t["H_pos"])
    return rl, rh, {k: v / n for k, v in counts.items()} if n else {}
