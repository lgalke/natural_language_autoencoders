from nla.hrm.verifiable import UNGROUNDED_CAP, verifiable_bonus, verifiable_terms

CTX = 'Winston asked ⟦ about⟧ her relationship with Giselle. "I never saw him," she said.'


def _fields(a: str, b: str = "Narrative fiction.") -> tuple[str, str]:
    return a, b


def test_correct_token_and_grounded_quote():
    t = verifiable_terms(_fields('Marked token: "about". She says "I never saw him" to Winston, said Giselle.'), CTX)
    assert t["token_correct"] is True
    assert t["ungrounded"] == 0


def test_wrong_token_and_invented_quote_and_name():
    t = verifiable_terms(_fields('Marked token: "of". The text says "the brave knight rides" and then, Leo smiles.'), CTX)
    assert t["token_correct"] is False
    assert t["ungrounded"] == 2  # the invented quote and the invented name Leo


def test_bonus_and_cap():
    assert verifiable_bonus({"token_correct": True, "ungrounded": 0}, 0.5, 0.5) == 0.5
    assert verifiable_bonus({"token_correct": False, "ungrounded": 5}, 0.5, 0.5) == -0.25
    assert verifiable_bonus({"token_correct": False, "ungrounded": 99}, 0.5, 0.5) == -0.5
    assert verifiable_bonus({"token_correct": True, "ungrounded": 3}, 0.0, 0.0) == 0.0
    assert UNGROUNDED_CAP == 10
