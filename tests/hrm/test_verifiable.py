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


def test_split_fact_terms_and_call_rewards():
    from nla.hrm.recon import ReconWeights  # noqa: F401  (import check only)
    from nla.hrm.split_av import single_response
    from nla.hrm.verifiable import split_call_rewards, split_fact_terms

    row = {"context_marked": "a b ⟦ of⟧ c", "position": 12, "prompt_len": 20, "dataset": "d"}  # fifth 4
    l_ok = single_response('Marked token: "the". Position: 4 of 5. text', "L")      # position right, token wrong
    h_ok = single_response('Marked token: "of". Position: 1 of 5. text', "H")       # token right, position wrong
    t = split_fact_terms(l_ok, h_ok, row)
    assert t == {"L_tok": False, "H_tok": True, "L_pos": True, "H_pos": False}
    assert split_fact_terms(l_ok, "garbage", row) == {"L_tok": False, "H_tok": False, "L_pos": True, "H_pos": False}
    assert split_fact_terms(l_ok, h_ok, {**row, "position": 19}) is None            # last prompt position: no bonus

    rl, rh, rates = split_call_rewards([1.0, 1.0], [l_ok, "bad"], [h_ok, "bad"], [row], group_size=2, w_fact=0.5)
    assert rl == [1.5, 1.0] and rh == [1.5, 1.0]                                     # L: +0.5 position; H: +0.5 token
    assert rates["L_pos"] == 0.5 and rates["H_tok"] == 0.5 and rates["L_tok"] == 0.0
