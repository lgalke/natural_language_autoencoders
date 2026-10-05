from nla.hrm.pair_check import opening, same_open_stats


def test_independent_pairs_match_expectation_and_anticorrelated_pairs_do_not():
    a, b = "The marked token is x.", "Narrative fiction genre: y."
    independent0 = [a, b] * 50
    independent1 = [a, a, b, b] * 25
    obs, exp = same_open_stats(independent0, independent1)
    assert abs(obs - exp) < 0.05
    anti1 = [b, a] * 50  # always the other format
    obs, exp = same_open_stats(independent0, anti1)
    assert obs == 0.0 and exp > 0.4


def test_opening_lowercases_and_truncates():
    assert opening("The Marked token is here") == "the marked token"
