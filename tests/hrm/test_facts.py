from nla.hrm.facts import field_prefix, has_position_fact, position_bin, position_sentence
from nla.hrm.split_av import add_position_fact


def test_position_bins_and_sentence():
    assert [position_bin(p, 100) for p in (0, 19, 20, 59, 60, 99)] == [1, 1, 2, 3, 4, 5]
    assert position_sentence(184, 194) == "Position: 5 of 5."


def test_field_prefix_and_no_double_insertion():
    assert field_prefix('Marked token: "of". ', None, None, False) == 'Marked token: "of". '
    assert field_prefix('Marked token: "of". ', 12, 20, True) == 'Marked token: "of". Position: 4 of 5. '
    already = 'Marked token: "of". Position: 4 of 5. prose'
    assert has_position_fact(already) and add_position_fact(already, 12, 20) == already
    assert add_position_fact('Marked token: "of". prose', 12, 20) == 'Marked token: "of". Position: 4 of 5. prose'
