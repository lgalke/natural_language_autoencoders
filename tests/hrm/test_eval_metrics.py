"""Fast unit tests for eval.py's text-level faithfulness helper."""

from nla.hrm.eval import marked_token, quote_match

CTX = "<bos><|turn>user\nA grandfather eats 5 burritos per⟦ day⟧, his partner eats 4<turn|>"


def test_marked_token():
    assert marked_token(CTX) == " day"
    assert marked_token("no marker") is None
    assert marked_token(None) is None


def test_quote_match_true_false_none():
    right = ('The marked token "day" is a time unit.', "H text")
    wrong = ('The marked token "wiggled" is a verb.', "H text")
    none = ("It describes a story.", "H text")
    assert quote_match(right, CTX) is True
    assert quote_match(wrong, CTX) is False
    assert quote_match(none, CTX) is None
    assert quote_match(None, CTX) is None
    assert quote_match(right, "no marker") is None


def test_quote_match_curly_quotes_case_and_second_field():
    fields = ("no quote here", "Marked token: “Day” completes the phrase")
    assert quote_match(fields, CTX) is True


def test_token_prefix_roundtrips_through_quote_match():
    from nla.hrm.build import token_prefix

    for ctx in (CTX, "<bos>text⟦\n⟧more", "<bos>x⟦ the⟧y"):
        pre = token_prefix(ctx)
        assert pre.startswith('Marked token: "')
        assert quote_match((pre + "rest of the explanation", "H"), ctx) is True
    assert token_prefix("no marker") == ""
