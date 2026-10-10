import torch

from nla.hrm.ar_ceiling import ar_hats, make_variant, split_facts

TRUE_TOK, TRUE_POS = 'Marked token: "of".', "Position: 3 of 5."
GEN = 'Marked token: "the". Position: 1 of 5. The state encodes a story.'


def test_split_facts_and_variants():
    assert split_facts(GEN) == ('Marked token: "the".', "Position: 1 of 5.", "The state encodes a story.")
    assert split_facts("Just prose.") == (None, None, "Just prose.")
    assert make_variant("gen", GEN, TRUE_TOK, TRUE_POS) == GEN
    assert make_variant("oracle_token", GEN, TRUE_TOK, TRUE_POS) == 'Marked token: "of". Position: 1 of 5. The state encodes a story.'
    assert make_variant("oracle_position", GEN, TRUE_TOK, TRUE_POS) == 'Marked token: "the". Position: 3 of 5. The state encodes a story.'
    assert make_variant("oracle_facts", GEN, TRUE_TOK, TRUE_POS) == 'Marked token: "of". Position: 3 of 5. The state encodes a story.'
    assert make_variant("facts_only_gen", GEN, TRUE_TOK, TRUE_POS) == 'Marked token: "the". Position: 1 of 5.'
    assert make_variant("facts_only_oracle", GEN, TRUE_TOK, TRUE_POS) == 'Marked token: "of". Position: 3 of 5.'
    assert make_variant("prose_only", GEN, TRUE_TOK, TRUE_POS) == "The state encodes a story."
    # a dump without position facts: the oracle does not invent one
    assert make_variant("oracle_facts", 'Marked token: "the". Prose.', TRUE_TOK, TRUE_POS) == 'Marked token: "of". Prose.'


class _Tok:
    def __init__(self):
        self.padding_side = "right"

    def __call__(self, texts, **kw):
        lens = [len(t.split()) for t in texts]
        m = max(lens)
        ids, attn = [], []
        for t, n in zip(texts, lens, strict=True):
            row = [int(w[1:]) if w.startswith("t") else 0 for w in t.split()]   # words "t5" -> id 5
            pad = [0] * (m - n)
            ids.append(pad + row if self.padding_side == "left" else row + pad)
            attn.append([0] * (m - n) + [1] * n if self.padding_side == "left" else [1] * n + [0] * (m - n))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}


class _Model:
    def set_adapter(self, name):
        pass

    def __call__(self, input_ids, attention_mask, output_hidden_states=True, logits_to_keep=1):
        class Out:
            pass

        o = Out()
        o.hidden_states = [None, input_ids.float().unsqueeze(-1)]   # hidden state = the token id
        return o


class _Heads:
    def forward_L(self, h):
        return h

    def forward_H(self, h):
        return h


def test_ar_hats_reads_the_last_token_whatever_the_padding_side_and_legacy_misreads_left_padding():
    tmpl = "{explanation}"
    fl = ["t1 t2 t3", "t4 t5 t6 t7 t8"]
    fh = ["t1 t2 t3", "t4 t5 t6 t7 t8"]
    for side in ("right", "left"):
        zl, _ = ar_hats(_Model(), _Tok(), _Heads(), tmpl, fl, fh, "cpu", "last", 2, side)
        assert zl[:, 0].tolist() == [3.0, 8.0]
    zl, _ = ar_hats(_Model(), _Tok(), _Heads(), tmpl, fl, fh, "cpu", "legacy", 2, "left")
    assert zl[0, 0].item() != 3.0     # the short row reads the wrong position under left padding
