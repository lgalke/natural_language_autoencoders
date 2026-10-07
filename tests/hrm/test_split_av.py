import torch

import nla.hrm.model as hrm_model
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER, wrap_lh_explanation
from nla.hrm.eval import generate_and_reconstruct
from nla.hrm.recon import ReconWeights, parse_fields, parse_single
from nla.hrm.split_av import convert_rows, join_split, single_response, stream_tag, tag_content
from nla.hrm.train_rl import _mask_streams
from tests.hrm.test_best_of import _TABLE, FakeHeads  # noqa: F401  (the fake AR reads "mode=k" out of the field text)


def test_parse_single():
    assert parse_single("<explanation>\nL: hello there\n</explanation>", "L") == "hello there"
    assert parse_single("<explanation>\nH: hi\n</explanation>", "H") == "hi"
    assert parse_single("<explanation>\nL: hi\n</explanation>", "H") is None       # wrong header
    assert parse_single("<explanation>\nL: \n</explanation>", "L") is None          # empty
    assert parse_single("L: no tags", "L") is None
    assert parse_single("<explanation>\nL: first\nH: second\n</explanation>", "L") == "first"  # extra field cut


def test_join_split_roundtrips_through_parse_fields():
    joint = join_split(single_response("a b", "L"), single_response("c d", "H"))
    assert parse_fields(joint) == ("a b", "c d")
    assert join_split(single_response("a", "L"), "garbage") == ""
    assert parse_fields(join_split("", "")) is None


def test_convert_rows_zeroes_the_other_stream_and_tags_the_prompt():
    row = {"prompt": [{"role": "user", "content": "describe"}], "response": wrap_lh_explanation("LL", "HH"),
           "z_L": [1.0, 2.0], "z_H": [3.0, 4.0], "doc_id": "d"}
    out, bad = convert_rows([row, {**row, "response": "broken"}])
    assert bad == 1 and len(out) == 2
    l_row, h_row = out
    assert l_row["response"] == single_response("LL", "L") and l_row["z_H"] == [0.0, 0.0] and l_row["z_L"] == [1.0, 2.0]
    assert h_row["response"] == single_response("HH", "H") and h_row["z_L"] == [0.0, 0.0] and h_row["z_H"] == [3.0, 4.0]
    assert l_row["prompt"][0]["content"] == tag_content("describe", "L") and stream_tag("H") in h_row["prompt"][0]["content"]
    assert l_row["prompt"][0]["role"] == "user" and l_row["doc_id"] == "d"


def test_mask_streams():
    zl, zh = torch.ones(2, 3), 2 * torch.ones(2, 3)
    a, b = _mask_streams(zl, zh, "H")
    assert a.equal(zl) and b.sum() == 0
    a, b = _mask_streams(zl, zh, "L")
    assert a.sum() == 0 and b.equal(zh)
    a, b = _mask_streams(zl, zh, None)
    assert a.equal(zl) and b.equal(zh)


class _Tok:
    padding_side = "left"
    pad_token_id = 0

    def apply_chat_template(self, msgs, **kw):
        self.last_contents = [m[0]["content"] for m in msgs]
        n = len(msgs)
        return {"input_ids": torch.ones(n, 3, dtype=torch.long), "attention_mask": torch.ones(n, 3, dtype=torch.long)}

    def __call__(self, texts, **kw):
        ids = torch.tensor([[int(t.split("mode=")[1][0])] * 4 for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        out = []
        for r in ids:
            v = int(r[0])
            out.append(f"<explanation>\n{'L' if v < 10 else 'H'}: mode={v % 10}\n</explanation>")
        return out


class _Model:
    def __init__(self):
        self.calls = 0

    def set_adapter(self, name):
        pass

    def generate(self, inputs_embeds, attention_mask, **kw):
        v = 1 + 10 * (self.calls % 2)  # first call = L call (mode 1), second = H call (mode 1, H header)
        self.calls += 1
        return torch.full((inputs_embeds.shape[0], 2), v, dtype=torch.long)

    def __call__(self, input_ids, attention_mask, output_hidden_states=True, logits_to_keep=1):
        class Out:
            pass

        o = Out()
        o.hidden_states = [None, _TABLE[input_ids]]
        return o


def test_eval_split_path_masks_one_stream_per_call_and_scores_the_joined_pair(monkeypatch):
    seen = []

    def fake_embeds(model, ids, zl, zh, *a):
        seen.append((zl.abs().sum().item(), zh.abs().sum().item()))
        return torch.zeros(ids.shape[0], 3, 4)

    monkeypatch.setattr(hrm_model, "build_inputs_embeds", fake_embeds)
    content = f"x {_INJECT_L_PLACEHOLDER} y {_INJECT_H_PLACEHOLDER}"
    rows = [{"prompt": [{"content": content}], "z_L": [1.0, 0.0, 0.0, 0.0], "z_H": [0.0, 1.0, 0.0, 0.0],
             "context_marked": "a ⟦b⟧ c", "dataset": "d"} for _ in range(2)]
    tok = _Tok()
    res = generate_and_reconstruct(_Model(), tok, rows, "A", "B", None, None, FakeHeads(), (0, 0, 0, 0, 0, 0),
                                   "{explanation}", "cpu", max_new_tokens=4, batch_size=2, weights=ReconWeights(),
                                   split_av=True)
    assert seen[0][1] == 0 and seen[0][0] > 0      # L call: z_H zeroed
    assert seen[1][0] == 0 and seen[1][1] > 0      # H call: z_L zeroed
    assert stream_tag("H") in tok.last_contents[0]  # the second prompt was tagged for the H stream
    assert all(r["parsed"] == ("mode=1", "mode=1") for r in res)
    assert torch.allclose(res[0]["z_L_hat"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_position_fact_in_both_calls_and_mask_and_eval_parser():
    from nla.hrm.build import prefix_token_mask
    from nla.hrm.eval import position_match_fields

    row = {"prompt": [{"role": "user", "content": "describe"}],
           "response": wrap_lh_explanation('Marked token: "a". LL', 'Marked token: "b". HH'),
           "z_L": [1.0, 2.0], "z_H": [3.0, 4.0], "position": 12, "prompt_len": 20}
    out, _ = convert_rows([row], facts=True)
    assert out[0]["response"] == single_response('Marked token: "a". Position: 4 of 5. LL', "L")
    assert out[1]["response"] == single_response('Marked token: "b". Position: 4 of 5. HH', "H")
    # the weight mask covers the token span and the position span
    class Tok:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    resp = out[0]["response"]
    mask = prefix_token_mask(Tok(), resp, len(resp))
    spans = [m.span() for m in __import__("re").finditer(r'Marked token: "a"\.|Position: 4 of 5\.', resp)]
    assert sum(mask) == sum(b - a for a, b in spans)
    assert position_match_fields(('Position: 4 of 5. x', 'Position: 2 of 5. y'), row) == (True, False)
    assert position_match_fields(('no fact', 'Position: 4 of 5.'), row) == (None, True)
