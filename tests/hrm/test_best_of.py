"""Best-of-N selection in eval.generate_and_reconstruct, with fake model/tokenizer/heads (no weights needed)."""

import torch

import nla.hrm.model as hrm_model
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.eval import best_of_report, generate_and_reconstruct
from nla.hrm.recon import ReconWeights

# what the fake AR "reads" out of an explanation labelled mode=k: k=1 reproduces the gold vectors exactly
_TABLE = torch.tensor([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0]])


class FakeTok:
    padding_side = "left"
    pad_token_id = 0

    def apply_chat_template(self, msgs, **kw):
        n = len(msgs)
        return {"input_ids": torch.ones(n, 3, dtype=torch.long), "attention_mask": torch.ones(n, 3, dtype=torch.long)}

    def __call__(self, texts, **kw):
        ids = torch.tensor([[int(t.split("mode=")[1][0])] * 4 for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"<explanation>\nL: mode={int(r[0])}\nH: mode={int(r[0])}\n</explanation>" for r in ids]


class FakeModel:
    def __init__(self):
        self.calls = 0

    def set_adapter(self, name):
        pass

    def generate(self, inputs_embeds, attention_mask, do_sample=False, **kw):
        n = inputs_embeds.shape[0]
        mode = 0 if not do_sample else 1 + (self.calls % 2)
        self.calls += int(do_sample)
        return torch.full((n, 2), mode, dtype=torch.long)

    def __call__(self, input_ids, attention_mask, output_hidden_states=True, logits_to_keep=1):
        class Out:
            pass

        o = Out()
        o.hidden_states = [None, _TABLE[input_ids]]
        return o


class FakeHeads:
    def forward_L(self, h):
        return h

    def forward_H(self, h):
        return torch.roll(h, 1, dims=-1)


def _rows():
    z_L, z_H = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
    content = f"x {_INJECT_L_PLACEHOLDER} y {_INJECT_H_PLACEHOLDER}"
    return [{"prompt": [{"content": content}], "z_L": z_L, "z_H": z_H, "context_marked": "a ⟦b⟧ c", "dataset": "d"}
            for _ in range(3)]


def _run(best_of, monkeypatch):
    monkeypatch.setattr(hrm_model, "build_inputs_embeds", lambda model, ids, zl, zh, *a: torch.zeros(ids.shape[0], 3, 4))
    return generate_and_reconstruct(FakeModel(), FakeTok(), _rows(), "A", "B", None, None, FakeHeads(), (0, 0, 0, 0, 0, 0),
                                    "{explanation}", "cpu", max_new_tokens=4, batch_size=2, best_of=best_of,
                                    weights=ReconWeights())


def test_best_of_one_is_the_greedy_completion(monkeypatch):
    res = _run(1, monkeypatch)
    assert all("mode=0" in r["text"] and "bo" not in r for r in res)


def test_best_of_three_keeps_the_candidate_closest_to_gold(monkeypatch):
    res = _run(3, monkeypatch)
    for r in res:
        assert r["bo"]["chosen"] == 1 and "mode=1" in r["text"]
        assert r["bo"]["losses"][1] < r["bo"]["losses"][2] < r["bo"]["losses"][0]
        assert torch.allclose(r["z_L_hat"], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    best_of_report("t", [{**r, "row": {**r["row"], "is_last_prompt_pos": False}} for r in res])
