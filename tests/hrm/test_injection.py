"""Fast, no-network unit tests for two-marker injection.

Uses nla.injection.inject_at_marked_positions directly with synthetic
tensors (no real tokenizer/model needed) — the two-call pattern
nla/hrm/model.py:build_inputs_embeds uses.
"""

import torch

from nla.hrm.model import InjectionAdapter
from nla.injection import inject_at_marked_positions


def test_two_marker_injection_writes_correct_slots():
    # Sequence: [BOS, left_L, INJ_L, right_L, mid, left_H, INJ_H, right_H, EOS]
    inj_l_id, left_l_id, right_l_id = 100, 10, 11
    inj_h_id, left_h_id, right_h_id = 200, 20, 21
    input_ids = torch.tensor([[1, left_l_id, inj_l_id, right_l_id, 99, left_h_id, inj_h_id, right_h_id, 2]])
    d = 4
    embeds = torch.zeros(1, 9, d)
    vec_l = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    vec_h = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

    out = inject_at_marked_positions(input_ids, embeds, vec_l, inj_l_id, left_l_id, right_l_id)
    out = inject_at_marked_positions(input_ids, out, vec_h, inj_h_id, left_h_id, right_h_id)

    assert torch.equal(out[0, 2], vec_l[0])
    assert torch.equal(out[0, 6], vec_h[0])
    # everywhere else stays zero (untouched)
    for pos in (0, 1, 3, 4, 5, 7, 8):
        assert torch.equal(out[0, pos], torch.zeros(d))


def test_injection_rejects_neighbor_mismatch():
    # marker present but with the WRONG neighbors — must not count as a match.
    inj_id, left_id, right_id = 100, 10, 11
    input_ids = torch.tensor([[1, 99, inj_id, 99, 2]])  # neighbors are 99, not left_id/right_id
    embeds = torch.zeros(1, 5, 4)
    vec = torch.ones(1, 4)
    try:
        inject_at_marked_positions(input_ids, embeds, vec, inj_id, left_id, right_id)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "expected a RuntimeError: 0 valid matches found but 1 vector expected"


def test_injection_adapter_near_identity_init():
    d_mimir, d_verb = 6, 6
    adapter = InjectionAdapter(d_mimir, d_verb, init_scale=3.0, init_noise_std=0.0)
    z = torch.eye(d_mimir)
    out = adapter(z)
    assert torch.allclose(out, 3.0 * z, atol=1e-5)


def test_injection_adapter_nonsquare_dims():
    d_mimir, d_verb = 8, 5
    adapter = InjectionAdapter(d_mimir, d_verb, init_scale=1.0, init_noise_std=0.0)
    z = torch.randn(3, d_mimir)
    out = adapter(z)
    assert out.shape == (3, d_verb)


def test_swapped_marker_ids_put_each_stream_in_the_other_slot():
    import torch

    from nla.injection import inject_at_marked_positions

    # ids: [pad, L-marker(10), x, H-marker(20), pad]; neighbors (left, right) around each marker
    ids = torch.tensor([[1, 10, 2, 20, 3]])
    emb = torch.zeros(1, 5, 4)
    vec_l, vec_h = torch.full((1, 4), 7.0), torch.full((1, 4), 9.0)
    meta = (10, 1, 2, 20, 2, 3)  # id, left, right for L then H
    swapped = meta[3:] + meta[:3]
    out = inject_at_marked_positions(ids, emb, vec_l, swapped[0], swapped[1], swapped[2])
    out = inject_at_marked_positions(ids, out, vec_h, swapped[3], swapped[4], swapped[5])
    assert out[0, 3].tolist() == [7.0] * 4  # z_L's vector went to the H slot
    assert out[0, 1].tolist() == [9.0] * 4  # z_H's vector went to the L slot
