import torch

from nla.hrm.model import last_real_index


def test_last_real_index_handles_left_and_right_padding():
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert last_real_index(right).tolist() == [2, 4]
    assert last_real_index(left).tolist() == [4, 4]


def test_legacy_readout_is_only_right_for_right_padding():
    right = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    left = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert last_real_index(right, "legacy").tolist() == [2, 4]
    assert last_real_index(left, "legacy").tolist() == [2, 4]       # row 0 should read 4: it reads a padding position
    assert last_real_index(left, "legacy")[0].item() != last_real_index(left, "last")[0].item()
