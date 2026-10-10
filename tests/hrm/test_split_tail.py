from nla.hrm.split import content_mask


def test_content_mask_drops_the_template_tail_including_the_last_position():
    # prompt of 10 tokens, tail of 5: positions 5..9 are the tail
    assert content_mask([2, 4, 5, 9], [10, 10, 10, 10], 5) == [True, True, False, False]
    assert content_mask([2, 9], [10, 10], 0) == [True, True]        # 0 keeps everything
    assert content_mask([4, 20], [10, 30], 5) == [True, True]       # per-prompt lengths
