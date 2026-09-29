"""Mimir rendering/hook-capture tests — need the real tokenizer/model
(network or local HF cache), so every test here skips cleanly if that's
unavailable rather than failing CI in an offline environment."""

import pytest

transformers = pytest.importorskip("transformers")


def _try_load_tokenizer():
    try:
        from transformers import AutoTokenizer

        from nla.hrm.mimir import DEFAULT_MIMIR
        return AutoTokenizer.from_pretrained(DEFAULT_MIMIR)
    except Exception as e:  # noqa: BLE001 — any load failure means "skip", not "fail"
        pytest.skip(f"Mimir tokenizer unavailable ({e.__class__.__name__}: {e})")


def test_render_and_encode_batch_single_bos_and_token_type_ids():
    tokenizer = _try_load_tokenizer()
    from nla.hrm.mimir import render_and_encode_batch

    batch = render_and_encode_batch(tokenizer, ["What is 2+2?", "Hvad er hovedstaden i Danmark, og hvorfor?"])
    for row in batch.input_ids.tolist():
        assert row.count(tokenizer.bos_token_id) == 1
        assert row[0] == tokenizer.bos_token_id
    # token_type_ids == attention_mask (whole prompt bidirectional, padding excluded)
    assert (batch.token_type_ids == batch.attention_mask).all()


def test_hrm_stream_capture_call_counts_and_sum_identity():
    tokenizer = _try_load_tokenizer()
    try:
        import torch

        from nla.hrm.mimir import DEFAULT_MIMIR, HrmStreamCapture, load_mimir, render_and_encode_batch
        model, tokenizer = load_mimir(DEFAULT_MIMIR, device="cpu", torch_dtype=torch.float32)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Mimir model unavailable ({e.__class__.__name__}: {e})")

    batch = render_and_encode_batch(tokenizer, ["What is 2+2?"])
    with HrmStreamCapture(model) as cap:
        with torch.no_grad():
            model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                  token_type_ids=batch.token_type_ids, use_cache=False)
    cap.verify_call_counts()  # raises if L/H didn't fire L_cycles*H_cycles / H_cycles times
    cap.verify_sum_identity()  # raises if H_in@2 != z_L + z_H
    d = model.config.hidden_size
    assert cap.z_L.shape == (1, batch.input_ids.shape[1], d)
    assert cap.z_H.shape == (1, batch.input_ids.shape[1], d)
