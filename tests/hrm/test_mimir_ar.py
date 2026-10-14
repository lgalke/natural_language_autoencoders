import torch
from peft import LoraConfig, get_peft_model
from transformers import HrmTextConfig, HrmTextForCausalLM

from nla.hrm.mimir_ar import mimir_ar_hidden
from nla.hrm.model import LORA_TARGET_MODULES


def _tiny():
    torch.manual_seed(0)
    cfg = HrmTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16, H_cycles=2, L_cycles=3, L_bp_cycles=[3, 3], embedding_scale=5.0,
                        pad_token_id=0, prefix_lm=True)
    base = HrmTextForCausalLM(cfg).float().eval()
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGET_MODULES, lora_dropout=0.0, task_type="CAUSAL_LM")
    return get_peft_model(base, lora, adapter_name="ar")


def test_readout_is_padding_invariant_and_bidirectional():
    model = _tiny()
    a = torch.tensor([[5, 6, 7, 8, 9]])
    b = torch.tensor([[11, 12, 13, 0, 0]])
    batch = torch.cat([a, b])
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    h = mimir_ar_hidden(model, batch, attn)
    assert h.shape == (2, 32)
    # a row padded to the right reads out exactly what it does alone
    alone = mimir_ar_hidden(model, b[:, :3], torch.ones(1, 3, dtype=torch.long))
    assert torch.allclose(h[1], alone[0], atol=1e-4)
    # the text is read as one block: changing the first token changes the last token's state
    a2 = a.clone()
    a2[0, 0] = 20
    assert not torch.allclose(mimir_ar_hidden(model, a2, torch.ones_like(a2))[0], h[0], atol=1e-5)


def test_only_lora_trains():
    model = _tiny()
    for n, p in model.named_parameters():
        p.requires_grad_(".ar." in n)
    ids = torch.tensor([[5, 6, 7, 8]])
    mimir_ar_hidden(model, ids, torch.ones_like(ids)).sum().backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads and all(".ar." in n for n in grads)
