"""Mimir (HRM-Text) as the reconstructor (AR) backbone.

The Qwen AR has to learn from scratch how text becomes a Mimir state; Mimir was pretrained to do exactly that on the
original text. Here the AR is a LoRA copy of Mimir that reads the critic prompt (`... <summary>`), with the whole text
under bidirectional attention (`token_type_ids = attention_mask`, as Mimir sees a rendered prompt) and the final-H output at
the last real token feeding the same `ReconHeads`. Only the AR changes: the AV, the reward, the sidecar and the judge are
untouched.

Differences to the Qwen AR worth knowing:
  * the readout is `last_hidden_state` of the inner `HrmTextModel` (the H output after the last cycle), not
    `hidden_states[-1]`, whose meaning under the recurrent forward is not guaranteed;
  * `config.L_bp_cycles` truncates gradients through the early L cycles (only the last ones carry gradient), so LoRA on the
    shared L/H stacks is trained through the trailing cycles only.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.hrm.model import LORA_TARGET_MODULES, ReconHeads, last_real_index, load_extra_modules, load_sft_adapter


def load_mimir_ar(model_name: str, device: str = "cpu", torch_dtype: torch.dtype = torch.bfloat16,
                  lora_r: int = 32, lora_alpha: int = 64):
    """Mimir with one trainable LoRA adapter named 'ar'; right-padding tokenizer (the readout assumes attended tokens)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, attn_implementation="sdpa").to(device)
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGET_MODULES, lora_dropout=0.0,
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg, adapter_name="ar")
    model.set_adapter("ar")
    return model, tokenizer


def mimir_ar_hidden(model, input_ids: torch.Tensor, attn: torch.Tensor, readout: str = "last") -> torch.Tensor:
    """[B, d] hidden state of the last attended token (final H output), all text bidirectional."""
    inner = model.get_base_model().model
    out = inner(input_ids=input_ids, attention_mask=attn, token_type_ids=attn)
    h = out.last_hidden_state
    return h[torch.arange(h.shape[0], device=h.device), last_real_index(attn, readout)]


def load_mimir_ar_checkpoint(model, ckpt_dir: str, d_mimir: int, device: str) -> ReconHeads:
    """Load a `train_ar_sft --backbone mimir` checkpoint (adapter + heads)."""
    from nla.hrm.model import InjectionAdapter
    load_sft_adapter(model, ckpt_dir, "ar")
    d = model.config.hidden_size
    heads = ReconHeads(d_mimir, d).to(device)
    load_extra_modules(f"{ckpt_dir}/extra_modules.safetensors", InjectionAdapter(d_mimir, d), InjectionAdapter(d_mimir, d), heads)
    return heads
