"""The HRM verbalizer/reconstructor: one frozen Qwen2.5-1.5B-Instruct base with
two PEFT LoRA adapters (`av`, `ar`; an optional frozen `av_ref` snapshot is
added for the RL KL penalty — see `train_rl.py`), plus small trainable
per-stream injection/reconstruction affine maps that are NOT part of any
LoRA adapter (they're new parameters, always trainable, saved/loaded
separately — see `save_extra_modules`/`load_extra_modules`).

Deviation from the original NLA (docs/hrm.md): AV and AR are Qwen2.5-1.5B,
not copies of the target model, and the AR is the FULL-DEPTH model (no layer
truncation) plus an affine head — the original NLA's `NLACriticModel`
truncates to K+1 layers, which has no well-defined analog for a recurrent
HRM target.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions

DEFAULT_VERBALIZER = "Qwen/Qwen2.5-1.5B-Instruct"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_verbalizer(
    model_name: str = DEFAULT_VERBALIZER,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    lora_r: int = 32,
    lora_alpha: int = 64,
) -> tuple[PeftModel, AutoTokenizer]:
    """Frozen base + two LoRA adapters ('av', 'ar'). `model.set_adapter(name)`
    switches which one is active for a forward pass; the base's own weights
    are never updated (`get_peft_model` freezes everything not matched by
    `target_modules`)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0, task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg, adapter_name="av")
    model.add_adapter("ar", lora_cfg)
    model.set_adapter("av")
    return model, tokenizer


def add_frozen_reference_adapter(model: PeftModel, source_adapter: str = "av", ref_name: str = "av_ref") -> None:
    """Clone `source_adapter`'s current weights into a new adapter that is
    never trained — the RL KL-penalty reference (`train_rl.py`), a frozen
    snapshot of the AV right after SFT. Must be called AFTER SFT, before RL."""
    cfg = model.peft_config[source_adapter]
    model.add_adapter(ref_name, cfg)
    src_sd = get_peft_model_state_dict_for(model, source_adapter)
    dst_sd = {k.replace(f".{source_adapter}.weight", f".{ref_name}.weight")
              .replace(f".{source_adapter}.", f".{ref_name}."): v for k, v in src_sd.items()}
    _missing, unexpected = model.load_state_dict(dst_sd, strict=False)
    assert not unexpected, f"unexpected keys loading {ref_name}: {unexpected}"
    for name, p in model.named_parameters():
        if f".{ref_name}." in name:
            p.requires_grad_(False)


def get_peft_model_state_dict_for(model: PeftModel, adapter_name: str) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if f".{adapter_name}." in k}


def _init_near_identity(linear: nn.Linear, d_in: int, d_out: int, init_scale: float, init_noise_std: float) -> None:
    """weight ~= init_scale * eye + noise. eye() only when square (d_in==d_out,
    the common case — Mimir's d_model happens to equal Qwen2.5-1.5B's hidden
    size); otherwise falls back to a scaled-orthogonal-ish init (eye of the
    non-square shape, i.e. the leading min(d_in,d_out)x min(d_in,d_out) block
    is identity) so the smoke tests can swap in a differently-sized verbalizer."""
    with torch.no_grad():
        eye = torch.zeros(d_out, d_in)
        k = min(d_in, d_out)
        eye[:k, :k] = torch.eye(k)
        linear.weight.copy_(init_scale * eye + init_noise_std * torch.randn(d_out, d_in))
        linear.bias.zero_()


class InjectionAdapter(nn.Module):
    """Per-stream affine map: Mimir's raw z (norm ~= sqrt(d_model) ~= 39.2,
    from HrmTextStack's un-weighted final RMSNorm), d_mimir-dim -> verbalizer
    embedding space, d_verbalizer-dim (equal for the default Qwen2.5-1.5B
    target, but kept as two separate dims for robustness). Near-identity
    init, scaled by `init_scale` (see docs/hrm.md — the original NLA's
    heuristic: init large, near the verbalizer's own ambient residual-stream
    scale, not its token-embedding scale)."""

    def __init__(self, d_mimir: int, d_verbalizer: int | None = None, init_scale: float = 1.0, init_noise_std: float = 0.01):
        super().__init__()
        d_verbalizer = d_mimir if d_verbalizer is None else d_verbalizer
        self.linear = nn.Linear(d_mimir, d_verbalizer, bias=True)
        _init_near_identity(self.linear, d_mimir, d_verbalizer, init_scale, init_noise_std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z.float())


class ReconHeads(nn.Module):
    """head_L, head_H: verbalizer final hidden state (d_verbalizer-dim) ->
    raw z_L / z_H space (d_mimir-dim). Same near-identity-scaled-affine init
    as InjectionAdapter; `init_scale` here should be small (Qwen's hidden-
    state norm >> Mimir's ~39.2) — train scripts compute it from
    norm_stats.json (see train_av_sft.py's `--head-init-scale`, defaulted
    from the injection scale's reciprocal)."""

    def __init__(self, d_mimir: int, d_verbalizer: int | None = None, init_scale: float = 0.02, init_noise_std: float = 0.001):
        super().__init__()
        d_verbalizer = d_mimir if d_verbalizer is None else d_verbalizer
        self.head_L = nn.Linear(d_verbalizer, d_mimir, bias=True)
        self.head_H = nn.Linear(d_verbalizer, d_mimir, bias=True)
        for head in (self.head_L, self.head_H):
            _init_near_identity(head, d_verbalizer, d_mimir, init_scale, init_noise_std)

    def forward_L(self, h: torch.Tensor) -> torch.Tensor:
        return self.head_L(h.float())

    def forward_H(self, h: torch.Tensor) -> torch.Tensor:
        return self.head_H(h.float())


def render_av_prompt(tokenizer, user_content: str, *, add_generation_prompt: bool = True) -> torch.Tensor:
    """Render a single-turn Qwen chat prompt -> input_ids [1, S]. Explicit
    `["input_ids"]` indexing: `apply_chat_template(tokenize=True)` returns a
    `BatchEncoding` (dict-like) on this repo's transformers version, not a
    bare list/tensor — see `nla/hrm/injection.py`'s note on the same gotcha."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True, add_generation_prompt=add_generation_prompt,
        return_tensors="pt", return_dict=True,
    )
    return enc["input_ids"]


def build_inputs_embeds(
    model: PeftModel,
    input_ids: torch.Tensor,
    z_L: torch.Tensor,
    z_H: torch.Tensor,
    inj_L: InjectionAdapter,
    inj_H: InjectionAdapter,
    inj_l_id: int, left_l_id: int, right_l_id: int,
    inj_h_id: int, left_h_id: int, right_h_id: int,
) -> torch.Tensor:
    """Embed input_ids, then overwrite the [L]/[H] marker positions with the
    (adapter-mapped) activation vectors. TWO separate calls to
    `inject_at_marked_positions` (one per marker) — see `nla/hrm/injection.py`
    module docstring for why two distinct markers instead of one repeated
    marker. z_L, z_H: [B, d_model] (one vector per sample — the row's gold or
    reconstructed stream state)."""
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids)
    vec_l = inj_L(z_L).to(embeds.dtype)
    vec_h = inj_H(z_H).to(embeds.dtype)
    embeds = inject_at_marked_positions(input_ids, embeds, vec_l, inj_l_id, left_l_id, right_l_id)
    embeds = inject_at_marked_positions(input_ids, embeds, vec_h, inj_h_id, left_h_id, right_h_id)
    return embeds


def load_sft_adapter(model: PeftModel, ckpt_dir: str, adapter_name: str) -> None:
    """Load a `train_av_sft.py`/`train_ar_sft.py` checkpoint's LoRA weights
    INTO the model's existing `adapter_name` slot (`save_extra_modules`'s
    companion on the LoRA side). `PeftModel.load_adapter` can only load into
    a NEW adapter name it invents from the checkpoint — `ckpt_dir` is that
    script's `--output` (PEFT nests under `{output}/{adapter_name}/`, only an
    adapter literally named "default" saves flat — see docs/hrm.md), so we
    load into a throwaway name, copy the state dict across, then drop it."""
    tmp_name = f"_load_tmp_{adapter_name}"
    model.load_adapter(f"{ckpt_dir}/{adapter_name}", adapter_name=tmp_name)
    src_sd = {k: v for k, v in model.state_dict().items() if f".{tmp_name}." in k}
    model.load_state_dict({k.replace(f".{tmp_name}.", f".{adapter_name}."): v for k, v in src_sd.items()}, strict=False)
    model.delete_adapter(tmp_name)


def load_rl_checkpoint(
    model: PeftModel, av_sft_ckpt: str, ar_sft_ckpt: str, d_mimir: int, d_verbalizer: int, device: str,
) -> tuple[InjectionAdapter, InjectionAdapter, ReconHeads]:
    """Load BOTH the av and ar LoRA adapters plus their extra modules
    (injection adapters from av_sft_ckpt, heads from ar_sft_ckpt) into
    `model`'s existing 'av'/'ar' adapter slots. Used by `train_rl.py` (then
    followed by `add_frozen_reference_adapter`) and `eval.py`/`judge.py`
    callers that need a trained AV+AR pair without any further training."""
    load_sft_adapter(model, av_sft_ckpt, "av")
    inj_L = InjectionAdapter(d_mimir, d_verbalizer).to(device)
    inj_H = InjectionAdapter(d_mimir, d_verbalizer).to(device)
    load_extra_modules(f"{av_sft_ckpt}/extra_modules.safetensors", inj_L, inj_H, ReconHeads(d_mimir, d_verbalizer))

    load_sft_adapter(model, ar_sft_ckpt, "ar")
    heads = ReconHeads(d_mimir, d_verbalizer).to(device)
    load_extra_modules(f"{ar_sft_ckpt}/extra_modules.safetensors",
                         InjectionAdapter(d_mimir, d_verbalizer), InjectionAdapter(d_mimir, d_verbalizer), heads)
    return inj_L, inj_H, heads


def save_extra_modules(path: str, inj_L: InjectionAdapter, inj_H: InjectionAdapter, heads: ReconHeads) -> None:
    sd = {}
    sd.update({f"inj_L.{k}": v for k, v in inj_L.state_dict().items()})
    sd.update({f"inj_H.{k}": v for k, v in inj_H.state_dict().items()})
    sd.update({f"heads.{k}": v for k, v in heads.state_dict().items()})
    save_file(sd, path)


def load_extra_modules(path: str, inj_L: InjectionAdapter, inj_H: InjectionAdapter, heads: ReconHeads) -> None:
    sd = load_file(path)
    inj_L.load_state_dict({k.removeprefix("inj_L."): v for k, v in sd.items() if k.startswith("inj_L.")})
    inj_H.load_state_dict({k.removeprefix("inj_H."): v for k, v in sd.items() if k.startswith("inj_H.")})
    heads.load_state_dict({k.removeprefix("heads."): v for k, v in sd.items() if k.startswith("heads.")})
