"""Shared Mimir (HRM-Text) utilities: loading, PrefixLM rendering, and the L/H
recurrent-state capture hook. Single source of truth reused by `stage0_hrm.py`
(extraction) and `judge.py` (patch-back eval) — both MUST agree on how the
prompt is rendered and where the hook fires, or the judge silently re-derives
a different z than what was stored (see docs/hrm.md).

Recurrence schedule (`transformers.models.hrm_text.modeling_hrm_text`,
`HrmTextModel.forward`): for `h` in 1..H_cycles: run L_module `L_cycles` times
on `z_L + z_H`, then run H_module once on `z_H + z_L`. Mimir-v1.5's schedule is
H_cycles=2, L_cycles=3 → L L L H L L L H. `L_module`/`H_module` are each a full
`HrmTextStack` (`num_layers_per_stack` blocks + a final RMSNorm) — hooking the
stack's forward (not individual layers) gives exactly the states in
HRM-Interp's `CycleStateCollector` ("L_out@h.l", "H_out@h").

Hook site (fixed by design, see docs/hrm.md): just before the SECOND H
application —
    z_L = L_out@2.3   (the 6th L call: last L application of H-cycle 2)
    z_H = H_out@1     (the 1st H application)
    s   = H_in@2 = z_L + z_H   (what H's second application actually reads)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MIMIR = "danish-foundation-models/DFM-Mimir-v1.5"


def load_mimir(
    model_name: str = DEFAULT_MIMIR,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load Mimir + tokenizer. `attn_implementation="sdpa"` — Mimir's PrefixLM 4-D
    mask overlay is not representable by flash-attention (see
    `HrmTextPreTrainedModel._check_and_adjust_attn_implementation`)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-pad: HrmStreamCapture slices [:, :length]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation=attn_implementation
    ).to(device).eval()
    return model, tokenizer


@dataclass
class RenderedBatch:
    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, S]
    token_type_ids: torch.Tensor  # [B, S] — 1 = bidirectional prompt, 0 = padding
    lengths: list[int]  # non-pad length per row
    rendered_texts: list[str]


def render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    """Single-turn Mimir chat rendering (text only, BOS included as text)."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]


def render_and_encode_batch(tokenizer, prompts: list[str], device: str = "cpu") -> RenderedBatch:
    """Render each prompt as a single-turn Mimir chat prompt and batch-encode it.

    The WHOLE rendered prompt (including <bos> and template scaffolding) is
    marked bidirectional (`token_type_ids=1`) — Mimir was pretrained only on
    (instruction, response) pairs with the instruction attended to
    bidirectionally end-to-end (see hrm-text-rlvr/train_grpo.py's
    `render_and_encode` and CLAUDE.md's "prefix_lm" note); there is no
    narrower "user content only" bidirectional span.

    `add_special_tokens=False`: the chat template already renders <bos> as
    text; re-tokenizing with the default would double it (prompt_templates.md
    "Double BOS on re-encode").
    """
    texts = render_prompts(tokenizer, prompts)
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    for row in input_ids.tolist():
        n_bos = row.count(tokenizer.bos_token_id)
        assert n_bos == 1, f"expected exactly one BOS per prompt, got {n_bos}: {row[:8]}"
        assert row[0] == tokenizer.bos_token_id, f"BOS must be the first token, got {row[:4]}"
    token_type_ids = attention_mask.clone()  # whole prompt bidirectional; padding stays 0
    lengths = attention_mask.sum(dim=1).tolist()
    return RenderedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        lengths=[int(x) for x in lengths],
        rendered_texts=texts,
    )


class HrmStreamCapture:
    """Forward-hook context manager capturing z_L, z_H, and H's 2nd-cycle input.

    Mirrors HRM-Interp's `CycleStateCollector` hook wiring (`hrm-interp/linear_probes.py`)
    but only tracks the three sites this project needs, plus a free correctness
    check: H's 2nd-cycle input must equal z_L + z_H exactly (that's literally how
    `HrmTextModel.forward` computes it — `H_module(hidden_states_high_cycle +
    hidden_states_low_cycle, ...)`).

    Usage:
        with HrmStreamCapture(model) as cap:
            model(**batch)
        cap.verify_call_counts()
        cap.verify_sum_identity()
        z_L, z_H = cap.z_L, cap.z_H   # each [B, S, d_model], float32
    """

    def __init__(self, model, target_h: int = 2):
        self.inner = model.model  # HrmTextModel
        self.L_cycles = self.inner.config.L_cycles
        self.H_cycles = self.inner.config.H_cycles
        assert self.H_cycles >= target_h, (
            f"hook targets H-application {target_h}, but config.H_cycles={self.H_cycles}"
        )
        self.target_h = target_h
        self.z_L: torch.Tensor | None = None
        self.z_H: torch.Tensor | None = None
        self.h_in_target: torch.Tensor | None = None
        self._l_counter = 0
        self._h_counter = 0
        self._handles = [
            self.inner.register_forward_pre_hook(self._on_pass_start),
            self.inner.L_module.register_forward_hook(self._on_L),
            self.inner.H_module.register_forward_pre_hook(self._on_H_pre),
            self.inner.H_module.register_forward_hook(self._on_H_post),
        ]

    def _on_pass_start(self, _module, _args):
        self._l_counter = 0
        self._h_counter = 0

    def _on_L(self, _module, _inputs, output):
        h, l_step = self._l_counter // self.L_cycles + 1, self._l_counter % self.L_cycles + 1
        if h == self.target_h and l_step == self.L_cycles:
            self.z_L = output.detach().float()
        self._l_counter += 1

    def _on_H_pre(self, _module, args):
        h = self._h_counter + 1
        if h == self.target_h:
            self.h_in_target = args[0].detach().float()

    def _on_H_post(self, _module, _inputs, output):
        h = self._h_counter + 1
        if h == self.target_h - 1:
            self.z_H = output.detach().float()
        self._h_counter += 1

    def verify_call_counts(self) -> None:
        expected_l = self.L_cycles * self.H_cycles
        assert self._l_counter == expected_l, (
            f"L_module fired {self._l_counter} times, expected {expected_l} "
            f"(L_cycles={self.L_cycles} × H_cycles={self.H_cycles})"
        )
        assert self._h_counter == self.H_cycles, (
            f"H_module fired {self._h_counter} times, expected H_cycles={self.H_cycles}"
        )
        assert self.z_L is not None and self.z_H is not None and self.h_in_target is not None, (
            "hook did not fire at the target site — target_h out of range for this schedule"
        )

    def verify_sum_identity(self, atol: float = 1e-3, rtol: float = 1e-3) -> None:
        """z_L + z_H must equal H's target-cycle input EXACTLY (same tensor,
        modulo dtype-cast float32 rounding) — that's how HrmTextModel computes it."""
        assert self.z_L is not None and self.z_H is not None and self.h_in_target is not None, (
            "call verify_call_counts() first"
        )
        s = self.z_L + self.z_H
        assert torch.allclose(s, self.h_in_target, atol=atol, rtol=rtol), (
            f"H_in@{self.target_h} != z_L + z_H (max abs diff="
            f"{(s - self.h_in_target).abs().max().item():.6g}). Hook wiring or "
            f"call-count assumption is wrong for this model config."
        )

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self) -> "HrmStreamCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def context_marked(tokenizer, input_ids: torch.Tensor, position: int, length: int) -> str:
    """Human-readable rendering of the prompt with the target token wrapped in
    ⟦ ⟧ — used for the API explainer prompt and the context-only baseline.
    Decoded piecewise (not re-tokenized) so merge/BPE boundaries never shift
    the marked token off its actual position."""
    ids = input_ids.tolist()
    before = tokenizer.decode(ids[:position], skip_special_tokens=False)
    target = tokenizer.decode(ids[position : position + 1], skip_special_tokens=False)
    after = tokenizer.decode(ids[position + 1 : length], skip_special_tokens=False)
    return f"{before}⟦{target}⟧{after}"
