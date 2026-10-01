"""Does the verbalizer READ the injected vector? Teacher-forced likelihood test.

For each row we compute the AV's mean per-token NLL of the teacher explanation
(`L: <e0>\\nH: <e1>`) under three conditions:
  real      the row's own (z_L, z_H) injected
  shuffled  another row's vectors injected (rolled by one)
A verbalizer that uses the vector assigns the explanation a clearly LOWER NLL with
the real vector (gap = shuffled - real > 0). Gap ~ 0 means the explanations are
predicted from the prompt template and language prior alone: the vector is ignored.
Cheap (forward passes only, no generation) and independent of RL rewards.

Use rows the AV never trained on, e.g. the AR-SFT bucket's explained parquet:
    python -m nla.hrm.nll_check --parquet splits/ar_sft_explained.parquet \\
        --sidecar-from av_sft.parquet --av-ckpt ckpt/av_sft --limit 300
Compare checkpoints (ckpt/av_sft vs ckpt/rl_logr/final) by running it on each.
"""

import argparse

import pyarrow.parquet as pq
import torch

from nla.datagen.storage import LocalStorage
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER, wrap_lh_explanation
from nla.hrm.devices import default_device, default_dtype
from nla.hrm.model import (
    DEFAULT_VERBALIZER, InjectionAdapter, ReconHeads, build_inputs_embeds, load_extra_modules,
    load_sft_adapter, load_verbalizer,
)
from nla.hrm.sidecar import read_sidecar
from nla.hrm.train_av_sft import _build_example, _collate

_D_MIMIR = 1536


@torch.no_grad()
def mean_nll(model, examples, inj_L, inj_H, ids_meta, pad_id, device, z_override=None, batch_size=8) -> float:
    """Mean per-token NLL of the response over all examples. z_override: (zL_list, zH_list) to inject instead."""
    model.set_adapter("av")
    total, n_tok = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, labels, attn, z_L, z_H = _collate(batch, pad_id, device)
        if z_override is not None:
            z_L = torch.tensor(z_override[0][start : start + batch_size], dtype=torch.float32, device=device)
            z_H = torch.tensor(z_override[1][start : start + batch_size], dtype=torch.float32, device=device)
        embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
        logits = model(inputs_embeds=embeds, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                ignore_index=-100, reduction="sum")
        total += nll.item()
        n_tok += (tgt != -100).sum().item()
    return total / max(n_tok, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="explained parquet with z_L, z_H, api_explanation_0/1 "
                                                    "(e.g. splits/ar_sft_explained.parquet: rows the AV never trained on)")
    p.add_argument("--sidecar-from", required=True, help="a built av_sft/rl parquet: supplies markers and the actor template")
    p.add_argument("--av-ckpt", required=True, help="dir containing av/ and extra_modules.safetensors "
                                                    "(ckpt/av_sft, or ckpt/rl_logr/final)")
    p.add_argument("--verbalizer-model", default=DEFAULT_VERBALIZER)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    args = p.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    meta = read_sidecar(LocalStorage(), args.sidecar_from)
    tm = meta.tokens
    assert tm is not None, f"{args.sidecar_from} has no token metadata"
    ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
    actor_content = meta.prompt_templates["actor"].format(inj_L=_INJECT_L_PLACEHOLDER, inj_H=_INJECT_H_PLACEHOLDER)

    model, tok = load_verbalizer(args.verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    load_sft_adapter(model, args.av_ckpt, "av")
    inj_L, inj_H = InjectionAdapter(_D_MIMIR, d_verb).to(args.device), InjectionAdapter(_D_MIMIR, d_verb).to(args.device)
    load_extra_modules(f"{args.av_ckpt}/extra_modules.safetensors", inj_L, inj_H, ReconHeads(_D_MIMIR, d_verb))
    model.eval()

    t = pq.read_table(args.parquet, columns=["z_L", "z_H", "api_explanation_0", "api_explanation_1"])
    rows = t.to_pylist()[: args.limit]
    assert len(rows) >= 2, "need at least 2 rows for the shuffled condition"
    examples = [
        _build_example(tok, {"prompt": [{"content": actor_content}],
                             "response": wrap_lh_explanation(r["api_explanation_0"], r["api_explanation_1"]),
                             "z_L": r["z_L"], "z_H": r["z_H"]}, tm.injection_char_L, tm.injection_char_H)
        for r in rows
    ]
    rolled = (examples[1:] + examples[:1])
    shuf = ([e["z_L"] for e in rolled], [e["z_H"] for e in rolled])

    real = mean_nll(model, examples, inj_L, inj_H, ids_meta, tok.pad_token_id, args.device, None, args.batch_size)
    shuffled = mean_nll(model, examples, inj_L, inj_H, ids_meta, tok.pad_token_id, args.device, shuf, args.batch_size)
    print(f"checkpoint {args.av_ckpt}  rows={len(rows)}")
    print(f"  NLL/token  real vector:     {real:.4f}")
    print(f"  NLL/token  shuffled vector: {shuffled:.4f}")
    print(f"  gap (shuffled - real):      {shuffled - real:+.4f}   "
          f"({'vector is used' if shuffled - real > 0.01 else 'vector looks IGNORED'}; 0.01 nats/token = rule of thumb)")


if __name__ == "__main__":
    main()
