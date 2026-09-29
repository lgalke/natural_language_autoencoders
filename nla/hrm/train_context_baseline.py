"""Context-only baseline: SAME architecture (Qwen2.5-1.5B + LoRA + ReconHeads)
as the real AR, trained SFT-only to predict z_L AND z_H directly from
`context_marked` (the raw prompt text, ⟦token⟧-marked) — no activation
vector is ever injected. The AV's explanations are only informative insofar
as they beat this: it answers "how much does knowing the text (without ever
seeing the vector) already predict the streams?" (see docs/hrm.md and
eval.py's `--baseline-metrics-json` comparison).

Reuses `recon.recon_loss` UNCHANGED — one forward pass over the context
predicts BOTH z_L and z_H (unlike AR-SFT's per-field rows), so the sum term
is a real, comparable target (gold s = z_L + z_H).
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.model import DEFAULT_VERBALIZER, ReconHeads, load_verbalizer, save_extra_modules
from nla.hrm.recon import ReconWeights, recon_loss

_D_MIMIR = 1536
_ADAPTER = "baseline"


def _load_rows(parquet_path: str) -> list[dict]:
    t = pq.read_table(parquet_path, columns=["context_marked", "z_L", "z_H"])
    return t.to_pylist()


def _collate(rows: list[dict], tokenizer, device: str):
    enc = tokenizer([r["context_marked"] for r in rows], return_tensors="pt", padding=True,
                     add_special_tokens=False, truncation=True, max_length=512)
    z_L = torch.tensor([r["z_L"] for r in rows], dtype=torch.float32)
    z_H = torch.tensor([r["z_H"] for r in rows], dtype=torch.float32)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), z_L.to(device), z_H.to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True, help="any split with context_marked+z_L+z_H (e.g. ar_sft/rl base rows)")
    p.add_argument("--verbalizer-model", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-init-scale", type=float, default=0.02)
    p.add_argument("--w-sum", type=float, default=1.0)
    p.add_argument("--w-comp", type=float, default=0.25)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    verbalizer_model = args.verbalizer_model or DEFAULT_VERBALIZER
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    weights = ReconWeights(w_sum=args.w_sum, w_comp=args.w_comp)

    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    tokenizer.padding_side = "right"
    model.add_adapter(_ADAPTER, model.peft_config["av"])
    heads = ReconHeads(_D_MIMIR, model.config.hidden_size, init_scale=args.head_init_scale).to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad_(f".{_ADAPTER}." in name)
    trainable = [pr for pr in model.parameters() if pr.requires_grad] + list(heads.parameters())
    print(f"[train_context_baseline] {sum(pr.numel() for pr in trainable):,} trainable params")
    model.set_adapter(_ADAPTER)
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    rows = _load_rows(args.train_parquet)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(rows)).tolist()
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"epoch {epoch}"):
            batch = [rows[i] for i in order[start : start + args.batch_size]]
            ids, attn, z_L, z_H = _collate(batch, tokenizer, args.device)
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            lengths = attn.sum(dim=1) - 1
            h_last = out.hidden_states[-1][torch.arange(ids.shape[0]), lengths]
            zL_hat, zH_hat = heads.forward_L(h_last), heads.forward_H(h_last)
            loss = recon_loss(zL_hat, zH_hat, z_L, z_H, weights)

            optim.zero_grad()
            loss.total.backward()
            optim.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  step={step} loss={loss.total.item():.4f} "
                      f"mse_sum={loss.mse_sum.item():.4f} mse_L={loss.mse_L.item():.4f} mse_H={loss.mse_H.item():.4f}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, selected_adapters=[_ADAPTER])
    from nla.hrm.model import InjectionAdapter
    save_extra_modules(f"{args.output}/extra_modules.safetensors",
                         InjectionAdapter(_D_MIMIR, model.config.hidden_size),
                         InjectionAdapter(_D_MIMIR, model.config.hidden_size), heads)
    print(f"saved '{_ADAPTER}' adapter + heads -> {args.output}")


if __name__ == "__main__":
    main()
