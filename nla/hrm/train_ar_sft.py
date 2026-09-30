"""AR-SFT: train the `ar` LoRA adapter + ReconHeads (head_L, head_H) to map a
per-field explanation string to its target stream vector. Format-only warm-up
— see `nla/hrm/build.py` module docstring for why "L field -> z_L, H field ->
z_H" here doesn't imply anything about real stream content (both explanations
are equally generic context descriptions at this stage; differentiation is
RL's job, see `train_rl.py`).

Loss: direction-only MSE (`nla.schema.normalize_activation` to sqrt(d_model),
same convention as the original NLA's critic SFT — `nla/loss.py:nla_critic_loss`),
independent of `nla/hrm/recon.py`'s shared_normalize (that needs BOTH z_L and
z_H simultaneously; AR-SFT rows only ever carry one target).
"""

import argparse
import math

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nla.hrm.devices import default_device, default_dtype
from nla.hrm.model import ReconHeads, load_verbalizer
from nla.schema import normalize_activation


class ArSftDataset(Dataset):
    def __init__(self, parquet_path: str):
        t = pq.read_table(parquet_path, columns=["prompt", "target_z", "head"])
        self.prompts = t.column("prompt").to_pylist()
        self.targets = t.column("target_z").to_pylist()
        self.heads = t.column("head").to_pylist()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> dict:
        return {"prompt": self.prompts[i], "target": self.targets[i], "head": self.heads[i]}


def collate(batch: list[dict], tokenizer, device: str):
    enc = tokenizer([b["prompt"] for b in batch], return_tensors="pt", padding=True,
                     add_special_tokens=False, truncation=True, max_length=512)
    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
    heads = [b["head"] for b in batch]
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), targets.to(device), heads


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--eval-parquet", default=None)
    p.add_argument("--verbalizer-model", default=None, help="defaults to nla.hrm.model.DEFAULT_VERBALIZER")
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-init-scale", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--output", required=True, help="output dir: adapter + extra_modules.safetensors")
    args = p.parse_args()

    from nla.hrm.model import DEFAULT_VERBALIZER
    verbalizer_model = args.verbalizer_model or DEFAULT_VERBALIZER
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    tokenizer.padding_side = "right"
    d_mimir = 1536  # Mimir's hidden_size — fixed, not re-derived per run
    heads = ReconHeads(d_mimir, model.config.hidden_size, init_scale=args.head_init_scale).to(args.device)

    # Only the 'ar' adapter + heads train here — freeze 'av' explicitly (belt
    # and suspenders; PEFT doesn't auto-freeze inactive adapters).
    for name, param in model.named_parameters():
        param.requires_grad_(".ar." in name)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train_ar_sft] {n_trainable:,} trainable params (ar LoRA + heads)")

    model.set_adapter("ar")
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    train_ds = ArSftDataset(args.train_parquet)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate(b, tokenizer, args.device))

    scale = math.sqrt(d_mimir)
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, attn, targets, heads_batch in tqdm(loader, desc=f"epoch {epoch}"):
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            h_last_all = out.hidden_states[-1]
            lengths = attn.sum(dim=1) - 1  # last real token index per row
            h_last = h_last_all[torch.arange(h_last_all.shape[0]), lengths]

            preds = torch.stack([
                heads.forward_L(h) if hd == "L" else heads.forward_H(h)
                for h, hd in zip(h_last, heads_batch, strict=True)
            ])
            pred_n = normalize_activation(preds, scale)
            target_n = normalize_activation(targets, scale)
            loss = ((pred_n - target_n) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            if step % args.log_every == 0:
                print(f"  step={step} loss={loss.item():.4f}")

    from pathlib import Path
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, selected_adapters=["ar"])
    from nla.hrm.model import save_extra_modules
    from nla.hrm.model import InjectionAdapter
    # AR-SFT doesn't train injection adapters (no injection in this phase) —
    # save zero-init placeholders so load_extra_modules' shape contract holds
    # for anything that loads this checkpoint's extra_modules alongside heads.
    save_extra_modules(f"{args.output}/extra_modules.safetensors",
                         InjectionAdapter(d_mimir, model.config.hidden_size),
                         InjectionAdapter(d_mimir, model.config.hidden_size), heads)
    print(f"saved 'ar' adapter + heads -> {args.output}")


if __name__ == "__main__":
    main()
