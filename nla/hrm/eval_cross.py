"""Cross-reconstruction matrix — the test for stream-specific content in the
AV's L/H fields. Freezes a trained AV, generates (L, H) field pairs on a
probe-train split and a probe-test split, then trains 6 FRESH, budget-matched
probes (new LoRA adapter + a single linear head each, same architecture/
hyperparameters as `train_ar_sft.py`, trained from scratch every time — never
reusing the real AR) for:

    L field  -> z_L      H field  -> z_L      joint(L+H) -> z_L
    L field  -> z_H      H field  -> z_H      joint(L+H) -> z_H

Report: a {z_L, z_H} x {L, H, joint} FVE matrix. Diagonal (L->z_L, H->z_H)
clearly above off-diagonal (L->z_H, H->z_L) means the fields carry
stream-specific content, not interchangeable generic context. The joint row
doubles as the fallback condition (docs/hrm.md) if L/H differentiation never
emerges from RL.
"""

import argparse
import json
import math

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.model import DEFAULT_VERBALIZER, load_rl_checkpoint, load_verbalizer
from nla.hrm.recon import parse_fields
from nla.hrm.sidecar import read_sidecar
from nla.datagen.storage import LocalStorage
from nla.schema import normalize_activation

_D_MIMIR = 1536


@torch.no_grad()
def _generate_fields(model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, ids_meta, device,
                       max_new_tokens, batch_size=16):
    from nla.hrm.model import build_inputs_embeds

    pairs = []
    for start in tqdm(range(0, len(rows), batch_size), desc="rollout"):
        batch_rows = rows[start : start + batch_size]
        contents = [r["prompt"][0]["content"].replace(_INJECT_L_PLACEHOLDER, inj_l_char)
                    .replace(_INJECT_H_PLACEHOLDER, inj_h_char) for r in batch_rows]
        tokenizer.padding_side = "left"
        enc = tokenizer.apply_chat_template(
            [[{"role": "user", "content": c}] for c in contents],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True,
        )
        input_ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        z_L = torch.tensor([r["z_L"] for r in batch_rows], dtype=torch.float32, device=device)
        z_H = torch.tensor([r["z_H"] for r in batch_rows], dtype=torch.float32, device=device)
        embeds = build_inputs_embeds(model, input_ids, z_L, z_H, inj_L, inj_H, *ids_meta)
        model.set_adapter("av")
        gen_ids = model.generate(inputs_embeds=embeds, attention_mask=attn, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for r, t in zip(batch_rows, texts, strict=True):
            parsed = parse_fields(t)
            if parsed is not None:
                pairs.append({"L": parsed[0], "H": parsed[1], "z_L": r["z_L"], "z_H": r["z_H"]})
    return pairs


def _probe_texts(pairs: list[dict], source: str) -> list[str]:
    if source == "joint":
        return [f"{p['L']} {p['H']}" for p in pairs]
    return [p[source] for p in pairs]


def _train_and_eval_probe(model, tokenizer, train_pairs, test_pairs, source: str, target: str, device,
                            epochs: int, lr: float, batch_size: int) -> dict:
    """A FRESH LoRA adapter ('probe') + a single Linear head, trained from
    scratch on (source-field text -> target-stream vector) pairs. Budget-
    matched across all 6 conditions (same epochs/lr/batch_size/architecture)
    so differences in FVE reflect the DATA, not probe capacity."""
    from peft import LoraConfig
    import torch.nn as nn

    adapter_name = f"probe_{source}_{target}"
    model.add_adapter(adapter_name, LoraConfig(
        r=16, lora_alpha=32, target_modules=model.peft_config["av"].target_modules, lora_dropout=0.0,
        task_type="CAUSAL_LM",
    ))
    for name, param in model.named_parameters():
        param.requires_grad_(f".{adapter_name}." in name)
    head = nn.Linear(model.config.hidden_size, _D_MIMIR, bias=True).to(device)
    trainable = [pr for pr in model.parameters() if pr.requires_grad] + list(head.parameters())
    model.set_adapter(adapter_name)
    optim = torch.optim.AdamW(trainable, lr=lr)

    train_texts = _probe_texts(train_pairs, source)
    train_targets = torch.tensor([p[target] for p in train_pairs], dtype=torch.float32)
    scale = math.sqrt(_D_MIMIR)

    model.train()
    for _epoch in range(epochs):
        order = torch.randperm(len(train_texts)).tolist()
        for start in range(0, len(train_texts), batch_size):
            idx = order[start : start + batch_size]
            texts = [train_texts[i] for i in idx]
            targets = train_targets[idx].to(device)
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                              truncation=True, max_length=512)
            ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            lengths = attn.sum(dim=1) - 1
            h_last = out.hidden_states[-1][torch.arange(ids.shape[0]), lengths]
            pred = head(h_last)
            loss = ((normalize_activation(pred, scale) - normalize_activation(targets, scale)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    test_texts = _probe_texts(test_pairs, source)
    test_targets = torch.tensor([p[target] for p in test_pairs], dtype=torch.float32).to(device)
    with torch.no_grad():
        enc = tokenizer(test_texts, return_tensors="pt", padding=True, add_special_tokens=False,
                          truncation=True, max_length=512)
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        lengths = attn.sum(dim=1) - 1
        h_last = out.hidden_states[-1][torch.arange(ids.shape[0]), lengths]
        pred = head(h_last)
        pred_n, target_n = normalize_activation(pred, scale), normalize_activation(test_targets, scale)
        mse = ((pred_n - target_n) ** 2).mean().item()
        mu = normalize_activation(train_targets.to(device), scale).mean(0, keepdim=True)
        mean_mse = ((target_n - mu) ** 2).mean().item()
        fve = 1.0 - mse / mean_mse if mean_mse > 1e-8 else float("nan")

    model.set_adapter("av")
    model.delete_adapter(adapter_name)
    return {"mse": mse, "mean_mse": mean_mse, "fve": fve, "n_train": len(train_pairs), "n_test": len(test_pairs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-train-parquet", required=True)
    p.add_argument("--probe-test-parquet", required=True)
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True, help="only used to load injection adapters; the real AR itself is unused here")
    p.add_argument("--verbalizer-model", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--probe-epochs", type=int, default=3)
    p.add_argument("--probe-lr", type=float, default=1e-4)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    verbalizer_model = args.verbalizer_model or DEFAULT_VERBALIZER
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    inj_L, inj_H, _heads = load_rl_checkpoint(model, args.av_ckpt, args.ar_ckpt, _D_MIMIR, d_verb, args.device)

    train_rows = pq.read_table(args.probe_train_parquet).to_pylist()
    test_rows = pq.read_table(args.probe_test_parquet).to_pylist()
    meta = read_sidecar(LocalStorage(), args.probe_train_parquet)
    tm = meta.tokens
    assert tm is not None
    ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)

    print("generating field pairs (train)...")
    train_pairs = _generate_fields(model, tokenizer, train_rows, tm.injection_char_L, tm.injection_char_H,
                                     inj_L, inj_H, ids_meta, args.device, args.max_new_tokens)
    print("generating field pairs (test)...")
    test_pairs = _generate_fields(model, tokenizer, test_rows, tm.injection_char_L, tm.injection_char_H,
                                    inj_L, inj_H, ids_meta, args.device, args.max_new_tokens)
    print(f"train pairs: {len(train_pairs)}  test pairs: {len(test_pairs)}")
    assert train_pairs and test_pairs, "no well-formed AV completions to probe — check the AV checkpoint / format rate"

    matrix: dict[str, dict[str, dict]] = {"z_L": {}, "z_H": {}}
    for target in ("z_L", "z_H"):
        for source in ("L", "H", "joint"):
            print(f"training probe: {source} -> {target} ...")
            matrix[target][source] = _train_and_eval_probe(
                model, tokenizer, train_pairs, test_pairs, source, target, args.device,
                args.probe_epochs, args.probe_lr, args.probe_batch_size,
            )

    with open(args.output, "w") as f:
        json.dump(matrix, f, indent=2)
    print("\nFVE matrix (rows=target, cols=source):")
    print(f"{'':>8}{'L':>10}{'H':>10}{'joint':>10}")
    for target in ("z_L", "z_H"):
        print(f"{target:>8}" + "".join(f"{matrix[target][s]['fve']:>10.3f}" for s in ("L", "H", "joint")))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
