"""AR ceiling tests: how much reconstruction (FVE) would a better verbalizer buy? No generation, one AR forward per variant.

    python -m nla.hrm.ar_ceiling --dump samples_split_pos_final.jsonl \\
        --eval-parquet iid=eval_iid_clean.parquet ood=eval_ood.parquet \\
        --ar-ckpt ckpt/rl_split_pos/final --norm-stats-json norm_stats.json --output ceiling_e12.json

The AR reads the explanation texts of a dump (`eval --dump-samples`) and predicts z_L / z_H; the gold vectors come from the
eval parquets (matched by `context_marked`). Variants of the SAME generated texts:
  gen_last          as generated, AR reads the last attended token (correct readout)
  gen_legacy        as generated, legacy readout (attn.sum-1 on left-padded batches of 16 rows: reproduces the eval FVE of runs
                    before 2026-10-11 and shows what the misaligned readout cost; needs --batch-size to equal the eval's)
  oracle_token      the stated `Marked token: "X".` replaced by the TRUE token in both fields, prose unchanged
  oracle_position   the stated `Position: K of 5.` replaced by the TRUE fifth (position-fact dumps only)
  oracle_facts      both replaced by the truth: the FVE a perfect token/position extraction would give with this prose
  facts_only_gen    only the generated fact sentences, no prose
  facts_only_oracle only the TRUE fact sentences, no prose (the ceiling of "facts only")
  prose_only        the generated prose without the fact sentences
Paired differences to gen_last (95% bootstrap over rows) are printed per split, all rows and non-last rows.
Caveat: the AR was trained on the generated style; oracle texts are slightly out of distribution (the fact sentences are
well-formed, so the shift is small), and the AR was trained with the legacy readout in runs before 2026-10-11, so
gen_last may be BELOW gen_legacy for such checkpoints: compare variants within one readout.
"""

import argparse
import json
import re

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.datagen.storage import LocalStorage
from nla.hrm.bootstrap import interval
from nla.hrm.build import token_prefix
from nla.hrm.devices import default_device, default_dtype
from nla.hrm.model import DEFAULT_VERBALIZER, last_real_index, load_rl_checkpoint, load_verbalizer
from nla.hrm.recon import ReconWeights, recon_loss
from nla.hrm.sidecar import read_sidecar
from nla.hrm.split_av import position_bin

_FACT_RE = re.compile(r'\s*(Marked token:\s*"[^"]*"\.)?\s*(Position:\s*\d\s*of\s*5\.)?\s*(.*)$', re.S)


def split_facts(text: str) -> tuple[str | None, str | None, str]:
    """(token fact sentence or None, position fact sentence or None, prose) of a generated field."""
    m = _FACT_RE.match(text)
    return (m.group(1), m.group(2), m.group(3).strip())


def _join(*parts: str | None) -> str:
    return " ".join(p for p in parts if p)


def make_variant(name: str, text: str, true_token: str, true_pos: str) -> str:
    tok, pos, prose = split_facts(text)
    if name == "gen":
        return text
    if name == "oracle_token":
        return _join(true_token, pos, prose)
    if name == "oracle_position":
        return _join(tok, true_pos if pos else None, prose)
    if name == "oracle_facts":
        return _join(true_token, true_pos if pos else None, prose)
    if name == "facts_only_gen":
        return _join(tok, pos) or text
    if name == "facts_only_oracle":
        return _join(true_token, true_pos if pos else None)
    if name == "prose_only":
        return prose or text
    raise ValueError(name)


@torch.no_grad()
def ar_hats(model, tokenizer, heads, critic_template, fields_l, fields_h, device, readout="last", batch_rows=16,
            padding_side="right"):
    """AR predictions (zL_hat, zH_hat) [N, d] for lists of L and H field texts, in batches of `batch_rows` rows (L and H
    texts of a batch are tokenized together, as in eval.py)."""
    model.set_adapter("ar")
    tokenizer.padding_side = padding_side
    out_l, out_h = [], []
    for start in range(0, len(fields_l), batch_rows):
        l_texts = [critic_template.format(explanation=t) for t in fields_l[start:start + batch_rows]]
        h_texts = [critic_template.format(explanation=t) for t in fields_h[start:start + batch_rows]]
        enc = tokenizer(l_texts + h_texts, return_tensors="pt", padding=True, add_special_tokens=False,
                        truncation=True, max_length=512)
        ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, logits_to_keep=1)
        h = out.hidden_states[-1][torch.arange(ids.shape[0]), last_real_index(attn, readout)]
        n = len(l_texts)
        out_l.append(heads.forward_L(h[:n]).float().cpu())
        out_h.append(heads.forward_H(h[n:]).float().cpu())
    return torch.cat(out_l), torch.cat(out_h)


def row_fve(zl_hat, zh_hat, gold_l, gold_h, weights, mean_mse) -> np.ndarray:
    rows = []
    for i in range(len(zl_hat)):
        loss = recon_loss(zl_hat[i:i + 1], zh_hat[i:i + 1], gold_l[i:i + 1], gold_h[i:i + 1], weights)
        f = loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"])
        rows.append([f["fve_sum"], f["fve_L"], f["fve_H"]])
    return np.array(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--eval-parquet", nargs="+", required=True, help="split=eval.parquet (same files as the eval)")
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--norm-stats-json", default="norm_stats.json")
    p.add_argument("--verbalizer-model", default=DEFAULT_VERBALIZER)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--batch-size", type=int, default=16, help="rows per AR batch (the eval's --batch-size, for gen_legacy)")
    p.add_argument("--n-boot", type=int, default=3000)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    mean_mse = json.load(open(args.norm_stats_json))["mean_mse"]
    gold, meta = {}, None
    for spec in args.eval_parquet:
        split, path = spec.split("=", 1)
        meta = meta or read_sidecar(LocalStorage(), path)
        for r in pq.read_table(path).to_pylist():
            gold[(split, r["context_marked"])] = r
    critic_template = meta.prompt_templates["critic"]

    rows = [json.loads(line) for line in open(args.dump, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("L_field") and r.get("H_field") and (r["split"], r["context_marked"]) in gold]
    has_pos = np.mean([split_facts(r["L_field"])[1] is not None for r in rows]) > 0.5
    print(f"{len(rows)} dump rows matched to the eval parquets; position facts in the dump: {has_pos}")

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_verbalizer(args.verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    _, _, heads = load_rl_checkpoint(model, args.ar_ckpt, args.ar_ckpt, 1536, d_verb, args.device)
    model.eval()

    g = [gold[(r["split"], r["context_marked"])] for r in rows]
    gold_l = torch.tensor([x["z_L"] for x in g], dtype=torch.float32)
    gold_h = torch.tensor([x["z_H"] for x in g], dtype=torch.float32)
    true_tok = [token_prefix(r["context_marked"]).strip() for r in rows]
    true_pos = [f"Position: {position_bin(x['position'], x['prompt_len'])} of 5." for x in g]
    weights = ReconWeights()

    variants = [("gen_last", "gen", "last", "right"), ("gen_legacy", "gen", "legacy", "left"),
                ("oracle_token", "oracle_token", "last", "right")]
    if has_pos:
        variants += [("oracle_position", "oracle_position", "last", "right"), ("oracle_facts", "oracle_facts", "last", "right")]
    variants += [("facts_only_gen", "facts_only_gen", "last", "right"), ("facts_only_oracle", "facts_only_oracle", "last", "right"),
                 ("prose_only", "prose_only", "last", "right")]
    fve: dict[str, np.ndarray] = {}
    for label, kind, readout, side in variants:
        fl = [make_variant(kind, r["L_field"], tt, tp) for r, tt, tp in zip(rows, true_tok, true_pos, strict=True)]
        fh = [make_variant(kind, r["H_field"], tt, tp) for r, tt, tp in zip(rows, true_tok, true_pos, strict=True)]
        zl, zh = ar_hats(model, tokenizer, heads, critic_template, fl, fh, args.device, readout, args.batch_size, side)
        fve[label] = row_fve(zl, zh, gold_l, gold_h, weights, mean_mse)
        print(f"  done {label}", flush=True)

    rng = np.random.default_rng(0)
    results: dict = {}
    for split in sorted({r["split"] for r in rows}):
        for subset in ("all", "non-last"):
            idx = [i for i, r in enumerate(rows) if r["split"] == split and (subset == "all" or not r.get("is_last_prompt_pos"))]
            if not idx:
                continue
            bi = rng.integers(0, len(idx), (args.n_boot, len(idx)))
            print(f"\n## {split}, {subset} (n={len(idx)})   FVE sum / L / H   (paired difference to gen_last, 95% CI)")
            base = fve["gen_last"][idx]
            for label in fve:
                v = fve[label][idx]
                cell = {}
                parts = []
                for c, nm in enumerate(("sum", "L", "H")):
                    est = float(v[:, c].mean())
                    d = v[:, c] - base[:, c]
                    lo, hi = interval(d[bi].mean(1))
                    cell[nm] = {"fve": est, "diff": float(d.mean()), "diff_lo": lo, "diff_hi": hi}
                    parts.append(f"{est:+.3f} ({d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}])" if label != "gen_last" else f"{est:+.3f}")
                print(f"  {label:18s} " + "  ".join(parts))
                results.setdefault(split, {}).setdefault(subset, {})[label] = cell
    json.dump(results, open(args.output, "w"), indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
