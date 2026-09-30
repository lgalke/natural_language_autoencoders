"""Full evaluation of a trained AV+AR pair (post-SFT or post-RL — pass the
same checkpoint dir for both --av-ckpt/--ar-ckpt when evaluating an RL
checkpoint, since train_rl.py saves both adapters into one directory).

Produces, per named eval split (repeat `--eval-parquet name=path`, e.g.
`iid=.../eval_iid.parquet ood=.../eval_ood.parquet`):
  - format_rate: fraction of completions that parse as L:/H: fields
  - per-term FVE (sum/L/H), relative to norm_stats.json's mean-predictor
    baseline if given, else relative to THIS split's own mean predictor
  - field monitoring: L/H field length ratio, empty-field rate (folded into
    format_rate — parse_fields rejects empty fields), L-vs-H token-overlap
    (duplication smell — near-1.0 overlap means the two fields are saying
    the same thing, i.e. no stream-specific differentiation)
  - cancellation stats for (ẑ_L, ẑ_H) vs the stored (z_L, z_H) — comparable
    to `diagnostics.py`'s numbers on the raw data
  - broken down by is_last_prompt_pos vs other

With --run-judge (+ --base-model), also runs the Mimir patch-back judge
(`judge.py`) on the SAME rollouts and merges its KL numbers into the report
— the two are naturally sequential (judge needs the ẑ_L/ẑ_H this script just
produced) so doing both in one process avoids a JSON round-trip.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.hrm.devices import default_device, default_dtype
from nla.hrm.build import _INJECT_H_PLACEHOLDER, _INJECT_L_PLACEHOLDER
from nla.hrm.diagnostics import compute_stream_stats
from nla.hrm.model import DEFAULT_VERBALIZER, load_rl_checkpoint, load_verbalizer
from nla.hrm.recon import ReconWeights, parse_fields, recon_loss
from nla.hrm.sidecar import read_sidecar
from nla.datagen.storage import LocalStorage

_D_MIMIR = 1536


@torch.no_grad()
def generate_and_reconstruct(model, tokenizer, rows, inj_l_char, inj_h_char, inj_L, inj_H, heads, ids_meta,
                               critic_template, device, max_new_tokens=300, batch_size=16):
    from nla.hrm.model import build_inputs_embeds

    out = []
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

        parsed = [parse_fields(t) for t in texts]
        ok_idx = [i for i, p in enumerate(parsed) if p is not None]
        zL_hat = [None] * len(batch_rows)
        zH_hat = [None] * len(batch_rows)
        if ok_idx:
            model.set_adapter("ar")
            l_texts = [critic_template.format(explanation=parsed[i][0]) for i in ok_idx]
            h_texts = [critic_template.format(explanation=parsed[i][1]) for i in ok_idx]
            c_enc = tokenizer(l_texts + h_texts, return_tensors="pt", padding=True,
                                add_special_tokens=False, truncation=True, max_length=512)
            c_ids, c_attn = c_enc["input_ids"].to(device), c_enc["attention_mask"].to(device)
            c_out = model(input_ids=c_ids, attention_mask=c_attn, output_hidden_states=True, logits_to_keep=1)
            lengths = c_attn.sum(dim=1) - 1
            h_last = c_out.hidden_states[-1][torch.arange(c_ids.shape[0]), lengths]
            n = len(ok_idx)
            zL_pred, zH_pred = heads.forward_L(h_last[:n]), heads.forward_H(h_last[n:])
            for j, i in enumerate(ok_idx):
                zL_hat[i] = zL_pred[j].float().cpu()  # CPU: gold vectors live on CPU, avoids device mixing
                zH_hat[i] = zH_pred[j].float().cpu()

        for i, r in enumerate(batch_rows):
            out.append({
                "row": r, "text": texts[i], "parsed": parsed[i],
                "z_L_hat": zL_hat[i], "z_H_hat": zH_hat[i],
            })
    return out


def _dump_samples(path: str, split: str, results: list[dict], weights: ReconWeights, mean_mse: dict | None) -> None:
    with open(path, "a") as f:
        for r in results:
            row = r["row"]
            rec = {"split": split, "dataset": row.get("dataset"), "position": row.get("position"),
                   "is_last_prompt_pos": row.get("is_last_prompt_pos"),
                   "context_marked": row.get("context_marked"), "completion": r["text"],
                   "L_field": r["parsed"][0] if r["parsed"] else None,
                   "H_field": r["parsed"][1] if r["parsed"] else None}
            if r["parsed"] is not None:
                z_L = torch.tensor(row["z_L"]).unsqueeze(0)
                z_H = torch.tensor(row["z_H"]).unsqueeze(0)
                loss = recon_loss(r["z_L_hat"].unsqueeze(0), r["z_H_hat"].unsqueeze(0), z_L, z_H, weights)
                rec["mse"] = {"sum": loss.mse_sum.item(), "L": loss.mse_L.item(), "H": loss.mse_H.item()}
                if mean_mse:
                    rec["fve"] = {k: v for k, v in loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"]).items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def summarize(results: list[dict], weights: ReconWeights, mean_mse: dict | None) -> dict:
    n = len(results)
    ok = [r for r in results if r["parsed"] is not None]
    format_rate = len(ok) / n if n else 0.0

    fve = {"sum": [], "L": [], "H": []}
    overlaps, len_l, len_h = [], [], []
    cancel_hat, cos_hat = [], []
    for r in ok:
        z_L = torch.tensor(r["row"]["z_L"], dtype=torch.float32).unsqueeze(0)
        z_H = torch.tensor(r["row"]["z_H"], dtype=torch.float32).unsqueeze(0)
        zL_hat, zH_hat = r["z_L_hat"].unsqueeze(0), r["z_H_hat"].unsqueeze(0)
        loss = recon_loss(zL_hat, zH_hat, z_L, z_H, weights)
        if mean_mse:
            f = loss.fve(mean_mse["sum"], mean_mse["L"], mean_mse["H"])
            fve["sum"].append(f["fve_sum"])
            fve["L"].append(f["fve_L"])
            fve["H"].append(f["fve_H"])
        l_field, h_field = r["parsed"]
        overlaps.append(_jaccard(l_field, h_field))
        len_l.append(len(l_field.split()))
        len_h.append(len(h_field.split()))
        stats = compute_stream_stats(zL_hat, zH_hat)
        cancel_hat.append(stats["cancellation"].item())
        cos_hat.append(stats["cos_LH"].item())

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "n": n, "n_parsed": len(ok), "format_rate": format_rate,
        "fve_sum_mean": _mean(fve["sum"]), "fve_L_mean": _mean(fve["L"]), "fve_H_mean": _mean(fve["H"]),
        "lh_jaccard_mean": _mean(overlaps),  # near 1.0 = fields are near-duplicates (no differentiation)
        "l_field_words_mean": _mean(len_l), "h_field_words_mean": _mean(len_h),
        "cancellation_hat_mean": _mean(cancel_hat), "cos_LH_hat_mean": _mean(cos_hat),
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-parquet", action="append", nargs="+", required=True,
                    help="name=path; several per flag (--eval-parquet a=x.parquet b=y.parquet) or repeat the flag")
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--verbalizer-model", default=None)
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--w-sum", type=float, default=1.0)
    p.add_argument("--w-comp", type=float, default=0.25)
    p.add_argument("--norm-stats-json", default=None)
    p.add_argument("--reuse-generations", action="store_true",
                    help="reuse <output>.gen_<split>.pt from an earlier run instead of regenerating "
                         "(generation is the slow part; it is always saved there right after it finishes)")
    p.add_argument("--limit", type=int, default=None,
                    help="evaluate only a random sample of N rows per split (quick check; seeded, reproducible)")
    p.add_argument("--dump-samples", default=None,
                    help="write one JSON line per eval row (context, completion, L/H fields, per-row FVE) to this path")
    p.add_argument("--mean-from", default=None,
                    help="parquet with z_L/z_H (e.g. rl.parquet) to compute the mean-ablation baselines for the judge "
                         "(adds 'fraction of KL recovered'); strongly recommended with --run-judge")
    p.add_argument("--run-judge", action="store_true")
    p.add_argument("--base-model", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    weights = ReconWeights(w_sum=args.w_sum, w_comp=args.w_comp)
    mean_mse = json.load(open(args.norm_stats_json))["mean_mse"] if args.norm_stats_json else None

    verbalizer_model = args.verbalizer_model or DEFAULT_VERBALIZER
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_verbalizer(verbalizer_model, device=args.device, torch_dtype=dtype)
    d_verb = model.config.hidden_size
    inj_L, inj_H, heads = load_rl_checkpoint(model, args.av_ckpt, args.ar_ckpt, _D_MIMIR, d_verb, args.device)

    if args.dump_samples:
        open(args.dump_samples, "w").close()  # truncate once; each split appends
    report: dict = {"splits": {}}
    all_results: dict[str, list[dict]] = {}
    for spec in [spec for group in args.eval_parquet for spec in group]:
        name, path = spec.split("=", 1)
        rows = pq.read_table(path).to_pylist()
        if args.limit is not None and len(rows) > args.limit:
            import random
            rows = random.Random(0).sample(rows, args.limit)
        meta = read_sidecar(LocalStorage(), path)
        tm = meta.tokens
        assert tm is not None
        ids_meta = (tm.injection_token_id_L, tm.injection_left_neighbor_id_L, tm.injection_right_neighbor_id_L,
                    tm.injection_token_id_H, tm.injection_left_neighbor_id_H, tm.injection_right_neighbor_id_H)
        gen_path = f"{args.output}.gen_{name}.pt"
        if args.reuse_generations and Path(gen_path).exists():
            saved = torch.load(gen_path)
            assert len(saved) == len(rows), f"{gen_path} has {len(saved)} rows, split now has {len(rows)}; drop --reuse-generations"
            results = [{"row": r, **g} for r, g in zip(rows, saved, strict=True)]
            print(f"[{name}] reused {len(results)} saved generations from {gen_path}")
        else:
            results = generate_and_reconstruct(model, tokenizer, rows, tm.injection_char_L, tm.injection_char_H,
                                                 inj_L, inj_H, heads, ids_meta, meta.prompt_templates["critic"], args.device,
                                                 args.max_new_tokens, args.batch_size)
            torch.save([{k: v for k, v in r.items() if k != "row"} for r in results], gen_path)
        all_results[name] = results
        if args.dump_samples:
            _dump_samples(args.dump_samples, name, results, weights, mean_mse)
        summary = summarize(results, weights, mean_mse)
        last = [r for r in results if r["row"].get("is_last_prompt_pos")]
        other = [r for r in results if not r["row"].get("is_last_prompt_pos")]
        report["splits"][name] = {
            "overall": summary,
            "last_prompt_pos": summarize(last, weights, mean_mse) if last else None,
            "other_positions": summarize(other, weights, mean_mse) if other else None,
        }
        print(f"[{name}] n={summary['n']} format_rate={summary['format_rate']:.2%} "
              f"fve_sum={summary['fve_sum_mean']:.3f} fve_L={summary['fve_L_mean']:.3f} "
              f"fve_H={summary['fve_H_mean']:.3f} lh_jaccard={summary['lh_jaccard_mean']:.3f}")

    if args.run_judge:
        from nla.hrm.judge import run_judge
        from nla.hrm.mimir import DEFAULT_MIMIR, load_mimir
        mimir_model, mimir_tok = load_mimir(args.base_model or DEFAULT_MIMIR, device=args.device, torch_dtype=dtype)
        report["judge"] = {}
        mean_s = mean_zH = None
        if args.mean_from:
            t = pq.read_table(args.mean_from, columns=["z_L", "z_H"]).slice(0, 20000)
            zl = torch.tensor(t.column("z_L").to_pylist(), dtype=torch.float32)
            zh = torch.tensor(t.column("z_H").to_pylist(), dtype=torch.float32)
            mean_s, mean_zH = (zl + zh).mean(0), zh.mean(0)
        else:
            print("  [judge] no --mean-from: skipping mean-ablation baselines (no 'fraction of KL recovered')")
        for name, results in all_results.items():
            ok = [r for r in results if r["parsed"] is not None]
            if not ok:
                continue
            judge_rows = [r["row"] for r in ok]
            zL_hat = torch.stack([r["z_L_hat"] for r in ok])
            zH_hat = torch.stack([r["z_H_hat"] for r in ok])
            j = run_judge(mimir_model, mimir_tok, judge_rows, zL_hat, zH_hat, args.device, mean_s, mean_zH)
            report["judge"][name] = j
            import statistics
            pr = j["primary"]
            msg = (f"[judge:{name}] sum-patch KL at position mean={statistics.mean(pr['kl_pred_at_patch']):.4f} "
                   f"(noise floor, stored gold: {statistics.mean(pr['kl_true_at_patch']):.4f})")
            if "frac_kl_recovered" in pr:
                fr = [v for v in pr["frac_kl_recovered"] if v == v]
                msg += f"  fraction of KL recovered vs mean-ablation: {statistics.mean(fr):.3f}" if fr else ""
            sec = j["secondary_zH"]
            msg += f"  | z_H-only patch KL mean={statistics.mean(sec['kl_pred_at_patch']):.4f}"
            if "frac_kl_recovered" in sec:
                fr = [v for v in sec["frac_kl_recovered"] if v == v]
                msg += f" recovered={statistics.mean(fr):.3f}" if fr else ""
            print(msg)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
