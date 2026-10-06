"""No-training baselines for FVE: what do simple group means already explain?

    python -m nla.hrm.baselines --base base.parquet \\
        --eval iid=eval_iid_clean.parquet ood=eval_ood.parquet --limit 100

For the SAME eval rows `eval` scores (same seeded --limit sample), predict the (shared-normalised) vectors with the
mean of the TRAINING rows in a group, and report FVE against the global mean predictor (norm_stats' definition, so the
numbers are directly comparable to eval's fve_*):
  global mean              FVE 0 by definition
  per dataset              knows only which source the prompt came from
  per dataset x last-pos   ... and whether it is the last prompt token
  per marked token         knows the TRUE marked token (falls back to dataset x last-pos if unseen / rare)
An explanation is informative beyond coarse context only if its FVE clearly exceeds these. Training rows = all base
rows whose (doc_id, position) is not in any eval parquet.
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from nla.hrm.lh_variance import ridge_fit


def _norm_pair(zl: np.ndarray, zh: np.ndarray):
    c = np.sqrt((zl**2).sum(-1, keepdims=True) + (zh**2).sum(-1, keepdims=True)).clip(1e-12)
    return zl / c, zh / c


def _fve(pred_l, pred_h, gold_l, gold_h, base) -> tuple[float, float, float]:
    mse_s = (((pred_l + pred_h) - (gold_l + gold_h)) ** 2).mean()
    mse_l, mse_h = ((pred_l - gold_l) ** 2).mean(), ((pred_h - gold_h) ** 2).mean()
    return 1 - mse_s / base["sum"], 1 - mse_l / base["L"], 1 - mse_h / base["H"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--eval", nargs="+", required=True, help="name=built_eval.parquet (same files/--limit as eval)")
    p.add_argument("--norm-stats-json", default="norm_stats.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-token-rows", type=int, default=3)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu",
                   help="device for the cross-stream ridge baselines")
    args = p.parse_args()
    base_mse = json.load(open(args.norm_stats_json))["mean_mse"]

    t = pq.read_table(args.base, columns=["z_L", "z_H", "dataset", "doc_id", "position", "prompt_ids", "is_last_prompt_pos"])
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    nl, nh = _norm_pair(zl, zh)
    ds, doc, pos = t.column("dataset").to_pylist(), t.column("doc_id").to_pylist(), t.column("position").to_pylist()
    last = t.column("is_last_prompt_pos").to_pylist()
    tok = [ids[q] for ids, q in zip(t.column("prompt_ids").to_pylist(), pos, strict=True)]
    index = {(d, q): i for i, (d, q) in enumerate(zip(doc, pos, strict=True))}

    evals = {}
    all_eval_keys = set()
    for spec in args.eval:
        name, path = spec.split("=", 1)
        full = pq.read_table(path, columns=["doc_id", "position"]).to_pylist()
        all_eval_keys |= {(r["doc_id"], r["position"]) for r in full}
        rows = pq.read_table(path, columns=["doc_id", "position"]).to_pylist()
        if args.limit is not None and len(rows) > args.limit:
            rows = random.Random(0).sample(rows, args.limit)  # same sampling as eval.py
        evals[name] = [index[(r["doc_id"], r["position"])] for r in rows]

    train = [i for i in range(n) if (doc[i], pos[i]) not in all_eval_keys]
    print(f"train rows (outside every eval file): {len(train)}")

    def group_means(keyfn):
        acc = defaultdict(lambda: [0, 0.0, 0.0])
        for i in train:
            a = acc[keyfn(i)]
            a[0] += 1
            a[1] = a[1] + nl[i]
            a[2] = a[2] + nh[i]
        return {k: (v[0], v[1] / v[0], v[2] / v[0]) for k, v in acc.items()}

    g_all = (np.mean([nl[i] for i in train], 0), np.mean([nh[i] for i in train], 0))
    m_ds = group_means(lambda i: ds[i])
    m_dsl = group_means(lambda i: (ds[i], last[i]))
    m_tok = group_means(lambda i: tok[i])

    def predictors(i):
        yield "global mean", g_all
        yield "per dataset", m_ds[ds[i]][1:] if ds[i] in m_ds else g_all
        fb = m_dsl.get((ds[i], last[i]))
        yield "per dataset x last-pos", fb[1:] if fb else g_all
        tk = m_tok.get(tok[i])
        yield "per TRUE marked token", tk[1:] if tk and tk[0] >= args.min_token_rows else (fb[1:] if fb else g_all)

    names = ["global mean", "per dataset", "per dataset x last-pos", "per TRUE marked token"]
    # Cross-stream linear baselines: predict one stream (and the sum) from the TRUE other stream with a ridge fit on the
    # train rows. Their FVE is the share of one stream's variation that the other stream already explains linearly; a
    # verbalizer whose L/H FVE equals these captures only the SHARED part of the streams.
    tr = np.array(train)
    s_all = nl + nh
    from_H = {"L": ridge_fit(nh[tr], nl[tr], device=args.device), "sum": ridge_fit(nh[tr], s_all[tr], device=args.device)}
    from_L = {"H": ridge_fit(nl[tr], nh[tr], device=args.device), "sum": ridge_fit(nl[tr], s_all[tr], device=args.device)}
    for split, idx in evals.items():
        for label, sel in (("all rows", idx), ("non-last rows", [i for i in idx if not last[i]])):
            if not sel:
                continue
            gl, gh = nl[sel], nh[sel]
            print(f"\n[{split}] {label} (n={len(sel)})   FVE sum / L / H   (eval's fve_* use the same definition)")
            for nm in names:
                pl = np.stack([dict(predictors(i))[nm][0] for i in sel])
                ph = np.stack([dict(predictors(i))[nm][1] for i in sel])
                s, l_, h = _fve(pl, ph, gl, gh, base_mse)
                print(f"  {nm:26s} {s:7.3f} {l_:7.3f} {h:7.3f}")
            sel_a = np.array(sel)
            for nm, fns, src, key in (("ridge from TRUE z_H", from_H, nh, "L"), ("ridge from TRUE z_L", from_L, nl, "H")):
                pred_s = fns["sum"](src[sel_a])
                pred_x = fns[key](src[sel_a])
                gold_x = (nl if key == "L" else nh)[sel_a]
                f_s = 1 - ((pred_s - s_all[sel_a]) ** 2).mean() / base_mse["sum"]
                f_x = 1 - ((pred_x - gold_x) ** 2).mean() / base_mse[key]
                cols = (f"{f_x:7.3f}  {'-':>7s}" if key == "L" else f"{'-':>7s}  {f_x:7.3f}")
                print(f"  {nm:26s} {f_s:7.3f} {cols}")


if __name__ == "__main__":
    main()
