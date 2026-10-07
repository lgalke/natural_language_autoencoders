"""Bootstrap confidence intervals (row resampling) for the headline eval numbers, from existing dumps and eval JSONs.

    python -m nla.hrm.bootstrap \\
        --dump E7=samples_tokw_final.jsonl E8=samples_ver_final.jsonl Split=samples_split_sft.jsonl \\
        --judge E7=eval_tokw_final.json \\
        --compare E7 E8 --out results.json

--dump   `eval --dump-samples` files (per-row FVE, token quotes, grounding). Every ratio metric is a ratio of sums over the
         resampled rows (the same definition as the eval summaries), so the point estimates match the printed eval numbers.
--judge  `eval --run-judge` JSONs (per-row KL lists): KL of the reconstruction, of the mean vector, of the shuffled
         reconstruction (if `--judge-shuffle` was used), and the share of the mean-ablation KL removed (ratio of means).
--compare A B   PAIRED difference B - A over the same rows (dumps must come from the same `--limit` sample: checked).
--out    results.json, read by `nla.hrm.plots`.

Caveats: rows are resampled as independent; several rows can come from one document (positions of a prompt), which makes
the intervals somewhat too narrow. The intervals say nothing about seeds, only about which rows were drawn.
"""

import argparse
import json

import numpy as np

METRICS_DUMP = ["fve_sum", "fve_L", "fve_H", "tok_first", "tok_L", "tok_H", "pos_L", "pos_H", "ground_span", "ground_name"]
SUBSETS = ("all", "non-last")


def load_dump(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def row_arrays(rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """metric -> (numerator, denominator) per row; the metric over a set of rows is sum(num) / sum(den)."""
    from nla.hrm.eval import quote_match_fields

    n = len(rows)
    out = {m: (np.zeros(n), np.zeros(n)) for m in METRICS_DUMP}
    for i, r in enumerate(rows):
        fve = r.get("fve")
        if fve:
            for key, m in (("fve_sum", "fve_sum"), ("fve_L", "fve_L"), ("fve_H", "fve_H")):
                out[m][0][i], out[m][1][i] = fve[key], 1.0
        q = r.get("marked_token_quote_correct")
        if q is not None:
            out["tok_first"][0][i], out["tok_first"][1][i] = float(q is True), 1.0
        if r.get("L_field") and r.get("H_field"):
            a, b = quote_match_fields((r["L_field"], r["H_field"]), r.get("context_marked"))
            for m, v in (("tok_L", a), ("tok_H", b)):
                if v is not None:
                    out[m][0][i], out[m][1][i] = float(v is True), 1.0
        for m, v in zip(("pos_L", "pos_H"), r.get("pos_correct") or (None, None), strict=True):
            if v is not None:
                out[m][0][i], out[m][1][i] = float(v is True), 1.0
        g = r.get("grounding")
        if g:
            out["ground_span"][0][i], out["ground_span"][1][i] = g["quoted_ok"], g["quoted_total"]
            out["ground_name"][0][i], out["ground_name"][1][i] = g["caps_ok"], g["caps_total"]
    return out


def boot_ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    d = den[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, num[idx].sum(1) / d, np.nan)


def interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def summarise(num, den, rng, n_boot):
    n = len(num)
    est = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    lo, hi = interval(boot_ratio(num, den, rng.integers(0, n, (n_boot, n)))) if den.sum() > 0 else (float("nan"),) * 2
    return {"est": est, "lo": lo, "hi": hi, "n_rows": int((den > 0).sum()), "den": float(den.sum())}


def dump_results(rows: list[dict], n_boot: int, seed: int) -> dict:
    res: dict = {}
    for split in sorted({r["split"] for r in rows}):
        for subset in SUBSETS:
            sel = [r for r in rows if r["split"] == split and (subset == "all" or not r.get("is_last_prompt_pos"))]
            if not sel:
                continue
            rng = np.random.default_rng(seed)
            arrays = row_arrays(sel)
            res.setdefault(split, {})[subset] = {m: summarise(*arrays[m], rng, n_boot) for m in METRICS_DUMP}
    return res


def paired_results(rows_a: list[dict], rows_b: list[dict], n_boot: int, seed: int) -> dict:
    res: dict = {}
    for split in sorted({r["split"] for r in rows_a}):
        a_rows = [r for r in rows_a if r["split"] == split]
        b_rows = [r for r in rows_b if r["split"] == split]
        assert len(a_rows) == len(b_rows) and all(
            x["context_marked"] == y["context_marked"] for x, y in zip(a_rows, b_rows, strict=True)), (
            f"{split}: the two dumps are not row-aligned (use the same --limit and the same eval parquet)")
        for subset in SUBSETS:
            keep = [i for i, r in enumerate(a_rows) if subset == "all" or not r.get("is_last_prompt_pos")]
            aa, bb = row_arrays([a_rows[i] for i in keep]), row_arrays([b_rows[i] for i in keep])
            n = len(keep)
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, n, (n_boot, n))
            entry = {}
            for m in METRICS_DUMP:
                da = boot_ratio(*aa[m], idx)
                db = boot_ratio(*bb[m], idx)
                est = float(bb[m][0].sum() / bb[m][1].sum() - aa[m][0].sum() / aa[m][1].sum()) if aa[m][1].sum() and bb[m][1].sum() else float("nan")
                lo, hi = interval(db - da)
                entry[m] = {"est": est, "lo": lo, "hi": hi, "n_rows": n}
            res.setdefault(split, {})[subset] = entry
    return res


def judge_results(path: str, n_boot: int, seed: int) -> dict:
    report = json.load(open(path))
    res: dict = {}
    for kind, key in (("real", "judge"), ("shuffled", "judge_shuffled")):
        for split, j in report.get(key, {}).items():
            for site in ("primary", "secondary_zH"):
                d = j.get(site, {})
                if "kl_pred_at_patch" not in d or "kl_mean_ablation_at_patch" not in d:
                    continue
                pred = np.array(d["kl_pred_at_patch"], dtype=float)
                mean = np.array(d["kl_mean_ablation_at_patch"], dtype=float)
                rng = np.random.default_rng(seed)
                n = len(pred)
                idx = rng.integers(0, n, (n_boot, n))
                ones = np.ones(n)
                removed = 1.0 - boot_ratio(pred, mean, idx)
                entry = res.setdefault(split, {}).setdefault(site, {})
                entry[f"{kind}_kl"] = {"est": float(pred.mean()), "lo": interval(boot_ratio(pred, ones, idx))[0],
                                       "hi": interval(boot_ratio(pred, ones, idx))[1], "n_rows": n}
                entry[f"{kind}_removed"] = {"est": float(1.0 - pred.sum() / mean.sum()), "lo": interval(removed)[0],
                                            "hi": interval(removed)[1], "n_rows": n}
                if kind == "real":
                    entry["mean_ablation_kl"] = {"est": float(mean.mean()), "lo": interval(boot_ratio(mean, ones, idx))[0],
                                                 "hi": interval(boot_ratio(mean, ones, idx))[1], "n_rows": n}
                    entry["beats_mean_share"] = {"est": float((pred < mean).mean()), "n_rows": n}
    return res


def fmt(m: dict | None) -> str:
    if not m or m.get("est") != m.get("est"):
        return "-"
    return f"{m['est']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]" if "lo" in m else f"{m['est']:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", nargs="+", default=[], metavar="NAME=PATH")
    p.add_argument("--judge", nargs="+", default=[], metavar="NAME=PATH")
    p.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    dumps = {s.split("=", 1)[0]: load_dump(s.split("=", 1)[1]) for s in args.dump}
    results: dict = {"models": {}, "paired": {}, "judge": {}}
    for name, rows in dumps.items():
        results["models"][name] = dump_results(rows, args.n_boot, args.seed)
    for a, b in args.compare:
        results["paired"][f"{b} - {a}"] = paired_results(dumps[a], dumps[b], args.n_boot, args.seed)
    for s in args.judge:
        name, path = s.split("=", 1)
        results["judge"][name] = judge_results(path, args.n_boot, args.seed)
    json.dump(results, open(args.out, "w"), indent=1)

    cols = ["fve_sum", "fve_L", "fve_H", "tok_L", "tok_H", "ground_span", "ground_name"]
    for split in ("iid", "ood"):
        for subset in SUBSETS:
            print(f"\n## {split}, {subset}   (estimate [95% CI]; token and grounding are ratios over quoted items)")
            print("| model | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
            for name, r in results["models"].items():
                cell = r.get(split, {}).get(subset)
                if cell:
                    print(f"| {name} | " + " | ".join(fmt(cell[c]) for c in cols) + " |")
            for name, r in results["paired"].items():
                cell = r.get(split, {}).get(subset)
                if cell:
                    print(f"| paired {name} | " + " | ".join(fmt(cell[c]) for c in cols) + " |")
    for name, r in results["judge"].items():
        for split, sites in r.items():
            for site, e in sites.items():
                print(f"\n[judge {name} {split} {site}] " + "; ".join(f"{k} {fmt(v)}" for k, v in e.items()))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
