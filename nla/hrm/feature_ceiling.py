"""Feature ceilings: how much FVE can ANY description reach that only knows a given set of features?

    python -m nla.hrm.feature_ceiling --base base_v2.parquet \\
        --eval iid=eval_iid_v2.parquet ood=eval_ood_v2.parquet --norm-stats-json norm_stats_v2.json \\
        --output feature_ceiling_v2.json

A lookup-table predictor sees ONLY the listed features of a row (never the vectors of the eval documents) and predicts the
shared-normalised (z_L, z_H). It is an additive model: one learned embedding per feature value, summed, regressed onto the
top principal components of the training vectors (ridge, solved exactly by block conjugate gradients, strength picked on held-out
DOCUMENTS). Only CONTENT rows are used for training and scoring (outside the last 5 chat-template tokens, as in the v2 training
buckets and in `norm_stats`). FVE is scored
exactly as `eval`/`baselines` do (against the train-mean predictor, norm_stats' definition). Nested feature sets:

  dataset, +last, +token, +neighbours (previous/next token), +position (relative fifths/20ths + absolute), +length,
  +bigrams (token with its previous / next token), +bag (hashed bag of ALL prompt tokens: an upper bound for "topic"),
  +trigrams (previous two tokens; previous+next), +window (hashed bag of the tokens within +-8 positions), +all
and two stand-alone sets, `token` and `position`, that split the main effect.

Reading: a description made of the features in a set cannot beat that set's FVE (up to the capacity of an additive table
model), so the facts and prose worth asking the verbalizer for are those that move this ladder. `pca ceiling` is the FVE of
projecting the TRUE vectors on the PCs used, i.e. the most this method can show. Caveats: a table model is not the best
possible reader of the features (a neural reader could do more), `+bag` sees the future of the prompt (the bidirectional
state does too), and eval rows are the full built eval parquets (not the 300-row eval samples).
"""

import argparse
import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch

SETS: dict[str, list[str]] = {
    "dataset": ["ds"],
    "token": ["tok"],
    "position": ["pos20", "abspos", "lenb", "last"],
    "+last": ["ds", "last"],
    "+token": ["ds", "last", "tok"],
    "+neighbours": ["ds", "last", "tok", "prev", "next"],
    "+position": ["ds", "last", "tok", "prev", "next", "pos20", "abspos"],
    "+length": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb"],
    "+bigrams": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb", "bi_prev", "bi_next"],
    "+bag": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb", "bi_prev", "bi_next", "bag"],
    "+trigrams": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb", "bi_prev", "bi_next", "tri_prev", "tri_mid"],
    "+window": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb", "bi_prev", "bi_next", "tri_prev", "tri_mid",
                "win"],
    "+all": ["ds", "last", "tok", "prev", "next", "pos20", "abspos", "lenb", "bi_prev", "bi_next", "tri_prev", "tri_mid",
             "win", "bag"],
}
BAG_NAMES = ("bag", "win")
_BAG_BUCKETS = 20011
_BAG_CAP = 128


def compact(col: np.ndarray, train_idx: np.ndarray, min_count: int = 3, max_cats: int = 30000) -> np.ndarray:
    """Map values to 1..K by training frequency (values rarer than `min_count`, or beyond the `max_cats` most common, -> 0)."""
    counts = Counter(col[train_idx].tolist())
    keep = [v for v, c in counts.most_common(max_cats) if c >= min_count]
    table = {v: i + 1 for i, v in enumerate(keep)}
    return np.fromiter((table.get(v, 0) for v in col.tolist()), dtype=np.int64, count=len(col))


def fve_from(pred_l, pred_h, gold_l, gold_h, base) -> tuple[float, float, float]:
    mse_s = (((pred_l + pred_h) - (gold_l + gold_h)) ** 2).mean()
    mse_l, mse_h = ((pred_l - gold_l) ** 2).mean(), ((pred_h - gold_h) ** 2).mean()
    return float(1 - mse_s / base["sum"]), float(1 - mse_l / base["L"]), float(1 - mse_h / base["H"])


def _design(cols: dict[str, np.ndarray], bags: dict[str, np.ndarray], names: list[str], idx: np.ndarray,
            offsets: dict[str, int], n_cols: int, device: str):
    """Sparse one-hot design matrix (CSR) and its transpose for the rows `idx`; the bag columns carry weight 1/|bag|."""
    rows, cidx, vals = [], [], []
    r = np.arange(len(idx))
    for n in names:
        if n in BAG_NAMES:
            continue
        rows.append(r)
        cidx.append(cols[n][idx] + offsets[n])
        vals.append(np.ones(len(idx), np.float32))
    for n in names:
        if n in BAG_NAMES:
            b = bags[n][idx]
            rr, cc = np.nonzero(b)
            w = 1.0 / np.maximum((b > 0).sum(1), 1).astype(np.float32)
            rows.append(rr)
            cidx.append(b[rr, cc].astype(np.int64) + offsets[n])
            vals.append(w[rr])
    coo = torch.sparse_coo_tensor(torch.as_tensor(np.stack([np.concatenate(rows), np.concatenate(cidx)])),
                                  torch.as_tensor(np.concatenate(vals)), (len(idx), n_cols)).coalesce()
    return coo.to(device), coo.t().coalesce().to(device)


def ridge_cg(a, at, y: torch.Tensor, lam: float, iters: int = 200, tol: float = 1e-4) -> torch.Tensor:
    """Solve (AtA + lam I) W = At Y for all columns of Y at once (block conjugate gradients, per-column step sizes)."""
    w = torch.zeros(at.shape[0], y.shape[1], device=y.device)
    rhs = torch.sparse.mm(at, y)
    r = rhs.clone()
    p = r.clone()
    rs = (r * r).sum(0)
    for _ in range(iters):
        ap = torch.sparse.mm(at, torch.sparse.mm(a, p)) + lam * p
        alpha = rs / (p * ap).sum(0).clamp(min=1e-20)
        w += p * alpha
        r -= ap * alpha
        rs_new = (r * r).sum(0)
        if float(rs_new.max().sqrt() / rhs.norm(dim=0).max().clamp(min=1e-20)) < tol:
            break
        p = r + p * (rs_new / rs.clamp(min=1e-20))
        rs = rs_new
    return w


def fit_set(cols: dict[str, np.ndarray], bags: dict[str, np.ndarray], coords: np.ndarray, train_idx: np.ndarray,
            val_idx: np.ndarray, names: list[str], device: str, lams: tuple[float, ...] = (1.0, 10.0, 100.0), iters: int = 200):
    """Additive table = ridge regression of the PC coordinates on one-hot features, solved exactly (CG); the ridge strength
    is picked on held-out DOCUMENTS (`val_idx`). Returns predict(idx) -> PC coordinates."""
    offsets, n_cols = {}, 0
    for n in names:
        offsets[n] = n_cols
        n_cols += _BAG_BUCKETS if n in BAG_NAMES else int(cols[n].max()) + 1
    y = torch.tensor(coords, dtype=torch.float32)
    mu = y[torch.as_tensor(train_idx)].mean(0)
    a_tr, at_tr = _design(cols, bags, names, train_idx, offsets, n_cols, device)
    a_va, _ = _design(cols, bags, names, val_idx, offsets, n_cols, device)
    y_tr, y_va = (y[torch.as_tensor(train_idx)] - mu).to(device), (y[torch.as_tensor(val_idx)] - mu).to(device)
    best, best_w = float("inf"), None
    for lam in lams:
        w = ridge_cg(a_tr, at_tr, y_tr, lam, iters)
        v = float(((torch.sparse.mm(a_va, w) - y_va) ** 2).sum(1).mean())
        if v < best:
            best, best_w = v, w
    print(f"    ridge: val loss {best:.4f} (train-mean predictor {float((y_va ** 2).sum(1).mean()):.4f})", flush=True)

    def predict(idx_np: np.ndarray) -> np.ndarray:
        a, _ = _design(cols, bags, names, idx_np, offsets, n_cols, device)
        return (torch.sparse.mm(a, best_w).cpu() + mu).numpy()

    return predict


def run_ladder(cols, bags, z, d, train_idx, val_idx, eval_sets: dict[str, np.ndarray], base, sets, pcs, device, **fit_kw):
    """z: [N, 2d] shared-normalised (z_L, z_H). Returns {set: {split: {subset: [sum, L, H]}}} plus the PCA ceiling."""
    mean = z[train_idx].mean(0)
    xc = torch.tensor(z[train_idx] - mean, dtype=torch.float32)
    cov = xc.T @ xc / len(xc)
    _, evecs = torch.linalg.eigh(cov)
    pc = evecs[:, -pcs:].numpy().astype(np.float32)  # [2d, pcs]
    coords = (z - mean) @ pc
    last = cols["last"].astype(bool)
    results: dict = {}

    def score(zhat: np.ndarray, idx: np.ndarray, subset: str) -> list[float]:
        if subset == "non-last":
            keep = ~last[idx]
            idx, zhat = idx[keep], zhat[keep]
        return list(fve_from(zhat[:, :d], zhat[:, d:], z[idx, :d], z[idx, d:], base))

    def add(name, predict_z):
        results[name] = {sp: {sub: score(predict_z(idx), idx, sub) for sub in ("all", "non-last")}
                         for sp, idx in eval_sets.items()}

    add("pca ceiling", lambda idx: (z[idx] - mean) @ pc @ pc.T + mean)
    for name in sets:
        pred = fit_set(cols, bags, coords, train_idx, val_idx, SETS[name], device, **fit_kw)
        add(name, lambda idx, p=pred: p(idx) @ pc.T + mean)
        print(f"  {name}: " + "  ".join(f"{sp} {results[name][sp]['all'][0]:+.3f}" for sp in eval_sets), flush=True)
    return results


def build_features(prompt_ids: list[list[int]], position: list[int], dataset: list[str], is_last: list[bool],
                   doc_id: list[str]):
    n = len(position)
    tok, prev, nxt, prev2 = (np.zeros(n, np.int64) for _ in range(4))
    win = np.zeros((n, 17), dtype=np.int32)
    plen = np.array([len(x) for x in prompt_ids])
    for i, (ids, p) in enumerate(zip(prompt_ids, position, strict=True)):
        tok[i] = ids[p]
        prev[i] = ids[p - 1] if p > 0 else -1
        nxt[i] = ids[p + 1] if p + 1 < len(ids) else -1
        prev2[i] = ids[p - 2] if p > 1 else -1
        w = [ids[q] for q in range(max(0, p - 8), min(len(ids), p + 9))]
        win[i, :len(w)] = np.asarray(w) % (_BAG_BUCKETS - 1) + 1
    pos = np.asarray(position)
    cols = {
        "tok": tok, "prev": prev, "next": nxt,
        "bi_prev": tok * 400000 + prev + 1, "bi_next": tok * 400000 + nxt + 1,
        "tri_prev": (prev2 + 1) * 160000000000 + (prev + 1) * 400000 + tok,
        "tri_mid": (prev + 1) * 160000000000 + tok * 400000 + nxt + 1,
        "pos20": np.minimum((20 * pos / plen).astype(np.int64), 19),
        "abspos": np.minimum(pos // 16, 63).astype(np.int64),
        "lenb": np.minimum(plen // 32, 63).astype(np.int64),
        "last": np.asarray(is_last, dtype=np.int64),
        "ds": np.array([{d: i for i, d in enumerate(sorted(set(dataset)))}[d] for d in dataset], dtype=np.int64),
    }
    # hashed bag of unique prompt tokens, one row per prompt, then mapped to rows
    first: dict = {}
    bag_rows = np.zeros((n, _BAG_CAP), dtype=np.int32)
    for i, (d, ids) in enumerate(zip(doc_id, prompt_ids, strict=True)):
        key = (d, len(ids), ids[0], ids[-1])
        if key not in first:
            u = np.unique(np.asarray(ids)) % (_BAG_BUCKETS - 1) + 1
            row = np.zeros(_BAG_CAP, dtype=np.int32)
            row[:min(len(u), _BAG_CAP)] = u[:_BAG_CAP]
            first[key] = row
        bag_rows[i] = first[key]
    return cols, {"bag": bag_rows, "win": win}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--eval", nargs="+", required=True, help="name=built_eval.parquet (their rows are held out of training)")
    p.add_argument("--norm-stats-json", default="norm_stats.json")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--by-doc", action="store_true",
                   help="evaluate on ALL base rows of the eval documents instead of the exact eval rows (automatic for "
                        "single-vector bases from extract_hf_states)")
    p.add_argument("--max-train", type=int, default=100000)
    p.add_argument("--pcs", type=int, default=512)
    p.add_argument("--tail", type=int, default=5,
                   help="drop rows within the last TAIL prompt tokens (chat-template tail; the v2 training buckets dropped them "
                        "and norm_stats is computed without them), from training AND evaluation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    single = "z" in pq.read_schema(args.base).names  # `extract_hf_states` output: one vector per row, no L/H split
    meta_cols = ["dataset", "doc_id", "position", "prompt_ids", "is_last_prompt_pos"]
    t = pq.read_table(args.base, columns=(["z"] if single else ["z_L", "z_H"]) + meta_cols)
    n = t.num_rows

    def col(name: str) -> np.ndarray:
        return t.column(name).combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)

    zl = col("z") if single else col("z_L")
    zh = np.zeros_like(zl) if single else col("z_H")
    d = zl.shape[1]
    c = np.sqrt((zl**2).sum(-1, keepdims=True) + (zh**2).sum(-1, keepdims=True)).clip(1e-12)
    z = np.concatenate([zl / c, zh / c], axis=1)
    doc, pos = t.column("doc_id").to_pylist(), t.column("position").to_pylist()
    prompt_ids = t.column("prompt_ids").to_pylist()
    cols, bags = build_features(prompt_ids, pos, t.column("dataset").to_pylist(),
                               t.column("is_last_prompt_pos").to_pylist(), doc)
    index = {(a, b): i for i, (a, b) in enumerate(zip(doc, pos, strict=True))}

    eval_sets, eval_keys, eval_docs = {}, set(), set()
    for spec in args.eval:
        name, path = spec.split("=", 1)
        rows = pq.read_table(path, columns=["doc_id", "position"]).to_pylist()
        docs_here = {r["doc_id"] for r in rows}
        if args.by_doc or single:  # all base rows of the eval documents (positions differ between tokenizers)
            eval_sets[name] = np.array([i for i in range(n) if doc[i] in docs_here])
        else:
            eval_sets[name] = np.array([index[(r["doc_id"], r["position"])] for r in rows])
        eval_keys |= {(r["doc_id"], r["position"]) for r in rows}
        eval_docs |= docs_here
    rng = np.random.default_rng(args.seed)
    plen = np.array([len(x) for x in prompt_ids])
    content = np.asarray(pos) < plen - args.tail
    eval_sets = {k: v[content[v]] for k, v in eval_sets.items()}
    print(f"content rows only (position < prompt_len - {args.tail}): eval {({k: len(v) for k, v in eval_sets.items()})}")
    train_all = np.array([i for i in range(n) if doc[i] not in eval_docs and content[i]])
    docs_u = sorted({doc[i] for i in train_all})
    val_docs = set(rng.choice(docs_u, size=max(1, len(docs_u) // 20), replace=False).tolist())
    val_idx = np.array([i for i in train_all if doc[i] in val_docs])
    train_idx = np.array([i for i in train_all if doc[i] not in val_docs])
    if len(train_idx) > args.max_train:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
    print(f"train {len(train_idx)} rows, val {len(val_idx)} rows (held-out documents), eval {({k: len(v) for k, v in eval_sets.items()})}")
    if single:  # baseline = variance of the normalised vectors around the training mean (only the sum / L column is meaningful)
        m = z[train_idx, :d].mean(0)
        v = float(((z[train_idx, :d] - m) ** 2).mean())
        base_mse = {"sum": v, "L": v, "H": 1.0}
    else:
        base_mse = json.load(open(args.norm_stats_json))["mean_mse"]

    cols = {k: compact(v, train_idx) if k in ("tok", "prev", "next", "bi_prev", "bi_next", "tri_prev", "tri_mid") else v for k, v in cols.items()}
    res = run_ladder(cols, bags, z, d, train_idx, val_idx, eval_sets, base_mse, args.sets, args.pcs, args.device)
    json.dump(res, open(args.output, "w"), indent=1)
    for sp in eval_sets:
        for sub in ("all", "non-last"):
            print(f"\n## {sp}, {sub}   FVE sum / L / H")
            for name, r in res.items():
                print(f"  {name:12s} " + " ".join(f"{x:+.3f}" for x in r[sp][sub]))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
