"""Feature ceilings: how much FVE can ANY description reach that only knows a given set of features?

    python -m nla.hrm.feature_ceiling --base base_v2.parquet \\
        --eval iid=eval_iid_v2.parquet ood=eval_ood_v2.parquet --norm-stats-json norm_stats_v2.json \\
        --output feature_ceiling_v2.json

A lookup-table predictor sees ONLY the listed features of a row (never the vectors of the eval documents) and predicts the
shared-normalised (z_L, z_H). It is an additive model: one learned embedding per feature value, summed, regressed onto the
top principal components of the training vectors (ridge by weight decay, early-stopped on held-out DOCUMENTS). FVE is scored
exactly as `eval`/`baselines` do (against the train-mean predictor, norm_stats' definition). Nested feature sets:

  dataset, +last, +token, +neighbours (previous/next token), +position (relative fifths/20ths + absolute), +length,
  +bigrams (token with its previous / next token), +bag (hashed bag of ALL prompt tokens: an upper bound for "topic")
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
}
_BAG_BUCKETS = 20011
_BAG_CAP = 256


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


class _Table(torch.nn.Module):
    def __init__(self, sizes: dict[str, int], k: int, bag: bool, mean: torch.Tensor):
        super().__init__()
        self.emb = torch.nn.ModuleDict({n: torch.nn.Embedding(s, k) for n, s in sizes.items()})
        for e in self.emb.values():
            torch.nn.init.zeros_(e.weight)  # type: ignore[arg-type]
        self.bag = torch.nn.Embedding(_BAG_BUCKETS, k) if bag else None
        if self.bag is not None:
            torch.nn.init.zeros_(self.bag.weight)
        self.bias = torch.nn.Parameter(mean.clone())

    def forward(self, cols: dict[str, torch.Tensor], bag_ids: torch.Tensor | None) -> torch.Tensor:
        out = self.bias.expand(next(iter(cols.values())).shape[0], -1)
        for n, e in self.emb.items():
            out = out + e(cols[n])
        if self.bag is not None and bag_ids is not None:
            mask = (bag_ids > 0).float().unsqueeze(-1)
            out = out + (self.bag(bag_ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return out


def fit_set(cols: dict[str, np.ndarray], bag: np.ndarray | None, coords: np.ndarray, train_idx: np.ndarray,
            val_idx: np.ndarray, names: list[str], device: str, epochs: int = 12, lr: float = 1e-2, l2: float = 1e-4,
            batch: int = 512, seed: int = 0):
    """Fit the additive table on PC coordinates; keep the epoch with the lowest validation loss. Returns a predict function."""
    g = torch.Generator().manual_seed(seed)
    k = coords.shape[1]
    y = torch.tensor(coords, dtype=torch.float32)
    sizes = {n: int(cols[n].max()) + 1 for n in names if n != "bag"}
    use_bag = "bag" in names
    model = _Table(sizes, k, use_bag, y[torch.as_tensor(train_idx)].mean(0)).to(device)
    ct = {n: torch.as_tensor(cols[n]) for n in sizes}
    bt = torch.as_tensor(bag) if (use_bag and bag is not None) else None

    def batch_in(idx: torch.Tensor):
        return ({n: ct[n][idx].to(device) for n in ct}, bt[idx].to(device).long() if bt is not None else None)

    def predict(idx_np: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx_np), 4096):
                idx = torch.as_tensor(idx_np[s:s + 4096])
                out.append(model(*batch_in(idx)).cpu())
        return torch.cat(out).numpy()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = torch.as_tensor(train_idx)
    best, best_state = float("inf"), {n: p.detach().clone() for n, p in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            inputs = batch_in(idx)
            pred = model(*inputs)
            reg = sum((model.emb[n](inputs[0][n]) ** 2).sum() for n in sizes) / len(idx)
            loss = ((pred - y[idx].to(device)) ** 2).sum(1).mean() + l2 * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(((predict(val_idx) - y[torch.as_tensor(val_idx)].numpy()) ** 2).sum(1).mean())
        if v < best:
            best, best_state = v, {n: p.detach().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict


def run_ladder(cols, bag, z, d, train_idx, val_idx, eval_sets: dict[str, np.ndarray], base, sets, pcs, device, **fit_kw):
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
        pred = fit_set(cols, bag, coords, train_idx, val_idx, SETS[name], device, **fit_kw)
        add(name, lambda idx, p=pred: p(idx) @ pc.T + mean)
        print(f"  {name}: " + "  ".join(f"{sp} {results[name][sp]['all'][0]:+.3f}" for sp in eval_sets), flush=True)
    return results


def build_features(prompt_ids: list[list[int]], position: list[int], dataset: list[str], is_last: list[bool],
                   doc_id: list[str]):
    n = len(position)
    tok, prev, nxt = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
    plen = np.array([len(x) for x in prompt_ids])
    for i, (ids, p) in enumerate(zip(prompt_ids, position, strict=True)):
        tok[i] = ids[p]
        prev[i] = ids[p - 1] if p > 0 else -1
        nxt[i] = ids[p + 1] if p + 1 < len(ids) else -1
    pos = np.asarray(position)
    cols = {
        "tok": tok, "prev": prev, "next": nxt,
        "bi_prev": tok * 400000 + prev + 1, "bi_next": tok * 400000 + nxt + 1,
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
    return cols, bag_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--eval", nargs="+", required=True, help="name=built_eval.parquet (their rows are held out of training)")
    p.add_argument("--norm-stats-json", default="norm_stats.json")
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    p.add_argument("--max-train", type=int, default=100000)
    p.add_argument("--pcs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    base_mse = json.load(open(args.norm_stats_json))["mean_mse"]

    t = pq.read_table(args.base, columns=["z_L", "z_H", "dataset", "doc_id", "position", "prompt_ids", "is_last_prompt_pos"])
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    d = zl.shape[1]
    c = np.sqrt((zl**2).sum(-1, keepdims=True) + (zh**2).sum(-1, keepdims=True)).clip(1e-12)
    z = np.concatenate([zl / c, zh / c], axis=1)
    doc, pos = t.column("doc_id").to_pylist(), t.column("position").to_pylist()
    cols, bag = build_features(t.column("prompt_ids").to_pylist(), pos, t.column("dataset").to_pylist(),
                               t.column("is_last_prompt_pos").to_pylist(), doc)
    index = {(a, b): i for i, (a, b) in enumerate(zip(doc, pos, strict=True))}

    eval_sets, eval_keys = {}, set()
    for spec in args.eval:
        name, path = spec.split("=", 1)
        rows = pq.read_table(path, columns=["doc_id", "position"]).to_pylist()
        eval_sets[name] = np.array([index[(r["doc_id"], r["position"])] for r in rows])
        eval_keys |= {(r["doc_id"], r["position"]) for r in rows}
    eval_docs = {a for a, _ in eval_keys}
    rng = np.random.default_rng(args.seed)
    train_all = np.array([i for i in range(n) if doc[i] not in eval_docs])
    docs_u = sorted({doc[i] for i in train_all})
    val_docs = set(rng.choice(docs_u, size=max(1, len(docs_u) // 20), replace=False).tolist())
    val_idx = np.array([i for i in train_all if doc[i] in val_docs])
    train_idx = np.array([i for i in train_all if doc[i] not in val_docs])
    if len(train_idx) > args.max_train:
        train_idx = rng.choice(train_idx, size=args.max_train, replace=False)
    print(f"train {len(train_idx)} rows, val {len(val_idx)} rows (held-out documents), eval {({k: len(v) for k, v in eval_sets.items()})}")

    cols = {k: compact(v, train_idx) if k in ("tok", "prev", "next", "bi_prev", "bi_next") else v for k, v in cols.items()}
    res = run_ladder(cols, bag, z, d, train_idx, val_idx, eval_sets, base_mse, args.sets, args.pcs, args.device,
                     epochs=args.epochs, seed=args.seed)
    json.dump(res, open(args.output, "w"), indent=1)
    for sp in eval_sets:
        for sub in ("all", "non-last"):
            print(f"\n## {sp}, {sub}   FVE sum / L / H")
            for name, r in res.items():
                print(f"  {name:12s} " + " ".join(f"{x:+.3f}" for x in r[sp][sub]))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
