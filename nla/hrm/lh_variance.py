"""How much of z_L is predictable from z_H (and vice versa)? The ceiling for any L-vs-H verbalizer.

    python -m nla.hrm.lh_variance --base base.parquet --limit 20000

Shared-normalised vectors (as in training). A ridge regression z_L ~ z_H (and z_H ~ z_L, and the sum) is fit on train
documents and scored on held-out documents; R^2 is against the train-mean predictor (so it is directly an 'FVE').
  R^2(z_L from z_H) near 1  -> z_L has little stream-specific variance; a separate L text has little to add
  R^2 well below 1          -> there is a stream-specific part; its share is 1 - R^2
Also reported: mean cosine(z_L, z_H), the share of the sum's variance, and the same numbers per dataset.
"""

import argparse
import collections
import hashlib

import numpy as np
import pyarrow.parquet as pq


def ridge_r2(x_tr, y_tr, x_te, y_te, lam: float = 1.0, device: str = "cpu") -> float:
    """R^2 of a ridge regression y ~ x on held-out rows, relative to the train-mean predictor."""
    return ridge_r2_multi(x_tr, {"y": y_tr}, x_te, {"y": y_te}, lam, device)["y"]


def ridge_r2_multi(x_tr, ys_tr: dict, x_te, ys_te: dict, lam: float = 1.0, device: str = "cpu") -> dict:
    """Same, for several targets sharing one design matrix: the Gram matrix is computed and factorised once.
    Runs in torch on `device` (use cuda on the cluster; numpy's BLAS there can be very slow)."""
    import torch

    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

    x_tr, x_te = t(x_tr), t(x_te)
    mx = x_tr.mean(0)
    xc = x_tr - mx
    gram = (xc.T @ xc).double()
    gram += lam * torch.eye(gram.shape[0], dtype=gram.dtype, device=device) * len(xc) / gram.shape[0]
    xte = x_te - mx
    out = {}
    for k, y_tr in ys_tr.items():
        y_tr, y_te = t(y_tr), t(ys_te[k])
        my = y_tr.mean(0)
        w = torch.linalg.solve(gram, (xc.T @ (y_tr - my)).double()).float()
        pred = xte @ w + my
        out[k] = float(1.0 - ((y_te - pred) ** 2).sum() / ((y_te - my) ** 2).sum())
    return out


def shared_normalise(zl: np.ndarray, zh: np.ndarray):
    c = np.sqrt((zl**2).sum(-1, keepdims=True) + (zh**2).sum(-1, keepdims=True)).clip(1e-12)
    return zl / c, zh / c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--lam", type=float, default=1.0, help="ridge strength (relative)")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = p.parse_args()

    t = pq.read_table(args.base, columns=["z_L", "z_H", "doc_id", "dataset"]).slice(0, args.limit)
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(n, -1)
    doc, ds = t.column("doc_id").to_pylist(), t.column("dataset").to_pylist()
    zl, zh = shared_normalise(zl, zh)
    s = zl + zh
    test = np.array([int(hashlib.sha256(str(d).encode()).hexdigest(), 16) % 5 == 0 for d in doc])  # document-level split
    tr, te = ~test, test
    print(f"device {args.device}")
    print(f"{n} rows ({tr.sum()} train / {te.sum()} test, split by document)")
    cos = (zl * zh).sum(-1) / (np.linalg.norm(zl, axis=1) * np.linalg.norm(zh, axis=1))
    print(f"mean cos(z_L, z_H) = {cos.mean():.3f}; |s|^2 share of (|z_L|^2+|z_H|^2) = {(s**2).sum() / ((zl**2).sum() + (zh**2).sum()):.3f}")
    for xname, x, targets in (("z_H", zh, {"z_L": zl, "sum": s}), ("z_L", zl, {"z_H": zh, "sum": s})):
        r2 = ridge_r2_multi(x[tr], {k: v[tr] for k, v in targets.items()}, x[te], {k: v[te] for k, v in targets.items()}, args.lam, args.device)
        for k, v in r2.items():
            print(f"  R^2 {k} from {xname}: {v:.3f}", flush=True)
    by = collections.defaultdict(list)
    for i, d in enumerate(ds):
        by[d].append(i)
    for d, idx in sorted(by.items()):
        idx = np.array(idx)
        a, b = idx[tr[idx]], idx[te[idx]]
        if len(a) > 200 and len(b) > 50:
            print(f"  {d}: R^2 z_L from z_H {ridge_r2(zh[a], zl[a], zh[b], zl[b], args.lam, args.device):.3f}  "
                  f"(n train {len(a)}, test {len(b)})")


if __name__ == "__main__":
    main()
