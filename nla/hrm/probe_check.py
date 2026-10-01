"""Information ceiling: how well can the MARKED TOKEN be read linearly from the stored vectors?

    python -m nla.hrm.probe_check --base base.parquet [--top-k 200]

Trains a multinomial linear probe (softmax regression) on z_L, z_H, their sum, and
the concatenation to predict the identity of the token at the extraction position
(`prompt_ids[position]`, restricted to the top-K most frequent tokens), on a
`world`-grouped train/test split. Reports test accuracy vs the majority-class
baseline. If the token is linearly decodable but the verbalizer never quotes it
correctly, the problem is how the verbalizer is trained; if it is NOT decodable,
"quote the token" is the wrong target for these vectors.
"""

import argparse
import hashlib
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.hrm.devices import default_device


def _load(base: str, max_rows: int):
    t = pq.read_table(base, columns=["z_L", "z_H", "prompt_ids", "position", "world"]).slice(0, max_rows)
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    ids, pos, world = t.column("prompt_ids").to_pylist(), t.column("position").to_pylist(), t.column("world").to_pylist()
    tok = np.array([i[p] for i, p in zip(ids, pos, strict=True)])
    return zl, zh, tok, world


def _fit_eval(x_tr, y_tr, x_te, y_te, n_cls, device, steps=300, wd=1e-3) -> float:
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    lin = torch.nn.Linear(x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(x_te).argmax(-1) == y_te).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base.parquet from stage0_hrm (needs prompt_ids/position)")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=60000)
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    zl, zh, tok, world = _load(args.base, args.max_rows)
    top = [t for t, _ in Counter(tok.tolist()).most_common(args.top_k)]
    keep = np.isin(tok, top)
    remap = {t: i for i, t in enumerate(top)}
    y = np.array([remap[t] for t in tok[keep]])
    is_test = np.array([int(hashlib.sha256(w.encode()).hexdigest(), 16) % 5 == 0 for w in np.array(world)[keep]])
    zl, zh = zl[keep], zh[keep]
    print(f"rows={len(y)} (top-{len(top)} tokens cover {keep.mean():.0%} of rows), train={int((~is_test).sum())} test={int(is_test.sum())}")
    assert is_test.any() and (~is_test).any(), "need rows in both train and test (more worlds)"
    maj = Counter(y[~is_test].tolist()).most_common(1)[0][0]
    print(f"majority-class baseline accuracy: {(y[is_test] == maj).mean():.3f}   (chance {1 / len(top):.3f})")

    feats = {"z_L": zl, "z_H": zh, "sum": zl + zh, "concat": np.concatenate([zl, zh], 1)}
    for name, x in feats.items():
        xt = torch.tensor(x, device=args.device)
        yt = torch.tensor(y, device=args.device)
        te = torch.tensor(is_test, device=args.device)
        acc = _fit_eval(xt[~te], yt[~te], xt[te], yt[te], len(top), args.device)
        print(f"  linear probe on {name:7s}: test accuracy {acc:.3f}")


if __name__ == "__main__":
    main()
