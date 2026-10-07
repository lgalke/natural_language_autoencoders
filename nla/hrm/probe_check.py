"""Information ceiling: how well can the MARKED TOKEN be read linearly from the stored vectors?

    python -m nla.hrm.probe_check --base base.parquet [--top-k 200]
    python -m nla.hrm.probe_check --base base.parquet --non-last --offsets -3 -2 -1 0 1 2 --mlp-hidden 512
    python -m nla.hrm.probe_check --base base.parquet --non-last --train-sizes 500 2000 8000 --mlp-hidden 512
    python -m nla.hrm.probe_check --base base.parquet --non-last --target dataset --mlp-hidden 512   # source identity
    python -m nla.hrm.probe_check --base base.parquet --non-last --target relpos --mlp-hidden 512    # position in the prompt (5 bins)

Profile mode (is H "easier to decode" than L, and for WHICH information?):
  --offsets K...      probe the token at position + K (K=0 is the extraction position; negative = earlier tokens, positive =
                      LATER tokens, which a bidirectional prefix-LM state can encode). Rows where position + K falls outside
                      the prompt are dropped; the top-K classes are recomputed per offset.
  --non-last          drop the last prompt position (always the same template token; it inflates the majority baseline).
  --mlp-hidden H      also fit a one-hidden-layer MLP probe: if the L-versus-H gap closes, the information is there but
                      NONLINEARLY encoded, i.e. "harder to read linearly", not "absent".
  --train-sizes N...  learning curve: train on N rows (for each offset): data hunger versus a real information gap.
Accuracies carry a 95% binomial interval over the test rows (the split is by prompt group, so rows of a prompt stay together;
the interval ignores the group correlation and probe-fit randomness, so read gaps below about 0.03 as ties).

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


def _load(base: str, max_rows: int, offset: int = 0, non_last: bool = False):
    """Returns zl, zh, tok (token at position + offset), world for the rows where that position exists."""
    cols = ["z_L", "z_H", "prompt_ids", "position", "world"] + (["is_last_prompt_pos"] if non_last else [])
    t = pq.read_table(base, columns=cols).slice(0, max_rows)
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    ids, pos, world = t.column("prompt_ids").to_pylist(), t.column("position").to_pylist(), t.column("world").to_pylist()
    last = t.column("is_last_prompt_pos").to_pylist() if non_last else [False] * n
    ok = [i for i in range(n) if not last[i] and 0 <= pos[i] + offset < len(ids[i])]
    tok = np.array([ids[i][pos[i] + offset] for i in ok])
    return zl[ok], zh[ok], tok, [world[i] for i in ok]


def _load_meta(base: str, max_rows: int, target: str, non_last: bool):
    """Non-token targets: 'dataset' (source identity) or 'relpos' (position / prompt length, 5 equal bins)."""
    cols = ["z_L", "z_H", "prompt_ids", "position", "world", "dataset"] + (["is_last_prompt_pos"] if non_last else [])
    t = pq.read_table(base, columns=cols).slice(0, max_rows)
    n = t.num_rows
    zl = t.column("z_L").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    zh = t.column("z_H").combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1)
    ids, pos = t.column("prompt_ids").to_pylist(), t.column("position").to_pylist()
    world, ds = t.column("world").to_pylist(), t.column("dataset").to_pylist()
    last = t.column("is_last_prompt_pos").to_pylist() if non_last else [False] * n
    ok = [i for i in range(n) if not last[i]]
    if target == "dataset":
        names = sorted(set(ds))
        y = np.array([names.index(ds[i]) for i in ok])
    else:
        y = np.array([min(int(5 * pos[i] / len(ids[i])), 4) for i in ok])
    return zl[ok], zh[ok], y, [world[i] for i in ok]


def _fit_eval(x_tr, y_tr, x_te, y_te, n_cls, device, steps=300, wd=1e-3, hidden: int = 0) -> float:
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(x_tr.shape[1], hidden), torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, n_cls)).to(device)
        lr = 1e-3
    else:
        model = torch.nn.Linear(x_tr.shape[1], n_cls).to(device)
        lr = 1e-2
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (model(x_te).argmax(-1) == y_te).float().mean().item()


def _ci(acc: float, n: int) -> float:
    return 1.96 * (acc * (1 - acc) / max(n, 1)) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base.parquet from stage0_hrm (needs prompt_ids/position)")
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--max-rows", type=int, default=60000)
    p.add_argument("--device", default=default_device())
    p.add_argument("--offsets", type=int, nargs="+", default=[0])
    p.add_argument("--non-last", action="store_true", help="drop the last prompt position of each prompt")
    p.add_argument("--mlp-hidden", type=int, default=0, help="also fit a one-hidden-layer MLP probe with this width")
    p.add_argument("--train-sizes", type=int, nargs="+", default=None, help="learning curve: train-set sizes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", choices=["token", "dataset", "relpos"], default="token",
                   help="token: identity of the token at position + offset; dataset: source identity; relpos: 5 position bins")
    args = p.parse_args()

    for offset in args.offsets if args.target == "token" else [0]:
        if args.target == "token":
            zl, zh, tok, world = _load(args.base, args.max_rows, offset, args.non_last)
            top = [t for t, _ in Counter(tok.tolist()).most_common(args.top_k)]
            keep = np.isin(tok, top)
            remap = {t: i for i, t in enumerate(top)}
            y = np.array([remap[t] for t in tok[keep]])
            cover = f"top-{len(top)} tokens cover {keep.mean():.0%} of rows"
        else:
            zl, zh, y_all, world = _load_meta(args.base, args.max_rows, args.target, args.non_last)
            top = sorted(set(y_all.tolist()))
            remap = {t: i for i, t in enumerate(top)}
            y = np.array([remap[t] for t in y_all])
            keep = np.ones(len(y), dtype=bool)
            cover = f"{len(top)} classes ({args.target})"
        is_test = np.array([int(hashlib.sha256(w.encode()).hexdigest(), 16) % 5 == 0 for w in np.array(world)[keep]])
        zl, zh = zl[keep], zh[keep]
        label = f"offset {offset:+d}" if args.target == "token" else f"target {args.target}"
        print(f"\n[{label}{', non-last' if args.non_last else ''}] rows={len(y)} ({cover}), "
              f"train={int((~is_test).sum())} test={int(is_test.sum())}")
        assert is_test.any() and (~is_test).any(), "need rows in both train and test (more worlds)"
        maj = Counter(y[~is_test].tolist()).most_common(1)[0][0]
        print(f"majority-class baseline accuracy: {(y[is_test] == maj).mean():.3f}   (chance {1 / len(top):.3f})")

        feats = {"z_L": zl, "z_H": zh, "sum": zl + zh, "concat": np.concatenate([zl, zh], 1)}
        yt = torch.tensor(y, device=args.device)
        te = torch.tensor(is_test, device=args.device)
        n_te = int(is_test.sum())
        sizes = args.train_sizes or [None]
        for size in sizes:
            for name, x in feats.items():
                xt = torch.tensor(x, device=args.device)
                x_tr, y_tr = xt[~te], yt[~te]
                if size is not None and size < len(y_tr):
                    sel = torch.randperm(len(y_tr), generator=torch.Generator().manual_seed(args.seed))[:size].to(args.device)
                    x_tr, y_tr = x_tr[sel], y_tr[sel]
                tag = f" [train {len(y_tr)}]" if size is not None else ""
                acc = _fit_eval(x_tr, y_tr, xt[te], yt[te], len(top), args.device)
                line = f"  linear probe on {name:7s}{tag}: test accuracy {acc:.3f} +- {_ci(acc, n_te):.3f}"
                if args.mlp_hidden:
                    acc2 = _fit_eval(x_tr, y_tr, xt[te], yt[te], len(top), args.device, hidden=args.mlp_hidden)
                    line += f"   | MLP({args.mlp_hidden}) {acc2:.3f} +- {_ci(acc2, n_te):.3f}"
                print(line)


if __name__ == "__main__":
    main()
