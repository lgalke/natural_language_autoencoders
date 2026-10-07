import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nla.hrm.probe_check import _fit_eval, _load


def _planted(tmp_path, n_prompts=300, d=24):
    """z_H encodes the token at the position, z_L encodes the NEXT token."""
    rng = np.random.default_rng(0)
    seqs = [list(map(int, rng.integers(0, 5, size=20))) for _ in range(n_prompts)]
    rows = [(j, p, s) for j, s in enumerate(seqs) for p in range(4, 20, 4)]
    tok = np.array([s[p] for _, p, s in rows])
    nxt = np.array([s[min(p + 1, 19)] for _, p, s in rows])
    emb = rng.normal(size=(5, d))
    zh = emb[tok] + 0.05 * rng.normal(size=(len(rows), d))
    zl = emb[nxt] + 0.05 * rng.normal(size=(len(rows), d))
    t = pa.table({"z_L": [r.tolist() for r in zl], "z_H": [r.tolist() for r in zh],
                  "prompt_ids": [s for _, _, s in rows], "position": [p for _, p, _ in rows],
                  "world": [f"w{j}" for j, _, _ in rows], "is_last_prompt_pos": [p == 19 for _, p, _ in rows]})
    path = tmp_path / "base.parquet"
    pq.write_table(t, path)
    return str(path)


def test_offsets_and_non_last_filter(tmp_path):
    path = _planted(tmp_path)
    zl0, zh0, tok0, _ = _load(path, 10**6, offset=0, non_last=True)
    zl1, zh1, tok1, _ = _load(path, 10**6, offset=1, non_last=True)
    assert len(tok0) == len(tok1) == len(zl0)           # no prompt ends at a probed position here
    _, _, tok_all, _ = _load(path, 10**6, offset=3, non_last=False)
    assert len(tok_all) <= len(tok0) + 300 * 4          # rows whose position + offset leaves the prompt are dropped


def test_probe_recovers_the_planted_stream_and_offset(tmp_path):
    import torch

    path = _planted(tmp_path)
    for offset, good, bad in ((0, "z_H", "z_L"), (1, "z_L", "z_H")):
        zl, zh, tok, _ = _load(path, 10**6, offset=offset, non_last=True)
        n = len(tok)
        tr, te = slice(0, int(n * 0.8)), slice(int(n * 0.8), n)
        y = torch.tensor(tok)
        acc = {}
        for name, x in (("z_L", zl), ("z_H", zh)):
            xt = torch.tensor(x)
            acc[name] = _fit_eval(xt[tr], y[tr], xt[te], y[te], 5, "cpu", steps=200)
        assert acc[good] > 0.95 and acc[bad] < 0.5, (offset, acc)
