"""Cancellation/geometry diagnostics for the (z_L, z_H) stream pair.

Run before training, on the raw stage-0 output, to see how much z_L and z_H
cancel when summed (H's actual input) — strong cancellation means Mimir
stores more in each stream than H's second application reads, which should
inform `--w-sum`/`--w-comp` in the reconstruction loss (`nla/hrm/recon.py`).
The same `compute_stream_stats` function is reused as training-time logging
for (ẑ_L, ẑ_H) so real and reconstructed geometry are directly comparable.
"""

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch


def compute_stream_stats(z_L: torch.Tensor, z_H: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-row geometry stats. z_L, z_H: [..., d_model] (any leading dims)."""
    z_L, z_H = z_L.float(), z_H.float()
    s = z_L + z_H
    norm_L = z_L.norm(dim=-1)
    norm_H = z_H.norm(dim=-1)
    norm_s = s.norm(dim=-1)
    return {
        "norm_L": norm_L,
        "norm_H": norm_H,
        "norm_s": norm_s,
        # ||z_L+z_H|| / (||z_L||+||z_H||): 1.0 = no cancellation (orthogonal-ish,
        # aligned magnitudes), <1 = partial cancellation, →0 = near-opposite.
        "cancellation": norm_s / (norm_L + norm_H).clamp_min(1e-12),
        "cos_LH": (z_L * z_H).sum(-1) / (norm_L * norm_H).clamp_min(1e-12),
    }


def _summarize(x: np.ndarray) -> dict:
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "n": int(x.size),
    }


def _load_streams(parquet_path: str, max_rows: int | None) -> dict[str, np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    cols = ["z_L", "z_H", "dataset", "is_last_prompt_pos"]
    chunks = {c: [] for c in cols}
    n = 0
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        z_l = batch.column("z_L").flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        z_h = batch.column("z_H").flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        m = len(batch)
        chunks["z_L"].append(z_l.reshape(m, -1))
        chunks["z_H"].append(z_h.reshape(m, -1))
        chunks["dataset"].append(np.array(batch.column("dataset").to_pylist()))
        chunks["is_last_prompt_pos"].append(np.array(batch.column("is_last_prompt_pos").to_pylist()))
        n += m
        if max_rows is not None and n >= max_rows:
            break
    out = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
    if max_rows is not None:
        out = {k: v[:max_rows] for k, v in out.items()}
    return out


def run_diagnostics(parquet_path: str, max_rows: int | None = 200_000) -> dict:
    data = _load_streams(parquet_path, max_rows)
    stats = compute_stream_stats(torch.from_numpy(data["z_L"]), torch.from_numpy(data["z_H"]))
    cancellation = stats["cancellation"].numpy()
    cos_lh = stats["cos_LH"].numpy()
    norm_s = stats["norm_s"].numpy()

    summary = {
        "n_rows": int(data["z_L"].shape[0]),
        "overall": {
            "cancellation": _summarize(cancellation),
            "cos_LH": _summarize(cos_lh),
            "norm_L": _summarize(stats["norm_L"].numpy()),
            "norm_H": _summarize(stats["norm_H"].numpy()),
            "norm_s": _summarize(norm_s),
        },
        "by_dataset": {},
        "by_position_kind": {},
    }
    for name in np.unique(data["dataset"]):
        mask = data["dataset"] == name
        summary["by_dataset"][str(name)] = {
            "cancellation": _summarize(cancellation[mask]),
            "cos_LH": _summarize(cos_lh[mask]),
        }
    for kind, mask in (("last_prompt_pos", data["is_last_prompt_pos"]),
                        ("other", ~data["is_last_prompt_pos"])):
        if mask.any():
            summary["by_position_kind"][kind] = {
                "cancellation": _summarize(cancellation[mask]),
                "cos_LH": _summarize(cos_lh[mask]),
            }
    return summary


def _maybe_plot(parquet_path: str, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping histograms)")
        return
    from pathlib import Path
    data = _load_streams(parquet_path, 200_000)
    stats = compute_stream_stats(torch.from_numpy(data["z_L"]), torch.from_numpy(data["z_H"]))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, tensor in (("cancellation", stats["cancellation"]), ("cos_LH", stats["cos_LH"])):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(tensor.numpy(), bins=50)
        ax.set_title(name)
        fig.savefig(f"{out_dir}/{name}.png", dpi=100)
        plt.close(fig)
    print(f"  histograms → {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="stage0_hrm parquet (or any parquet with z_L/z_H cols)")
    p.add_argument("--output-json", default=None, help="defaults to {input}.diagnostics.json")
    p.add_argument("--plots-dir", default=None, help="defaults to {input}.diagnostics_plots/")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--max-rows", type=int, default=200_000)
    args = p.parse_args()

    summary = run_diagnostics(args.input, args.max_rows)
    out_json = args.output_json or f"{args.input}.diagnostics.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"n_rows={summary['n_rows']}")
    print(f"cancellation (||zL+zH||/(||zL||+||zH||)): mean={summary['overall']['cancellation']['mean']:.3f} "
          f"p10={summary['overall']['cancellation']['p10']:.3f} p90={summary['overall']['cancellation']['p90']:.3f}")
    print(f"cos(zL, zH): mean={summary['overall']['cos_LH']['mean']:.3f}")
    print(f"wrote {out_json}")

    if not args.no_plots:
        _maybe_plot(args.input, args.plots_dir or f"{args.input}.diagnostics_plots")


if __name__ == "__main__":
    main()
