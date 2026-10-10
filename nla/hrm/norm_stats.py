"""Fit the mean-predictor MSE baselines (for FVE reporting) and the injection
scale init, on TRAIN buckets only (av_sft + ar_sft + rl — never eval_iid/
eval_ood, or FVE would be optimistic and the judge would leak).

Two independent numbers, both needed before training starts:
  - mean_mse.{sum,L,H}: MSE of the constant (per-term mean) predictor in the
    shared-normalized space (`nla/hrm/recon.py:shared_normalize`). This is
    the FVE=0 baseline — `ReconLoss.fve()` divides by it.
  - injection_scale_p75: the 75th-percentile residual-stream norm of the
    VERBALIZER (Qwen2.5-1.5B-Instruct) over a small text sample, used to
    init the per-stream injection adapter's scale "large" (the original
    NLA's heuristic — see docs/hrm.md). Optional (`--skip-injection-scale`)
    since it needs the verbalizer downloaded; norm_stats.py otherwise has no
    dependency on the verbalizer at all.
"""

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from nla.hrm.devices import default_device
from nla.hrm.recon import shared_normalize


def _load_zLzH(parquet_paths: list[str], max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    zLs, zHs = [], []
    n = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8192, columns=["z_L", "z_H"]):
            m = len(batch)
            zLs.append(batch.column("z_L").flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(m, -1))
            zHs.append(batch.column("z_H").flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(m, -1))
            n += m
            if n >= max_rows:
                break
        if n >= max_rows:
            break
    return np.concatenate(zLs, axis=0)[:max_rows], np.concatenate(zHs, axis=0)[:max_rows]


def fit_mean_mse(z_L: torch.Tensor, z_H: torch.Tensor) -> dict[str, float]:
    """MSE of the constant-mean predictor, in shared-normalized space (matches
    the frame `recon.recon_loss` scores predictions in)."""
    z_L_n, z_H_n, s_n = shared_normalize(z_L, z_H)
    mu_L, mu_H, mu_s = z_L_n.mean(0, keepdim=True), z_H_n.mean(0, keepdim=True), s_n.mean(0, keepdim=True)
    return {
        "sum": ((s_n - mu_s) ** 2).mean().item(),
        "L": ((z_L_n - mu_L) ** 2).mean().item(),
        "H": ((z_H_n - mu_H) ** 2).mean().item(),
    }


def fit_injection_scale_p75(verbalizer_model: str, sample_texts: list[str], device: str = "cpu",
                             dtype: torch.dtype = torch.float32) -> float:
    """75th-percentile per-token residual-stream norm of the verbalizer's LAST
    hidden layer, over `sample_texts`. Used to init the injection adapter's
    scale "large" (original NLA heuristic: injected vectors should land near
    the ambient residual scale the verbalizer already operates at, not near
    its token-embedding scale)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(verbalizer_model)
    model = AutoModelForCausalLM.from_pretrained(verbalizer_model, torch_dtype=dtype).to(device).eval()
    norms = []
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            print(f"  injection-scale sample {i + 1}/{len(sample_texts)}", flush=True)
            ids = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            out = model(**ids, output_hidden_states=True, logits_to_keep=1)
            h = out.hidden_states[-1][0]  # [T, d]
            norms.append(h.float().norm(dim=-1).cpu())
    all_norms = torch.cat(norms).numpy()
    return float(np.percentile(all_norms, 75))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-parquet", action="append", required=True,
                    help="av_sft/ar_sft/rl parquet(s) from split.py — repeatable")
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--verbalizer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--skip-injection-scale", action="store_true")
    p.add_argument("--device", default=default_device())
    p.add_argument("--injection-scale-n-samples", type=int, default=32)
    p.add_argument("--output", required=True, help="output JSON path")
    args = p.parse_args()

    z_L_np, z_H_np = _load_zLzH(args.train_parquet, args.max_rows)
    z_L, z_H = torch.from_numpy(z_L_np), torch.from_numpy(z_H_np)
    mean_mse = fit_mean_mse(z_L, z_H)
    print(f"mean_mse: sum={mean_mse['sum']:.4f} L={mean_mse['L']:.4f} H={mean_mse['H']:.4f} (n={z_L.shape[0]})", flush=True)

    injection_scale_p75 = None
    if not args.skip_injection_scale:
        texts = []
        for path in args.train_parquet:
            pf = pq.ParquetFile(path)
            if "context_marked" in pf.schema_arrow.names:
                col = pf.read(columns=["context_marked"]).column("context_marked").to_pylist()
                texts += [c.replace("⟦", "").replace("⟧", "") for c in col[: args.injection_scale_n_samples]]
        texts = texts[: args.injection_scale_n_samples] or ["The quick brown fox jumps over the lazy dog."]
        injection_scale_p75 = fit_injection_scale_p75(
            args.verbalizer_model, texts, args.device,
            torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
        print(f"injection_scale_p75 ({args.verbalizer_model}): {injection_scale_p75:.2f}")

    out = {
        "d_model": int(z_L.shape[1]),
        "n_rows": int(z_L.shape[0]),
        "mean_mse": mean_mse,
        "verbalizer_model": args.verbalizer_model,
        "injection_scale_p75": injection_scale_p75,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
