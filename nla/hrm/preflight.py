"""Quick environment/artifact check — run this first in a fresh session.

    python -m nla.hrm.preflight [--dir .]

Reports GPU availability, package versions, whether the expected pipeline
artifacts exist in --dir, and whether a provider key is set (never prints it).
Exit code 1 if a hard requirement (transformers>=5.13, peft) is missing.
"""

import argparse
import importlib.metadata as md
import os
from pathlib import Path


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


ARTIFACTS = {
    "corpus.jsonl": "prompt corpus", "base.parquet": "stage0 extraction", "norm_stats.json": "FVE baselines",
    "splits/": "raw split buckets", "av_sft.parquet": "AV-SFT data", "ar_sft.parquet": "AR-SFT data",
    "rl.parquet": "RL data", "eval_iid.parquet": "in-distribution eval", "eval_ood.parquet": "OOD eval (MuSR)",
    "judge_subset.parquet": "judge subset (needs sidecar, see docs/okf/observations/pitfalls.md)",
    "ckpt/ar_sft": "AR-SFT checkpoint", "ckpt/av_sft": "AV-SFT checkpoint", "ckpt/rl/final": "RL checkpoint",
    "nla/hrm/injection_token_cache_hrm.yaml": "marker cache (machine-local, git-ignored)",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default=".", help="directory holding the pipeline artifacts (default: cwd)")
    args = p.parse_args()
    bad = False

    import torch
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  gpu={torch.cuda.get_device_name(0)}  free={free / 2**30:.0f}/{total / 2**30:.0f} GiB")
    else:
        print("  (NO GPU: scripts will silently run on CPU and be very slow)")

    tf = _ver("transformers")
    ok_tf = tf is not None and tuple(int(x) for x in tf.split(".")[:2]) >= (5, 13)
    print(f"transformers {tf}  {'ok' if ok_tf else 'NEED >=5.13 (hrm_text support)'}")
    bad |= not ok_tf
    for pkg in ("peft", "pyarrow", "datasets", "httpx", "anthropic"):
        v = _ver(pkg)
        print(f"{pkg} {v}" + ("" if v else "  MISSING"))
        bad |= pkg == "peft" and v is None

    print("\nartifacts in", Path(args.dir).resolve())
    for rel, what in ARTIFACTS.items():
        base = Path(args.dir) / rel if not rel.startswith("nla/") else Path(rel)
        print(f"  [{'x' if base.exists() else ' '}] {rel:42s} {what}")

    print("\nprovider keys (presence only):")
    for var in ("UCLOUD_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        print(f"  {var}: {'set' if os.environ.get(var) else 'not set'}")
    print(f"  ./.env present: {Path('.env').exists()}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
