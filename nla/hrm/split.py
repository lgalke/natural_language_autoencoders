"""Split stage-0 output into training buckets + eval splits.

Two orthogonal holdouts, per the user's spec:
  - eval_ood: EVERY row whose `dataset` == --holdout-dataset (default "musr")
    goes here, entirely — a whole source held out for cross-dataset
    generalization, never touched by training.
  - Everything else is partitioned by `world` (group-aware, like
    `nla/datagen/stage1_split.py`'s doc-level split, but grouped on `world`
    instead of `doc_id` since one prompt can contribute multiple positions/
    rows and HRMMix's `world` may itself group several prompts) into
    av_sft / ar_sft / rl / eval_iid.

The fixed judge subset (used by `judge.py`'s periodic eval, never for
training) is written separately from eval_iid + eval_ood — see --judge-frac.
"""

import argparse
import random
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, make_storage
from nla.hrm.sidecar import read_sidecar, write_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base.parquet from stage0_hrm")
    p.add_argument("--holdout-dataset", default="musr",
                    help="'dataset' value held out entirely as eval_ood")
    p.add_argument("--av-sft-frac", type=float, default=0.30)
    p.add_argument("--ar-sft-frac", type=float, default=0.30)
    p.add_argument("--rl-frac", type=float, default=0.30)
    p.add_argument("--eval-iid-frac", type=float, default=0.10)
    p.add_argument("--judge-frac", type=float, default=0.5,
                    help="fraction of (eval_iid ∪ eval_ood) rows written to the fixed judge subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)
    fracs = (args.av_sft_frac, args.ar_sft_frac, args.rl_frac, args.eval_iid_frac)
    assert all(f >= 0 for f in fracs), f"fractions must be non-negative, got {fracs}"
    assert abs(sum(fracs) - 1.0) < 1e-6, f"fractions must sum to 1.0, got {sum(fracs)}"

    base_meta = read_sidecar(storage, args.base)
    assert base_meta.stage == "base", f"expected stage=base, got stage={base_meta.stage!r}"

    pf = pq.ParquetFile(storage.open_read(args.base))
    idx_cols = pf.read(columns=["world", "dataset"])
    worlds = idx_cols.column("world").to_pylist()
    datasets = idx_cols.column("dataset").to_pylist()

    ood_worlds = sorted({w for w, d in zip(worlds, datasets, strict=True) if d == args.holdout_dataset})
    other_worlds = sorted({w for w, d in zip(worlds, datasets, strict=True) if d != args.holdout_dataset})
    assert other_worlds, (
        f"--holdout-dataset={args.holdout_dataset!r} matched every row — nothing left to train on"
    )
    if not ood_worlds:
        print(f"[split] WARNING: no rows with dataset={args.holdout_dataset!r} — eval_ood will be empty")

    rng = random.Random(args.seed)
    rng.shuffle(other_worlds)
    n = len(other_worlds)
    n_av = int(n * args.av_sft_frac)
    n_ar = int(n * args.ar_sft_frac)
    n_rl = int(n * args.rl_frac)
    world_buckets = {
        "av_sft": set(other_worlds[:n_av]),
        "ar_sft": set(other_worlds[n_av : n_av + n_ar]),
        "rl": set(other_worlds[n_av + n_ar : n_av + n_ar + n_rl]),
        "eval_iid": set(other_worlds[n_av + n_ar + n_rl :]),
        "eval_ood": set(ood_worlds),
    }

    schema = pf.schema_arrow
    out_paths = {s: f"{args.output_dir.rstrip('/')}/{s}.parquet" for s in world_buckets}
    for path in out_paths.values():
        storage.ensure_parent(path)
    writers = {s: pq.ParquetWriter(storage.open_write(out_paths[s]), schema) for s in world_buckets}
    row_counts = {s: 0 for s in world_buckets}

    judge_rng = random.Random(args.seed + 1)
    judge_rows: list[pa.RecordBatch] = []

    for batch in pf.iter_batches(batch_size=65536):
        batch_worlds = batch.column("world").to_pylist()
        for stage, bucket in world_buckets.items():
            mask = pa.array([w in bucket for w in batch_worlds], type=pa.bool_())
            subset = batch.filter(mask)
            if subset.num_rows == 0:
                continue
            writers[stage].write_table(pa.Table.from_batches([subset]))
            row_counts[stage] += subset.num_rows
            if stage in ("eval_iid", "eval_ood"):
                keep = pa.array(
                    [judge_rng.random() < args.judge_frac for _ in range(subset.num_rows)], type=pa.bool_()
                )
                judged = subset.filter(keep)
                if judged.num_rows > 0:
                    judge_rows.append(judged)

    for w in writers.values():
        w.close()

    judge_path = f"{args.output_dir.rstrip('/')}/judge_subset.parquet"
    storage.ensure_parent(judge_path)
    with pq.ParquetWriter(storage.open_write(judge_path), schema) as jw:
        for rb in judge_rows:
            jw.write_table(pa.Table.from_batches([rb]))
    n_judge = sum(rb.num_rows for rb in judge_rows)
    write_sidecar(storage, judge_path, replace(
        base_meta, dataset_id=f"{base_meta.dataset_id}__judge_subset", stage="base", row_count=n_judge,
        parent_datasets=[base_meta.dataset_id], created_by="nla.hrm.split", created_at="", git_commit="",
    ))

    for stage, bucket in world_buckets.items():
        sub_meta = replace(
            base_meta,
            dataset_id=f"{base_meta.dataset_id}__{stage}",
            stage="base",  # still base-schema rows — build.py produces the real av_sft/ar_sft/rl parquets
            row_count=row_counts[stage],
            parent_datasets=[base_meta.dataset_id],
            created_by="nla.hrm.split",
            created_at="",
            git_commit="",
        )
        write_sidecar(storage, out_paths[stage], sub_meta)
        print(f"{stage}: {len(bucket)} worlds -> {row_counts[stage]} rows -> {out_paths[stage]}")
    print(f"judge_subset: {n_judge} rows -> {judge_path} "
          f"(sampled from eval_iid + eval_ood at judge_frac={args.judge_frac})")


if __name__ == "__main__":
    main()
