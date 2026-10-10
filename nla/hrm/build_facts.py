"""Facts-only SFT rows: turn a base-schema bucket (e.g. splits_v2/rl.parquet) into a pseudo-"explained" parquet with EMPTY
teacher explanations, so that

    python -m nla.hrm.build --input X_facts.parquet --stage av_sft --prefix-token --position-fact --output av_sft_facts.parquet

writes AV-SFT rows whose L and H fields are only the programmatic fact sentences (`Marked token: "X". Position: K of 5.`):
a teacher-free readout curriculum on as many rows as there are extracted vectors. Then `split_av` converts them to the
one-stream-per-call format and `train_av_sft --init-from` continues with the teacher-prose rows.

    python -m nla.hrm.build_facts --input splits_v2/rl.parquet --output splits_v2/rl_facts.parquet [--limit N]
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.storage import LocalStorage
from nla.hrm.sidecar import read_sidecar, write_sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (smoke test)")
    args = p.parse_args()

    meta = read_sidecar(LocalStorage(), args.input)
    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow.append(pa.field("api_explanation_0", pa.string())).append(pa.field("api_explanation_1", pa.string()))
    n_out = 0
    with pq.ParquetWriter(args.output, schema) as writer:
        for batch in pf.iter_batches(batch_size=8192):
            if args.limit is not None and n_out >= args.limit:
                break
            if args.limit is not None:
                batch = batch.slice(0, args.limit - n_out)
            n = len(batch)
            empty = pa.array([""] * n, type=pa.string())
            writer.write_table(pa.Table.from_batches([batch]).append_column("api_explanation_0", empty)
                               .append_column("api_explanation_1", empty))
            n_out += n
    out_meta = type(meta)(**{**meta.__dict__, "dataset_id": f"{meta.dataset_id}__facts", "row_count": n_out,
                             "parent_datasets": [*meta.parent_datasets, meta.dataset_id], "created_by": "nla.hrm.build_facts",
                             "created_at": "", "git_commit": ""})
    write_sidecar(LocalStorage(), args.output, out_meta)
    print(f"wrote {n_out} rows with empty explanations → {args.output}")


if __name__ == "__main__":
    main()
