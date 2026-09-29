"""Build the HRM prompt corpus — JSONL of `{"prompt","dataset","world","doc_id"}`
rows, one row per prompt, in the schema HRM-Interp's `linear_probes.py --task
cycle` already uses (`cycle_counter_probes.md` "Data schema").

Two kinds of source:
  --source NAME=HF_ID[:CONFIG]:SPLIT:COLUMN   pull from the HF Hub
  --local-jsonl PATH                          pass rows through (HRMMix; each
                                                row needs at least "prompt")

`world` is the semantic-cluster id used for the group-aware train/test split
in `split.py` (so no near-duplicate prompt crosses a bucket boundary). HF
sources default to one group per prompt (HRMMix is expected to carry real
clusters and is passed through with whatever `world` it already has).

Mixed by default: the public HRMMix reasoning sources (GSM-Symbolic,
ProofWriter, BBH, MuSR) plus DA/EN instruction prompts. Point `--local-jsonl`
at the full HRMMix corpus once its local path is available — this script
doesn't care whether HRMMix comes from the Hub subset or the local file, only
that every row ends up with prompt/dataset/world/doc_id.
"""

import argparse
import json
import random
from dataclasses import dataclass

from datasets import load_dataset

# Default source specs: public HRMMix reasoning sources + DA/EN instruction
# prompts. `--source` on the CLI is additive to these — pass `--no-defaults`
# to start from an empty list. Hub IDs are deliberately overridable: swap any
# entry (or the DA/EN instruction IDs) for what's actually reachable in your
# environment via repeated `--source`.
DEFAULT_SOURCES = [
    "gsm-symbolic=apple/GSM-Symbolic:main:test:question",
    "proofwriter=voidful/ProofWriter::validation:question",
    "bbh=lukaemon/bbh:causal_judgement:test:input",
    "musr=TAUR-Lab/MuSR::murder_mysteries:context",
    "da_instruct=danish-foundation-models/danish-dynaword::train:text",
]


@dataclass
class SourceSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    column: str

    @staticmethod
    def parse(spec: str) -> "SourceSpec":
        name, _, rest = spec.partition("=")
        assert name and rest, f"--source must be 'name=hf_id[:config]:split:column', got {spec!r}"
        parts = rest.split(":")
        assert len(parts) == 4, (
            f"--source {spec!r}: expected hf_id[:config]:split:column (4 colon-separated "
            f"fields, empty config allowed), got {len(parts)} fields: {parts}"
        )
        hf_id, config, split, column = parts
        return SourceSpec(name=name, hf_id=hf_id, config=config or None, split=split, column=column)


def _rows_from_source(spec: SourceSpec, max_rows: int | None, seed: int) -> list[dict]:
    ds = load_dataset(spec.hf_id, name=spec.config, split=spec.split)
    n = len(ds) if max_rows is None else min(max_rows, len(ds))
    if max_rows is not None and len(ds) > max_rows:
        idx = sorted(random.Random(f"{seed}|{spec.name}").sample(range(len(ds)), n))
        ds = ds.select(idx)
    rows = []
    for i, row in enumerate(ds):
        prompt = row[spec.column]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append({
            "prompt": prompt.strip(),
            "dataset": spec.name,
            "world": f"{spec.name}:{i}",
            "doc_id": f"{spec.hf_id}:{spec.split}:{i}",
        })
    return rows


def _rows_from_local_jsonl(path: str, default_dataset: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert "prompt" in row, f"{path}:{i + 1}: missing required 'prompt' field"
            rows.append({
                "prompt": row["prompt"],
                "dataset": row.get("dataset", default_dataset),
                "world": row.get("world", f"{row.get('dataset', default_dataset)}:{i}"),
                "doc_id": row.get("doc_id", f"{path}:{i}"),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", default=[],
                    help="name=hf_id[:config]:split:column — repeatable, additive to the defaults")
    p.add_argument("--no-defaults", action="store_true", help="don't include DEFAULT_SOURCES")
    p.add_argument("--local-jsonl", action="append", default=[],
                    help="path to a local JSONL corpus (e.g. HRMMix) — repeatable")
    p.add_argument("--local-jsonl-dataset-name", default="hrmmix",
                    help="'dataset' value for local-jsonl rows that don't already carry one")
    p.add_argument("--max-per-source", type=int, default=2000,
                    help="cap rows pulled per HF source (random subsample, keyed on --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    specs = [] if args.no_defaults else [SourceSpec.parse(s) for s in DEFAULT_SOURCES]
    specs += [SourceSpec.parse(s) for s in args.source]
    assert specs or args.local_jsonl, "no sources: pass --source, or --local-jsonl, or drop --no-defaults"

    all_rows: list[dict] = []
    for spec in specs:
        rows = _rows_from_source(spec, args.max_per_source, args.seed)
        print(f"  {spec.name}: {len(rows)} rows from {spec.hf_id} ({spec.split})")
        all_rows += rows
    for path in args.local_jsonl:
        rows = _rows_from_local_jsonl(path, args.local_jsonl_dataset_name)
        print(f"  local:{path}: {len(rows)} rows")
        all_rows += rows

    doc_ids = [r["doc_id"] for r in all_rows]
    assert len(doc_ids) == len(set(doc_ids)), "doc_id collision across sources — rename a --source"

    with open(args.output, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(all_rows)} rows → {args.output}")
    by_dataset = {}
    for r in all_rows:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
    for name, n in sorted(by_dataset.items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
