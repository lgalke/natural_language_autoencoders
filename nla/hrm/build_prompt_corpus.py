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
from dataclasses import dataclass

from datasets import load_dataset

# Default source specs: public HRMMix reasoning sources + DA/EN instruction
# prompts. `--source` on the CLI is additive to these — pass `--no-defaults`
# to start from an empty list. Hub IDs are deliberately overridable: swap any
# entry (or the DA/EN instruction IDs) for what's actually reachable in your
# environment via repeated `--source`.
DEFAULT_SOURCES = [
    "gsm-symbolic=apple/GSM-Symbolic:main:test:question",
    # "proofwriter=voidful/ProofWriter::validation:question",
    "bbh=lukaemon/bbh:causal_judgement:test:input",
    "musr=TAUR-Lab/MuSR::murder_mysteries:narrative",
    "simplestories=SimpleStories/SimpleStories::train:story",
    "da_instruct=danish-foundation-models/danish-dynaword::train:text",
]


@dataclass
class SourceSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    column: str
    max_rows: int | None = None  # optional per-source cap (5th field), else --max-per-source

    @staticmethod
    def parse(spec: str) -> "SourceSpec":
        name, _, rest = spec.partition("=")
        assert name and rest, f"--source must be 'name=hf_id[:config]:split:column[:max_rows]', got {spec!r}"
        parts = rest.split(":")
        assert len(parts) in (4, 5), (
            f"--source {spec!r}: expected hf_id[:config]:split:column[:max_rows] (4 or 5 colon-separated "
            f"fields, empty config allowed), got {len(parts)} fields: {parts}"
        )
        hf_id, config, split, column = parts[:4]
        max_rows = int(parts[4]) if len(parts) == 5 and parts[4] else None
        return SourceSpec(name=name, hf_id=hf_id, config=config or None, split=split, column=column, max_rows=max_rows)


def _rows_from_source(spec: SourceSpec, max_rows: int | None, seed: int, config_in_doc_id: bool = False) -> list[dict]:
    max_rows = spec.max_rows if spec.max_rows is not None else max_rows
    # Streaming + buffered shuffle: big sources (SimpleStories is ~2M rows) are
    # never fully downloaded, only ~max_rows (+ shuffle buffer) are read.
    ds = load_dataset(spec.hf_id, name=spec.config, split=spec.split, streaming=True)
    assert spec.column in (ds.column_names or [spec.column]), (
        f"source {spec.name!r} ({spec.hf_id}): no column {spec.column!r}; available: {ds.column_names}"
    )
    if max_rows is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000).take(max_rows)
    rows = []
    for i, row in enumerate(ds):
        prompt = row[spec.column]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append({
            "prompt": prompt.strip(),
            "dataset": spec.name,
            "world": f"{spec.name}:{i}",
            # two sources of one Hub dataset (e.g. ARC-Easy / ARC-Challenge) would collide: add the config then
            "doc_id": f"{spec.hf_id}:{spec.config}:{spec.split}:{i}" if config_in_doc_id else f"{spec.hf_id}:{spec.split}:{i}",
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


def prompt_hash(prompt: str) -> str:
    import hashlib
    return hashlib.sha1(" ".join(prompt.split()).encode("utf-8")).hexdigest()


def filter_local_rows(rows: list[dict], exclude_hashes: set[str], drop_datasets: set[str],
                      max_per_dataset: int | None, seed: int) -> tuple[list[dict], dict[str, int]]:
    """Local-corpus hygiene: drop datasets (e.g. the held-out OOD source), drop prompts that already occur in another
    corpus (whitespace-normalised exact match: keeps a v2 training corpus clear of the v1 eval prompts), then cap the rows
    per dataset with a seeded sample. Returns (rows, counts of what was removed)."""
    import random
    stats = {"dropped_dataset": 0, "dropped_duplicate": 0, "dropped_cap": 0}
    kept = []
    for r in rows:
        if r["dataset"] in drop_datasets:
            stats["dropped_dataset"] += 1
        elif exclude_hashes and prompt_hash(r["prompt"]) in exclude_hashes:
            stats["dropped_duplicate"] += 1
        else:
            kept.append(r)
    if max_per_dataset is not None:
        by: dict[str, list[dict]] = {}
        for r in kept:
            by.setdefault(r["dataset"], []).append(r)
        kept = []
        for name in sorted(by):
            rs = by[name]
            random.Random(f"{seed}|{name}").shuffle(rs)
            kept += rs[:max_per_dataset]
            stats["dropped_cap"] += max(0, len(rs) - max_per_dataset)
    return kept, stats


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
    p.add_argument("--strict", action="store_true", help="fail on the first broken source instead of skipping it")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-dataset", "--local-max-per-dataset", dest="max_per_dataset", type=int, default=None,
                   help="final seeded cap on the rows per 'dataset' value over ALL sources (Hub and local); "
                        "--max-per-source / the optional 5th field of --source cap a source before this")
    p.add_argument("--exclude-from", action="append", default=[],
                   help="corpus JSONL whose prompts are removed from ALL rows (whitespace-normalised exact match); "
                        "use the v1 corpus.jsonl so the v2 training corpus does not contain the v1 eval prompts")
    p.add_argument("--drop-dataset", action="append", default=[],
                   help="drop rows of this 'dataset' value (repeatable), e.g. the held-out OOD source musr")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    specs = [] if args.no_defaults else [SourceSpec.parse(s) for s in DEFAULT_SOURCES]
    specs += [SourceSpec.parse(s) for s in args.source]
    assert specs or args.local_jsonl, "no sources: pass --source, or --local-jsonl, or drop --no-defaults"

    hf_id_counts: dict[str, int] = {}
    for spec in specs:
        hf_id_counts[spec.hf_id] = hf_id_counts.get(spec.hf_id, 0) + 1
    all_rows: list[dict] = []
    for spec in specs:
        try:
            rows = _rows_from_source(spec, args.max_per_source, args.seed,
                                     config_in_doc_id=hf_id_counts[spec.hf_id] > 1)
        except Exception as e:  # noqa: BLE001 — one unreachable/misconfigured source shouldn't kill the run
            if args.strict:
                raise
            print(f"  WARNING: skipping source {spec.name!r}: {e.__class__.__name__}: {e}")
            continue
        print(f"  {spec.name}: {len(rows)} rows from {spec.hf_id} ({spec.split})")
        all_rows += rows
    for path in args.local_jsonl:
        rows = _rows_from_local_jsonl(path, args.local_jsonl_dataset_name)
        print(f"  local:{path}: {len(rows)} rows")
        all_rows += rows

    exclude: set[str] = set()
    for ex in args.exclude_from:
        with open(ex) as f:
            exclude |= {prompt_hash(json.loads(line)["prompt"]) for line in f if line.strip()}
    if exclude or args.drop_dataset or args.max_per_dataset is not None:
        all_rows, stats = filter_local_rows(all_rows, exclude, set(args.drop_dataset), args.max_per_dataset, args.seed)
        print(f"  filters ({len(exclude)} excluded prompt hashes): {stats} -> {len(all_rows)} rows kept")

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
