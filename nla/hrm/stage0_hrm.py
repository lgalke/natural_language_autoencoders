"""Stage 0 (HRM variant): forward Mimir over a prompt corpus, capture z_L/z_H at
the fixed hook site (see `nla/hrm/mimir.py` module docstring), sample token
positions, write RAW vectors to parquet.

Not a drop-in for `nla/datagen/stage0_extract.py`: that script hooks a single
decoder layer of a plain causal LM and samples deep into long raw-text
documents (`_MIN_POSITION=50`). Here the whole "document" is one short
rendered chat prompt, attended to bidirectionally end-to-end (PrefixLM), and
we capture TWO recurrent states per position instead of one per-layer state.

Vectors are stored UNNORMALIZED (norm="none", matching the repo invariant —
normalization is a training-time decision, see `nla/hrm/recon.py:shared_normalize`).
"""

import argparse
import hashlib
import random

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from nla.datagen._common import add_storage_args, make_storage
from nla.hrm.mimir import HrmStreamCapture, context_marked, load_mimir, render_and_encode_batch, render_prompts
from nla.hrm.sidecar import HrmDatasetMeta, HrmExtractionMeta, write_sidecar

# Skip the first few template tokens (<bos><|turn>user\n) — not meaningful
# "content" positions. Unlike stage0_extract's 50 (deep left-context for long
# docs), these prompts are short chat turns; the template prefix is ~4-6
# tokens depending on BPE merges at the content boundary.
_MIN_POSITION = 3


def _sample_positions(length: int, n_positions: int, doc_id: str, seed: int) -> list[int]:
    """Per-doc keyed RNG (same convention as stage0_extract._sample_positions):
    same (seed, doc_id) -> same positions regardless of batching/chunking.
    Always includes the last prompt position (length-1) — the point right
    before generation would begin, and the position HRM-Interp's probes call
    the "answer position"."""
    last = length - 1
    rng = random.Random(hashlib.sha256(f"{seed}|{doc_id}".encode()).digest())
    candidates = [i for i in range(_MIN_POSITION, last)]
    k = min(max(n_positions - 1, 0), len(candidates))
    positions = rng.sample(candidates, k=k) if candidates else []
    positions.append(last)
    return sorted(set(positions))


def _schema(d_model: int) -> pa.Schema:
    return pa.schema([
        ("z_L", pa.list_(pa.float32(), d_model)),
        ("z_H", pa.list_(pa.float32(), d_model)),
        ("position", pa.int64()),
        ("prompt_len", pa.int64()),
        ("is_last_prompt_pos", pa.bool_()),
        ("dataset", pa.string()),
        ("world", pa.string()),
        ("doc_id", pa.string()),
        ("context_marked", pa.string()),
        # The EXACT Mimir input_ids for this row's prompt (up to prompt_len,
        # i.e. with right-padding already stripped) — `judge.py` re-renders
        # from this (not from re-tokenizing `context_marked`'s decoded text,
        # which is a lossy round-trip through detokenize+retokenize) so its
        # patch-back forward pass is bit-identical to the one that produced
        # z_L/z_H here. Variable-length ListArray (not FixedSizeList) since
        # prompt_len varies per row.
        ("prompt_ids", pa.list_(pa.int64())),
    ])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="JSONL from build_prompt_corpus.py")
    p.add_argument("--base-model", default=None, help="defaults to nla.hrm.mimir.DEFAULT_MIMIR")
    p.add_argument("--positions-per-prompt", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8, help="max prompts per batch")
    p.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="cap on batch_size x padded_len; prompts are length-sorted so long "
                         "ones get small batches (attention memory is quadratic in length)")
    p.add_argument("--max-prompt-tokens", type=int, default=2048,
                    help="skip prompts whose rendered length exceeds this (never truncated: "
                         "cutting the chat template would change what the model sees)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="only process the first N corpus rows (smoke test)")
    p.add_argument("--output", required=True)
    add_storage_args(p)
    args = p.parse_args()

    import json

    storage = make_storage(args)

    from nla.hrm.mimir import DEFAULT_MIMIR
    base_model = args.base_model or DEFAULT_MIMIR
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_mimir(base_model, device=args.device, torch_dtype=dtype)
    d_model = model.config.hidden_size

    corpus_rows = []
    with open(args.corpus) as f:
        for line in f:
            line = line.strip()
            if line:
                corpus_rows.append(json.loads(line))
    if args.limit is not None:
        corpus_rows = corpus_rows[: args.limit]
    assert corpus_rows, f"no rows read from {args.corpus}"

    # Length-sort + token-budget batching (row order in the parquet doesn't matter;
    # positions are keyed on (seed, doc_id), not batch composition).
    lens = [len(t) for t in (tokenizer(render_prompts(tokenizer, [c["prompt"] for c in corpus_rows]),
                                        add_special_tokens=False)["input_ids"])]
    keep = [i for i, n in enumerate(lens) if n <= args.max_prompt_tokens]
    print(f"skipping {len(lens) - len(keep)}/{len(lens)} prompts longer than {args.max_prompt_tokens} tokens")
    keep.sort(key=lambda i: lens[i])
    batches: list[list[dict]] = []
    cur: list[int] = []
    for i in keep:
        if cur and (len(cur) >= args.batch_size or (len(cur) + 1) * lens[i] > args.max_batch_tokens):
            batches.append([corpus_rows[j] for j in cur])
            cur = []
        cur.append(i)
    if cur:
        batches.append([corpus_rows[j] for j in cur])
    schema = _schema(d_model)
    storage.ensure_parent(args.output)
    row_count = 0
    n_prompts_skipped = 0

    with pq.ParquetWriter(storage.open_write(args.output), schema) as writer:
        for chunk in tqdm(batches, desc="batches"):
            prompts = [c["prompt"] for c in chunk]
            batch = render_and_encode_batch(tokenizer, prompts, device=args.device)

            with HrmStreamCapture(model) as cap:
                with torch.no_grad():
                    model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        token_type_ids=batch.token_type_ids,
                        use_cache=False,
                        # We only need the hooked hidden states — without this the
                        # LM head materializes [B, S, vocab=262144] logits (OOM).
                        logits_to_keep=1,
                    )
            cap.verify_call_counts()
            # bf16 forward: sum identity holds only approximately (rounding
            # accumulates over 8 layer applications) — loosen tolerance vs.
            # the float32 default used by tests.
            atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
            cap.verify_sum_identity(atol=atol, rtol=atol)
            assert cap.z_L is not None and cap.z_H is not None  # narrow for the type checker

            rows: dict[str, list] = {k: [] for k in schema.names}
            for b, meta in enumerate(chunk):
                length = batch.lengths[b]
                if length <= _MIN_POSITION + 1:
                    n_prompts_skipped += 1
                    continue
                positions = _sample_positions(
                    length, args.positions_per_prompt, meta["doc_id"], args.seed
                )
                for pos in positions:
                    rows["z_L"].append(cap.z_L[b, pos].tolist())
                    rows["z_H"].append(cap.z_H[b, pos].tolist())
                    rows["position"].append(pos)
                    rows["prompt_len"].append(length)
                    rows["is_last_prompt_pos"].append(pos == length - 1)
                    rows["dataset"].append(meta["dataset"])
                    rows["world"].append(meta.get("world", meta["doc_id"]))
                    rows["doc_id"].append(meta["doc_id"])
                    rows["context_marked"].append(
                        context_marked(tokenizer, batch.input_ids[b], pos, length)
                    )
                    rows["prompt_ids"].append(batch.input_ids[b, :length].tolist())

            if rows["doc_id"]:
                writer.write_table(pa.Table.from_pydict(rows, schema=schema))
                row_count += len(rows["doc_id"])

    meta = HrmDatasetMeta(
        dataset_id=f"hrm_base_{base_model.split('/')[-1]}_{hashlib.sha256(args.corpus.encode()).hexdigest()[:8]}",
        stage="base",
        row_count=row_count,
        extraction=HrmExtractionMeta(
            base_model=base_model,
            d_model=d_model,
            hook_site="H_in@2 (z_L=L_out@2.3, z_H=H_out@1)",
            L_cycles=model.config.L_cycles,
            H_cycles=model.config.H_cycles,
            norm="none",
            corpus=args.corpus,
            positions_per_prompt=args.positions_per_prompt,
        ),
        created_by="nla.hrm.stage0_hrm",
    )
    write_sidecar(storage, args.output, meta)
    print(f"wrote {row_count} rows ({len(keep) - n_prompts_skipped}/{len(lens)} prompts) → {args.output}")
    if n_prompts_skipped:
        print(f"  skipped {n_prompts_skipped} prompts shorter than {_MIN_POSITION + 2} tokens")
    print(f"sidecar → {args.output}.nla_meta.yaml")


if __name__ == "__main__":
    main()
