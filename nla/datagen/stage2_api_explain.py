"""Stage 2: API explanations — SL subset → +api_explanation column.

Call an external LLM to produce natural-language explanations of each row's
source text (`detokenized_text_truncated`). The explanation becomes the
`response` for AV-SFT (actor SFT) and the `<text>` content for AR-SFT (critic SFT).
RL subset skips this stage — actor generates responses during rollout.

The completion backend is pluggable via CompletionProvider. Default: Anthropic.

Processes in chunks for bounded memory + progress visibility. Each completed
chunk is written to {output}.chunks/chunk_{N}.parquet immediately — restart
skips existing chunk files, so a crash at chunk 150/489 loses only that
chunk's API calls. At the end, chunks are concatenated into the output.
"""

import argparse
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.prompt_cache import load_explanation_cache, lookup
from nla.datagen.providers import CompletionProvider
from nla.datagen.sidecar import NLAApiSummaryMeta, read_sidecar, write_sidecar

# Instruction prompt adapted from the prompt template in the appendix — the proven
# NLA prompt. Shortened from 4-5 → 2-3 features and ~100 word budget so
# responses reliably fit in 300 tokens WITH closing tag (truncated responses
# fail the extract pattern and get dropped — better to constrain the prompt
# than accept half-finished output).
_DEFAULT_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

# Strict: both opening and closing tags MUST be present. Truncated responses
# (max_tokens cut off before </analysis>) fail this and get dropped — we'd
# rather lose the row than train on half a thought.
_DEFAULT_RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"

# Minimum features required — the prompt asks for 2-3, so fewer than 2 means
# the model ignored format. These rows are dropped.
_MIN_FEATURES = 2

# Prefix stripping — API models use all kinds of list markers. We want plain
# paragraphs separated by \n\n, no formatting.
_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"              # bullet chars (incl. en/em dash)
    r"|\d+[.)]"              # 1. 1)
    r"|\(\d+\)"              # (1)
    r"|[a-zA-Z][.)]"         # a. a) A. A)
    r"|\([a-zA-Z]\)"         # (a) (A)
    r"|[ivxIVX]+[.)]"        # i. ii) IV.
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def _extract_and_clean(raw: str, pattern: str) -> str | None:
    """Extract content inside response tags, strip list formatting, emit \\n\\n-separated paragraphs.

    Returns None if the pattern doesn't match (truncated, no tags) — caller drops the row.
    """
    m = re.search(pattern, raw, flags=re.DOTALL)
    if m is None:
        return None
    content = m.group(1)

    cleaned: list[str] = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)  # **Header:** text → Header: text
        line = line.strip().strip("*_")  # trailing emphasis markers
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned)


def explain_table(
    table: pa.Table,
    *,
    output_path: str,
    storage,
    provider: CompletionProvider,
    text_column: str,
    instruction_template: str,
    response_extract_pattern: str,
    samples_per_row: int,
    chunk_size: int,
    cache: dict,
) -> tuple[int, int]:
    """Row-processing + crash-resumable chunk loop, factored out of `main()` so
    it's reusable against ANY sidecar schema (the sidecar read/write on either
    side is schema-specific — see `nla/hrm/explain.py`, which calls this with
    its own `nla.hrm.sidecar` read/write instead of this module's
    `nla.datagen.sidecar`). Returns (row_count, dropped_count); writes the
    output parquet (with api_explanation[_i] column(s) appended) as a side
    effect — caller writes the sidecar after this returns.
    """
    n = samples_per_row
    expl_fields = [pa.field("api_explanation", pa.string())] if n == 1 else [
        pa.field(f"api_explanation_{i}", pa.string()) for i in range(n)
    ]
    out_schema = table.schema
    for f in expl_fields:
        out_schema = out_schema.append(f)
    storage.ensure_parent(output_path)

    chunks_dir = Path(f"{output_path}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def _clean_or_none(raw: str | None) -> str | None:
        if raw is None:
            return None
        assert isinstance(raw, str) and raw, (
            f"provider returned bad completion: {raw!r}. "
            f"CompletionProvider.complete() must return str or None."
        )
        cleaned = _extract_and_clean(raw, response_extract_pattern)
        if cleaned is not None and cleaned.count("\n\n") + 1 < _MIN_FEATURES:
            return None
        return cleaned

    def _process_chunk_single(chunk: pa.Table) -> tuple[pa.Table, int]:
        texts = chunk.column(text_column).to_pylist()
        cached_expls = [lookup(cache, t) for t in texts]
        miss_idx = [i for i, e in enumerate(cached_expls) if e is None]
        miss_prompts = [instruction_template.format(text=texts[i]) for i in miss_idx]
        raw_completions = provider.complete(miss_prompts) if miss_prompts else []
        assert len(raw_completions) == len(miss_prompts), (
            f"provider returned {len(raw_completions)} completions for {len(miss_prompts)} prompts — "
            f"length mismatch violates the CompletionProvider contract"
        )
        miss_cleaned = {j: _clean_or_none(raw) for j, raw in zip(miss_idx, raw_completions, strict=True)}

        dropped = 0
        keep_mask: list[bool] = []
        explanations: list[str] = []
        for i, hit in enumerate(cached_expls):
            cleaned = hit if hit is not None else miss_cleaned[i]
            if cleaned is None:
                dropped += 1
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            explanations.append(cleaned)
        if not all(keep_mask):
            chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
        return chunk.append_column("api_explanation", pa.array(explanations, type=pa.string())), dropped

    def _process_chunk_multi(chunk: pa.Table) -> tuple[pa.Table, int]:
        # No cache path here (caller asserts incompatible) — every row gets
        # `n` independent temperature-sampled completions in one batched call,
        # laid out [row0_sample0, row0_sample1, ..., row1_sample0, ...].
        texts = chunk.column(text_column).to_pylist()
        prompts = [instruction_template.format(text=t) for t in texts for _ in range(n)]
        raw_completions = provider.complete(prompts) if prompts else []
        assert len(raw_completions) == len(prompts), (
            f"provider returned {len(raw_completions)} completions for {len(prompts)} prompts — "
            f"length mismatch violates the CompletionProvider contract"
        )
        dropped = 0
        keep_mask: list[bool] = []
        per_field: list[list[str]] = [[] for _ in range(n)]
        for i in range(len(texts)):
            cleaned = [_clean_or_none(raw_completions[i * n + k]) for k in range(n)]
            if any(c is None for c in cleaned):
                dropped += 1
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            for k in range(n):
                per_field[k].append(cleaned[k])  # type: ignore[arg-type]
        if not all(keep_mask):
            chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
        for k in range(n):
            chunk = chunk.append_column(f"api_explanation_{k}", pa.array(per_field[k], type=pa.string()))
        return chunk, dropped

    _process_chunk = _process_chunk_single if n == 1 else _process_chunk_multi

    dropped_count = 0
    chunk_paths: list[Path] = []
    chunk_starts = list(range(0, table.num_rows, chunk_size))
    skipped = 0
    for chunk_start in tqdm(chunk_starts, desc="chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            skipped += 1
            continue
        chunk_out, dropped = _process_chunk(table.slice(chunk_start, chunk_size))
        dropped_count += dropped
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.rename(chunk_path)
    if skipped:
        print(f"  resumed: skipped {skipped}/{len(chunk_starts)} already-completed chunks")

    row_count = 0
    with pq.ParquetWriter(storage.open_write(output_path), out_schema) as writer:
        for p in chunk_paths:
            t = pq.read_table(p)
            writer.write_table(t)
            row_count += t.num_rows
    return row_count, dropped_count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="SL subset parquet from stage1")
    p.add_argument("--output", required=True)
    p.add_argument("--provider-cls", default="nla.datagen.providers.AnthropicProvider")
    p.add_argument("--provider-kwargs", default=None, help="JSON dict of extra kwargs for the provider constructor")
    p.add_argument("--instruction-template", default=_DEFAULT_INSTRUCTION,
                   help="prompt template with {text} placeholder")
    p.add_argument("--response-extract-pattern", default=_DEFAULT_RESPONSE_PATTERN,
                   help="regex with one capture group — extracts content from API response. "
                        "Default requires both <analysis> and </analysis> (truncated "
                        "responses are dropped). MUST match the tag your instruction asks for.")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per provider.complete() call")
    p.add_argument("--text-column", default="detokenized_text_truncated",
                   help="input column to fill {text} with — e.g. 'context_marked' for the "
                        "HRM pipeline's ⟦-marked⟧ prompt rows (nla/hrm/stage0_hrm.py)")
    p.add_argument("--samples-per-row", type=int, default=1,
                   help="independent completions per row, written as api_explanation_0.."
                        "api_explanation_{N-1} columns (temperature-sampled, so N>1 gives N "
                        "genuinely different explanations of the same input — used by the HRM "
                        "pipeline to fill the L and H SFT fields independently, see "
                        "nla/hrm/build.py). N=1 is the original single-column behavior. "
                        "Incompatible with --cache-from (a cache entry is one explanation per "
                        "text, not N independent ones) and rows are DROPPED (not padded) if "
                        "fewer than N completions survive extraction.")
    p.add_argument("--cache-from", action="append", default=[],
                   help="path(s) to existing *_explained.parquet to reuse explanations from "
                        "(joins on detokenized_text_truncated — same tokenizer + corpus slice "
                        "→ cache hit even across different base models). Repeat flag for "
                        "multiple sources. Storage backend: --cache-storage-cls.")
    p.add_argument("--cache-storage-cls", default="nla.datagen.storage.LocalStorage",
                   help="storage backend for --cache-from paths (may differ from main "
                        "--storage-cls, e.g. cache on cloud storage, output local)")
    add_storage_args(p)
    args = p.parse_args()

    assert "{text}" in args.instruction_template, "instruction-template must contain {text} placeholder"
    assert args.samples_per_row >= 1, f"--samples-per-row must be >=1, got {args.samples_per_row}"
    assert args.samples_per_row == 1 or not args.cache_from, (
        "--cache-from is incompatible with --samples-per-row>1 — a cache entry is one cached "
        "explanation per text, not N independent samples. Drop --cache-from or use N=1."
    )

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    provider: CompletionProvider = load_class(args.provider_cls)(**parse_kwargs(args.provider_kwargs))
    cache = load_explanation_cache(args.cache_from, args.cache_storage_cls) if args.cache_from else {}

    table = pq.read_table(storage.open_read(args.input))
    row_count, dropped_count = explain_table(
        table,
        output_path=args.output,
        storage=storage,
        provider=provider,
        text_column=args.text_column,
        instruction_template=args.instruction_template,
        response_extract_pattern=args.response_extract_pattern,
        samples_per_row=args.samples_per_row,
        chunk_size=args.chunk_size,
        cache=cache,
    )

    # Record provider config in sidecar. Pull model/max_tokens/temperature via
    # getattr — providers aren't required to have these, but the default does.
    api_meta = NLAApiSummaryMeta(
        model=getattr(provider, "model", args.provider_cls),
        max_tokens=getattr(provider, "max_tokens", -1),
        temperature=getattr(provider, "temperature", -1.0),
        instruction_prompt=args.instruction_template,
    )
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__explained",
        row_count=row_count,
        api_summaries=api_meta,
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage2_api_explain",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    assert row_count > 0, (
        f"ALL {dropped_count} rows dropped — either responses didn't match "
        f"--response-extract-pattern={args.response_extract_pattern!r} (truncated? "
        f"wrong tag?), or had fewer than {_MIN_FEATURES} features after cleanup. "
        f"Try: increase max_tokens, shorten the instruction, or check the tag matches "
        f"what your prompt asks for."
    )
    print(f"wrote {row_count} rows → {args.output}")
    if dropped_count > 0:
        print(f"  DROPPED {dropped_count} rows (response didn't match extract pattern)")


if __name__ == "__main__":
    main()
