"""API explanations for HRM AV-SFT/AR-SFT rows — thin HRM-specific CLI around
`nla.datagen.stage2_api_explain.explain_table` (the reusable row-processing +
crash-resumable chunk loop). Exists as its own file (not a `--stage`-flag
branch inside stage2_api_explain.py) because the sidecar type differs:
`nla.hrm.sidecar.HrmDatasetMeta`, not `nla.datagen.sidecar.NLADatasetMeta`.

Default `--samples-per-row 2`: build.py needs two INDEPENDENT explanations
per row to randomly assign to the L and H SFT fields (there's no
stream-specific teacher — see docs/hrm.md "SFT is format-only").
"""

import argparse

import pyarrow.parquet as pq

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.providers import CompletionProvider
from nla.datagen.stage2_api_explain import _DEFAULT_RESPONSE_PATTERN, explain_table
from nla.hrm.sidecar import read_sidecar, write_sidecar

# Bidirectional-prompt instruction: unlike the original NLA's causal-LM
# "predict what comes next" framing, Mimir attends to the WHOLE prompt when
# producing this state — so the state at ⟦token⟧ can depend on text on
# EITHER side of it. ~100 words, 4-5 features, matching the original's
# warm-up-prompt style but adapted for bidirectionality (see docs/hrm.md).
DEFAULT_HRM_INSTRUCTION = """You are analyzing an internal state of a language model at one token position inside a prompt it is reading. This model reads the ENTIRE prompt bidirectionally before producing anything, so the state at the marked token can reflect information from anywhere in the prompt — both before and after it, not just what precedes it.

Identify the 4-5 most important features this state is likely to encode. Consider: the local syntactic/semantic role of the marked token; what question or task the surrounding text poses; salient entities, values, or constraints stated anywhere in the prompt; the register/genre; and anything distinctive about the marked position's role in the whole passage.

Each feature: a concise ~15-20 word description, with specific examples inline where useful. About 100 words total.

Format — IMPORTANT: keep to ~100 words and ALWAYS close the tag:
<analysis>
[feature 1]

[feature 2]

[feature 3 — the marked token's own role and immediate context]
</analysis>

Text (⟦ ⟧ marks the token whose state you are describing):

<begin_text>{text}<end_text>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="av_sft.parquet or ar_sft.parquet from split.py")
    p.add_argument("--output", required=True)
    p.add_argument("--provider-cls", default="nla.datagen.providers.AnthropicProvider")
    p.add_argument("--provider-kwargs", default=None,
                    help='JSON dict, e.g. \'{"max_tokens": 400}\' — the HRM template runs longer '
                         "than the original's 300-token default")
    p.add_argument("--instruction-template", default=DEFAULT_HRM_INSTRUCTION)
    p.add_argument("--response-extract-pattern", default=_DEFAULT_RESPONSE_PATTERN)
    p.add_argument("--samples-per-row", type=int, default=2,
                    help="2 independent explanations per row (for the L and H SFT fields)")
    p.add_argument("--chunk-size", type=int, default=512)
    add_storage_args(p)
    args = p.parse_args()

    assert "{text}" in args.instruction_template, "instruction-template must contain {text} placeholder"

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    assert in_meta.stage == "base", f"expected stage=base, got stage={in_meta.stage!r}"

    provider_kwargs = parse_kwargs(args.provider_kwargs)
    # The HRM template runs longer than AnthropicProvider's 300-token default; other
    # providers (e.g. UCloudGLMProvider) carry their own, larger defaults — don't override.
    if args.provider_cls.endswith("AnthropicProvider"):
        provider_kwargs.setdefault("max_tokens", 400)
    provider: CompletionProvider = load_class(args.provider_cls)(**provider_kwargs)

    table = pq.read_table(storage.open_read(args.input))
    row_count, dropped_count = explain_table(
        table,
        output_path=args.output,
        storage=storage,
        provider=provider,
        text_column="context_marked",
        instruction_template=args.instruction_template,
        response_extract_pattern=args.response_extract_pattern,
        samples_per_row=args.samples_per_row,
        chunk_size=args.chunk_size,
        cache={},  # no cache path for HRM — samples must be independent, see stage2's assert
    )
    assert row_count > 0, (
        f"ALL {dropped_count} rows dropped — check --response-extract-pattern matches the tag "
        f"the instruction asks for, or raise --provider-kwargs max_tokens if responses were "
        f"truncated before the closing tag."
    )

    from dataclasses import replace

    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__explained",
        row_count=row_count,
        api_summary={
            "model": getattr(provider, "model", args.provider_cls),
            "max_tokens": getattr(provider, "max_tokens", -1),
            "temperature": getattr(provider, "temperature", -1.0),
            "instruction_prompt": args.instruction_template,
            "samples_per_row": args.samples_per_row,
        },
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.hrm.explain",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {row_count} rows → {args.output}")
    if dropped_count > 0:
        print(f"  DROPPED {dropped_count} rows (fewer than {args.samples_per_row} completions survived extraction)")


if __name__ == "__main__":
    main()
