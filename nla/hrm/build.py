"""Build final training parquets — av_sft / ar_sft / rl — from split.py's
base-schema buckets (+ explain.py's explanations for av_sft/ar_sft).

  AV-SFT (actor SFT):
    prompt    list[dict]  [{"role":"user","content": actor_template, BOTH
                            [L]/[H] markers present}]
    response  str         "<explanation>\nL: {e_a}\nH: {e_b}\n</explanation>"
                           — e_a/e_b = api_explanation_0/1, order picked by a
                           per-row keyed RNG (no stream-specific teacher; see
                           docs/hrm.md "SFT is format-only").
    z_L, z_H  RAW gold vectors (carried through, unused by the SFT loss but
              handy for eyeballing/debug).

  AR-SFT (critic SFT): ONE OUTPUT ROW PER FIELD — api_explanation_0 always
  pairs with head="L"/target=z_L, api_explanation_1 with head="H"/target=z_H
  (arbitrary but fixed; both explanations are equally generic context
  descriptions at this stage, so which index goes to which head doesn't
  matter — it's still format-only warm-up, differentiation is RL's job):
    prompt    str   critic_template.format(explanation=...), ends with the
                     fixed suffix (`nla.datagen.injection_tokens.compute_critic_suffix_ids`)
    target_z  RAW target vector (z_L or z_H)
    head      "L" | "H"

  RL / eval (eval_iid, eval_ood, judge_subset all go through --stage rl too —
  identical schema, just no `response` column):
    prompt    list[dict], BOTH markers present
    z_L, z_H  RAW gold vectors — reward target

Injection markers are picked against the VERBALIZER's tokenizer (Qwen), not
Mimir's — see `nla/hrm/injection.py`.
"""

import argparse
import hashlib
import random
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

from nla.datagen._common import add_storage_args, make_storage
from nla.datagen.injection_tokens import compute_critic_suffix_ids
from nla.hrm.injection import compute_canonical_neighbors_two, find_two_injection_tokens
from nla.hrm.model import DEFAULT_VERBALIZER
from nla.hrm.sidecar import HrmTokenMeta, read_sidecar, write_sidecar

_INJECT_L_PLACEHOLDER = "<INJECT_L>"
_INJECT_H_PLACEHOLDER = "<INJECT_H>"

# {inj_L}/{inj_H} become the placeholders above in the STORED parquet (never
# the real marker char) — same "store <INJECT>, swap at load time" pattern as
# the rest of NLA (nla/schema.py:INJECT_PLACEHOLDER), so a re-tokenized
# marker never accidentally matches literal corpus text.
DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher investigating the internal recurrent states of a hierarchical reasoning language model (HRM). At each position, the model maintains two interacting memory streams: a fast "L" stream and a slow "H" stream. Describe the semantic content of EACH stream separately.

L stream activation vector:
<concept>{inj_L}</concept>

H stream activation vector:
<concept>{inj_H}</concept>

Respond in exactly this format, 2-3 features per stream:
<explanation>
L: [2-3 features describing the L stream]
H: [2-3 features describing the H stream]
</explanation>"""

DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"

_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
_PROVENANCE_COLS = ["position", "prompt_len", "is_last_prompt_pos", "dataset", "world", "doc_id"]
_PROVENANCE_FIELDS = [
    ("position", pa.int64()), ("prompt_len", pa.int64()), ("is_last_prompt_pos", pa.bool_()),
    ("dataset", pa.string()), ("world", pa.string()), ("doc_id", pa.string()),
]


def token_prefix(context_marked: str | None) -> str:
    """`Marked token: "X". ` built from the ⟦marked⟧ token in the stored context (no LLM call). Whitespace-only
    tokens are shown JSON-escaped (e.g. "\\n"). The wording matches eval.py's quote-accuracy regex."""
    import json
    import re
    m = re.search(r"⟦(.*?)⟧", context_marked or "", re.S)
    if m is None:
        return ""
    tok = m.group(1)
    shown = tok.strip() or json.dumps(tok)[1:-1]
    return f'Marked token: "{shown}". '


def prefix_token_mask(tok, response: str, n_resp_ids: int) -> list[int]:
    """1 for response tokens (tokenizer ids of `response`, no special tokens) inside a `Marked token: "X".` span,
    by character-offset overlap, else 0. Used to weight/measure the token span (train_av_sft, nll_check)."""
    import re
    spans = [m.span() for m in re.finditer(r'(?:Marked token: "[^"]*"|Position: \d of 5)\.', response)]
    offs = tok(response, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    mask = [int(any(a < e and b > s0 for s0, e in spans)) for a, b in offs]
    assert len(mask) == n_resp_ids
    return mask


def wrap_lh_explanation(l_text: str, h_text: str) -> str:
    return f"<explanation>\nL: {l_text}\nH: {h_text}\n</explanation>"


def _schema_for(stage: str, d_model: int, keep_debug: bool) -> pa.Schema:
    if stage == "av_sft":
        core = [("prompt", _PROMPT_STRUCT), ("response", pa.string()),
                ("z_L", pa.list_(pa.float32(), d_model)), ("z_H", pa.list_(pa.float32(), d_model))]
    elif stage == "ar_sft":
        core = [("prompt", pa.string()), ("target_z", pa.list_(pa.float32(), d_model)), ("head", pa.string())]
    elif stage == "rl":
        # prompt_ids: judge.py needs Mimir's EXACT original input_ids (not a
        # re-tokenization of context_marked's decoded text) to reproduce the
        # forward pass that produced z_L/z_H bit-for-bit — see stage0_hrm.py's
        # schema docstring. Only rl/eval rows need it (av_sft/ar_sft never
        # touch Mimir again).
        core = [("prompt", _PROMPT_STRUCT),
                ("z_L", pa.list_(pa.float32(), d_model)), ("z_H", pa.list_(pa.float32(), d_model)),
                ("prompt_ids", pa.list_(pa.int64()))]
    else:
        raise AssertionError(f"unreachable: stage={stage!r}")
    fields = core + _PROVENANCE_FIELDS
    if keep_debug:
        fields += [("context_marked", pa.string())]
    return pa.schema(fields)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="*_explained.parquet (av_sft/ar_sft) or the raw "
                                                     "split parquet (rl / eval_iid / eval_ood / judge_subset)")
    p.add_argument("--stage", required=True, choices=["av_sft", "ar_sft", "rl"],
                    help="use 'rl' for eval_iid/eval_ood/judge_subset too — identical schema")
    p.add_argument("--output", required=True)
    p.add_argument("--verbalizer-model", default=DEFAULT_VERBALIZER)
    p.add_argument("--actor-template", default=DEFAULT_ACTOR_TEMPLATE)
    p.add_argument("--critic-template", default=DEFAULT_CRITIC_TEMPLATE)
    p.add_argument("--prefix-token", action="store_true",
                    help="av_sft/ar_sft only: prepend 'Marked token: \"X\".' (from the stored context) to every L/H field. "
                         "The token is linearly decodable from the vectors (probe_check), so this is a strongly supervised, "
                         "learnable target, and eval's quote accuracy then measures whether the AV reads it. No LLM calls.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-debug-metadata", action=argparse.BooleanOptionalAction, default=True)
    add_storage_args(p)
    args = p.parse_args()

    assert "{inj_L}" in args.actor_template and "{inj_H}" in args.actor_template, (
        "--actor-template must contain both {inj_L} and {inj_H}"
    )
    assert not args.prefix_token or args.stage in ("av_sft", "ar_sft"), "--prefix-token only applies to av_sft/ar_sft"
    if args.stage == "ar_sft":
        assert "{explanation}" in args.critic_template, "--critic-template must contain {explanation}"

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    assert in_meta.stage == "base", f"expected stage=base input, got stage={in_meta.stage!r}"
    if args.stage in ("av_sft", "ar_sft"):
        in_cols = pq.ParquetFile(storage.open_read(args.input)).schema_arrow.names
        assert "api_explanation_0" in in_cols and "api_explanation_1" in in_cols, (
            f"stage={args.stage} requires api_explanation_0/1 columns — run explain.py first "
            f"(--samples-per-row 2). Available: {in_cols}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.verbalizer_model)
    char_l, id_l, char_h, id_h = find_two_injection_tokens(tokenizer, args.actor_template)
    (left_l, right_l), (left_h, right_h) = compute_canonical_neighbors_two(
        tokenizer, args.actor_template, char_l, id_l, char_h, id_h
    )
    suffix_ids = compute_critic_suffix_ids(tokenizer, args.critic_template) if args.stage == "ar_sft" else None

    actor_prompt_content = args.actor_template.format(inj_L=_INJECT_L_PLACEHOLDER, inj_H=_INJECT_H_PLACEHOLDER)

    in_pf = pq.ParquetFile(storage.open_read(args.input))
    d_model = in_meta.extraction.d_model
    out_schema = _schema_for(args.stage, d_model, args.keep_debug_metadata)
    storage.ensure_parent(args.output)
    row_count = 0
    dropped_suffix = 0

    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for batch in tqdm(in_pf.iter_batches(batch_size=2048), desc="rows",
                            total=(in_pf.metadata.num_rows + 2047) // 2048):
            n = len(batch)
            doc_ids = batch.column("doc_id").to_pylist()
            prov = {c: batch.column(c) for c in _PROVENANCE_COLS if c in batch.schema.names}
            debug_col = batch.column("context_marked") if (
                args.keep_debug_metadata and "context_marked" in batch.schema.names
            ) else None

            if args.stage == "av_sft":
                e0 = batch.column("api_explanation_0").to_pylist()
                e1 = batch.column("api_explanation_1").to_pylist()
                if args.prefix_token:
                    assert "context_marked" in batch.schema.names, "--prefix-token needs the context_marked column"
                    pre = [token_prefix(c) for c in batch.column("context_marked").to_pylist()]
                    e0, e1 = [a + b for a, b in zip(pre, e0, strict=True)], [a + b for a, b in zip(pre, e1, strict=True)]
                responses = []
                for i in range(n):
                    rng = random.Random(hashlib.sha256(f"{args.seed}|{doc_ids[i]}|av".encode()).digest())
                    pair = [e0[i], e1[i]]
                    rng.shuffle(pair)
                    responses.append(wrap_lh_explanation(pair[0], pair[1]))
                built = {
                    "prompt": pa.array([[{"role": "user", "content": actor_prompt_content}]] * n, type=_PROMPT_STRUCT),
                    "response": pa.array(responses, type=pa.string()),
                    "z_L": batch.column("z_L"),
                    "z_H": batch.column("z_H"),
                }
                for c in prov:
                    built[c] = prov[c]
                if debug_col is not None:
                    built["context_marked"] = debug_col
                writer.write_table(pa.table(built, schema=out_schema))
                row_count += n

            elif args.stage == "ar_sft":
                e0 = batch.column("api_explanation_0").to_pylist()
                e1 = batch.column("api_explanation_1").to_pylist()
                if args.prefix_token:
                    assert "context_marked" in batch.schema.names, "--prefix-token needs the context_marked column"
                    pre = [token_prefix(c) for c in batch.column("context_marked").to_pylist()]
                    e0, e1 = [a + b for a, b in zip(pre, e0, strict=True)], [a + b for a, b in zip(pre, e1, strict=True)]
                zL_list = batch.column("z_L").to_pylist()
                zH_list = batch.column("z_H").to_pylist()
                for head_name, expls, targets in (("L", e0, zL_list), ("H", e1, zH_list)):
                    prompts, keep_idx = [], []
                    for i in range(n):
                        prompt = args.critic_template.format(explanation=expls[i])
                        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                        assert suffix_ids is not None
                        if len(ids) >= len(suffix_ids) and ids[-len(suffix_ids):] == suffix_ids:
                            prompts.append(prompt)
                            keep_idx.append(i)
                        else:
                            dropped_suffix += 1
                    if not keep_idx:
                        continue
                    keep_mask = pa.array([i in set(keep_idx) for i in range(n)], type=pa.bool_())
                    sub = batch.filter(keep_mask)
                    built = {
                        "prompt": pa.array(prompts, type=pa.string()),
                        "target_z": pa.array([targets[i] for i in keep_idx],
                                              type=pa.list_(pa.float32(), d_model)),
                        "head": pa.array([head_name] * len(keep_idx), type=pa.string()),
                    }
                    for c in _PROVENANCE_COLS:
                        if c in sub.schema.names:
                            built[c] = sub.column(c)
                    if args.keep_debug_metadata and "context_marked" in sub.schema.names:
                        built["context_marked"] = sub.column("context_marked")
                    writer.write_table(pa.table(built, schema=out_schema))
                    row_count += len(keep_idx)

            elif args.stage == "rl":
                assert "prompt_ids" in batch.schema.names, (
                    "input parquet has no prompt_ids column — it must come from stage0_hrm.py "
                    "(older extractions predate this column; re-run stage0_hrm.py)"
                )
                built = {
                    "prompt": pa.array([[{"role": "user", "content": actor_prompt_content}]] * n, type=_PROMPT_STRUCT),
                    "z_L": batch.column("z_L"),
                    "z_H": batch.column("z_H"),
                    "prompt_ids": batch.column("prompt_ids"),
                }
                for c in prov:
                    built[c] = prov[c]
                if debug_col is not None:
                    built["context_marked"] = debug_col
                writer.write_table(pa.table(built, schema=out_schema))
                row_count += n

    tokens_meta = HrmTokenMeta(
        injection_char_L=char_l, injection_token_id_L=id_l,
        injection_left_neighbor_id_L=left_l, injection_right_neighbor_id_L=right_l,
        injection_char_H=char_h, injection_token_id_H=id_h,
        injection_left_neighbor_id_H=left_h, injection_right_neighbor_id_H=right_h,
        critic_suffix_ids=suffix_ids,
    )
    out_meta = replace(
        in_meta,
        dataset_id=f"{args.stage}_{in_meta.dataset_id.removesuffix('__explained')}",
        stage=args.stage,
        row_count=row_count,
        verbalizer_model=args.verbalizer_model,
        tokens=tokens_meta,
        prompt_templates={"actor": args.actor_template, "critic": args.critic_template},
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.hrm.build",
        created_at="",
        git_commit="",
        api_summary=None,
        build_options={"prefix_token": True} if args.prefix_token else None,
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {row_count} rows ({args.stage}) → {args.output}")
    print(f"markers: L={char_l!r}(id={id_l}) H={char_h!r}(id={id_h})")
    if args.stage == "ar_sft" and dropped_suffix:
        print(f"  DROPPED {dropped_suffix} rows (critic suffix mismatch — BPE merge at explanation boundary)")


if __name__ == "__main__":
    main()
