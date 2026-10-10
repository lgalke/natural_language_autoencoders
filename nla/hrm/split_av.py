"""Split AV: describe the two streams in SEPARATE calls of the same verbalizer.

Motivation (docs/okf/experiments.md): the joint verbalizer's FVE equals what the OTHER true stream explains linearly, and
swapping the streams between the slots changes nothing, so its L and H texts carry only the shared part of the streams.
Here each call sees ONE stream (the other slot gets an all-zero vector, which the adapter maps to its constant bias) and
writes ONE field; the two fields of a pair are scored together by the unchanged AR and reward.

Nothing else changes: same prompt template, markers, injection adapters, AR and judge. Only
  * the vectors fed to the AV (one stream zeroed per call),
  * a one-line tag appended to the prompt ("Describe the L stream only ..."),
  * the response format (`<explanation>\nL: ...\n</explanation>`; H likewise).

    python -m nla.hrm.split_av --in av_sft_tok.parquet --out av_sft_split.parquet

converts an AV-SFT parquet (two rows out per row in: the L call and the H call, with the teacher texts already assigned
to L and H) and writes a matching sidecar.
"""

import argparse
import re

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen.storage import LocalStorage
from nla.hrm.build import wrap_lh_explanation
from nla.hrm.facts import has_position_fact, position_bin, position_sentence  # noqa: F401  (position_bin re-exported)
from nla.hrm.recon import parse_fields, parse_single
from nla.hrm.sidecar import read_sidecar, write_sidecar


def stream_tag(stream: str) -> str:
    return f"\n\nDescribe the {stream} stream only. The other stream's slot is empty."


def tag_content(content: str, stream: str) -> str:
    return content + stream_tag(stream)


def single_response(text: str, stream: str) -> str:
    return f"<explanation>\n{stream}: {text}\n</explanation>"


def join_split(l_completion: str, h_completion: str) -> str:
    """Two single-field completions -> one joint completion that `recon.parse_fields` and the AR path understand;
    an empty string (which fails to parse, so gets the failure reward) if either call is malformed."""
    l_text, h_text = parse_single(l_completion, "L"), parse_single(h_completion, "H")
    return wrap_lh_explanation(l_text, h_text) if l_text and h_text else ""


def add_position_fact(text: str, position: int, prompt_len: int) -> str:
    """Insert `Position: K of 5. ` right after the `Marked token: "X". ` prefix (or at the start); a text that already has a
    position fact (built with `build --position-fact`) is returned unchanged."""
    if has_position_fact(text):
        return text
    fact = position_sentence(position, prompt_len) + " "
    m = re.match(r'(Marked token: "[^"]*"\. )', text)
    return (m.group(1) + fact + text[m.end():]) if m else fact + text


def convert_rows(rows: list[dict], facts: bool = False) -> tuple[list[dict], int]:
    """AV-SFT rows (joint response) -> split rows (L call with z_H zeroed, H call with z_L zeroed).
    facts=True also writes the programmatic `Position: K of 5.` fact into BOTH calls (needs position and prompt_len)."""
    out, bad = [], 0
    for r in rows:
        parsed = parse_fields(r["response"])
        if parsed is None:
            bad += 1
            continue
        zeros = [0.0] * len(r["z_L"])
        for stream, text in (("L", parsed[0]), ("H", parsed[1])):
            if facts:
                text = add_position_fact(text, r["position"], r["prompt_len"])
            c = dict(r)
            p0 = dict(r["prompt"][0])
            p0["content"] = tag_content(p0["content"], stream)
            c["prompt"] = [p0]
            c["response"] = single_response(text, stream)
            if stream == "L":
                c["z_H"] = zeros
            else:
                c["z_L"] = zeros
            out.append(c)
    return out, bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True, help="AV-SFT parquet from build.py (e.g. av_sft_tok.parquet)")
    p.add_argument("--out", required=True)
    p.add_argument("--facts", choices=["position"], default=None,
                   help="also write a programmatic fact into both calls: position = `Position: K of 5.` (needs the "
                        "position and prompt_len columns); the loss-weight mask (--prefix-weight) covers it")
    args = p.parse_args()
    t = pq.read_table(args.inp)
    rows, bad = convert_rows(t.to_pylist(), facts=args.facts == "position")
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), args.out)
    meta = read_sidecar(LocalStorage(), args.inp)
    meta.row_count = len(rows)
    meta.build_options = {**(meta.build_options or {}), "split_av": True, **({"facts": args.facts} if args.facts else {})}
    meta.parent_datasets = [*meta.parent_datasets, args.inp]
    write_sidecar(LocalStorage(), args.out, meta)
    print(f"{t.num_rows} rows in, {len(rows)} rows out ({bad} malformed responses skipped) -> {args.out}")


if __name__ == "__main__":
    main()
