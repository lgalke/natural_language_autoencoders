"""Print readable examples from an `eval --dump-samples` file (for slides / reading the outputs).

    python -m nla.hrm.show_examples samples_tokw_final.jsonl --split iid --non-last --n 5 --seed 1
    python -m nla.hrm.show_examples samples_tokw_final.jsonl --correct          # only rows whose quoted token is right
    python -m nla.hrm.show_examples samples_tokw_final.jsonl --wrong --md > examples.md

Each example shows the prompt around the marked token (⟦ ⟧), the true token, the L and H fields, per-row FVE and the
grounding counts. Selection is a seeded random sample, so the same flags give the same examples. Pick examples by
the seed/filters BEFORE looking at the text if you want them to be representative (do not cherry-pick for the talk).
"""

import argparse
import json
import random
import re
import textwrap


def _window(context: str | None, width: int) -> str:
    c = context or ""
    m = re.search(r"⟦.*?⟧", c, re.S)
    if not m:
        return c[: 2 * width]
    a, b = max(0, m.start() - width), min(len(c), m.end() + width)
    return ("…" if a else "") + c[a:b].replace("\n", " ⏎ ") + ("…" if b < len(c) else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", help="jsonl written by eval --dump-samples")
    p.add_argument("--split", default=None, help="only this split (iid / ood)")
    p.add_argument("--non-last", action="store_true", help="skip last-prompt-position rows (always the same newline token)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--correct", action="store_true", help="only rows whose quoted marked token is the real one")
    g.add_argument("--wrong", action="store_true", help="only rows whose quoted marked token is wrong")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=250, help="characters of prompt context on each side of the marked token")
    p.add_argument("--width", type=int, default=110, help="wrap width for the L/H text")
    p.add_argument("--md", action="store_true", help="markdown output")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.samples, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("L_field") is not None]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.non_last:
        rows = [r for r in rows if not r.get("is_last_prompt_pos")]
    if args.correct:
        rows = [r for r in rows if r.get("marked_token_quote_correct") is True]
    if args.wrong:
        rows = [r for r in rows if r.get("marked_token_quote_correct") is False]
    random.Random(args.seed).shuffle(rows)
    picked = rows[: args.n]
    print(f"{len(rows)} matching rows, showing {len(picked)} (seed {args.seed})\n")

    for i, r in enumerate(picked, 1):
        m = re.search(r"⟦(.*?)⟧", r.get("context_marked") or "", re.S)
        true_tok = m.group(1) if m else "?"
        fve = (r.get("fve") or {})
        gr = r.get("grounding") or {}
        q = r.get("marked_token_quote_correct")
        head = (f"[{i}] {r.get('split')} / {r.get('dataset')} / pos {r.get('position')}  true token {true_tok!r}  "
                f"quote {'correct' if q else 'WRONG' if q is False else 'none'}  "
                f"FVE sum {fve.get('fve_sum', float('nan')):.2f} L {fve.get('fve_L', float('nan')):.2f} "
                f"H {fve.get('fve_H', float('nan')):.2f}  "
                f"grounded spans {gr.get('quoted_ok', 0)}/{gr.get('quoted_total', 0)} names {gr.get('caps_ok', 0)}/{gr.get('caps_total', 0)}")
        print(("### " if args.md else "") + head)
        print(("> " if args.md else "  PROMPT: ") + _window(r.get("context_marked"), args.window))
        for name in ("L_field", "H_field"):
            label = name[0]
            body = textwrap.fill(r[name].strip(), args.width, initial_indent="    ", subsequent_indent="    ")
            print(f"  {label}:" if not args.md else f"**{label}:**")
            print(body)
        print()


if __name__ == "__main__":
    main()
