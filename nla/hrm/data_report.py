"""Composition report for a stage-0 extraction (CPU only, seconds).

    python -m nla.hrm.data_report --base base.parquet

Answers: which datasets/prompts do the activation rows actually come from, how long
are the prompts, how much of the data is the always-included last-prompt-position
(one identical template token per prompt), which tokens dominate, and how many
near-duplicate prompts there are. Use it to decide how to rebalance the corpus.
"""

import argparse
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower())).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--max-rows", type=int, default=200000)
    args = p.parse_args()

    t = pq.read_table(args.base, columns=["dataset", "doc_id", "position", "prompt_len", "is_last_prompt_pos",
                                          "prompt_ids", "context_marked"]).slice(0, args.max_rows)
    n = t.num_rows
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    print(f"rows: {n}   prompts: {len(set(cols['doc_id']))}")

    by_ds = defaultdict(lambda: {"rows": 0, "prompts": set(), "lens": {}, "last": 0})
    for ds, doc, ln, last in zip(cols["dataset"], cols["doc_id"], cols["prompt_len"], cols["is_last_prompt_pos"], strict=True):
        d = by_ds[ds]
        d["rows"] += 1
        d["prompts"].add(doc)
        d["lens"][doc] = ln
        d["last"] += bool(last)
    print(f"\n{'dataset':16s}{'rows':>8s}{'share':>8s}{'prompts':>9s}{'mean_len':>10s}{'max_len':>9s}{'last_pos%':>11s}")
    for ds, d in sorted(by_ds.items(), key=lambda kv: -kv[1]["rows"]):
        lens = list(d["lens"].values())
        print(f"{ds:16s}{d['rows']:8d}{d['rows'] / n:8.1%}{len(d['prompts']):9d}{sum(lens) / len(lens):10.0f}"
              f"{max(lens):9d}{d['last'] / d['rows']:11.1%}")

    last_share = sum(cols["is_last_prompt_pos"]) / n
    toks = Counter(ids[pos] for ids, pos in zip(cols["prompt_ids"], cols["position"], strict=True))
    top = toks.most_common(8)
    print(f"\nlast-prompt-position rows: {last_share:.1%} of the data (one identical template token per prompt)")
    print(f"distinct marked tokens: {len(toks)};  top-8 tokens cover {sum(c for _, c in top) / n:.1%} of rows:")
    print("  " + ", ".join(f"id {t}: {c / n:.1%}" for t, c in top))

    # chat-template tail shared by every prompt (e.g. <turn|> \n <|turn> model \n): rows sampled there are all the same
    # few tokens regardless of the prompt, so they add little diversity (and many rows for short prompts)
    uniq = {doc: ids for doc, ids in zip(cols["doc_id"], cols["prompt_ids"], strict=True)}
    tails = [ids[::-1] for ids in uniq.values()]
    k = 0
    while all(len(t) > k for t in tails) and len({t[k] for t in tails}) == 1:
        k += 1
    in_tail = sum(pos >= ln - k for pos, ln in zip(cols["position"], cols["prompt_len"], strict=True))
    print(f"common chat-template tail: {k} tokens; rows sampled inside it: {in_tail / n:.1%} (includes the last position)")

    rel = [pos / ln for pos, ln in zip(cols["position"], cols["prompt_len"], strict=True)]
    bins = Counter(min(int(r * 5), 4) for r in rel)
    print("position within prompt (quintiles): " + "  ".join(f"{b + 1}:{bins[b] / n:.0%}" for b in range(5)))

    # near-duplicate prompts: first ~200 chars of context, digits collapsed
    seen: dict[str, set[str]] = defaultdict(set)
    for doc, ctx in zip(cols["doc_id"], cols["context_marked"], strict=True):
        key = _norm(re.sub(r"⟦|⟧", "", ctx))[:200]
        seen[key].add(doc)
    dup_prompts = sum(len(v) for v in seen.values() if len(v) > 1)
    n_prompts = len(set(cols["doc_id"]))
    print(f"near-duplicate prompts (same first 200 chars, digits collapsed): {dup_prompts}/{n_prompts} "
          f"({dup_prompts / n_prompts:.1%}) in {sum(len(v) > 1 for v in seen.values())} groups")
    print("\n(near-duplicates in different split buckets make the in-distribution eval optimistic)")


if __name__ == "__main__":
    main()
