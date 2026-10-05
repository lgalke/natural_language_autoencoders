"""Can a text classifier tell the L field from the H field? And does the difference follow the slot or the vector?

    python -m nla.hrm.lh_text samples_tokw_final.jsonl                                 # cross-validated accuracy
    python -m nla.hrm.lh_text samples_tokw_final.jsonl --apply samples_tokw_final_swap.jsonl

Inputs are `eval --dump-samples` files (L_field / H_field). The shared `Marked token: "X".` prefix is removed first.
With --apply, the classifier is trained on the first file (labels = which slot the text came from) and applied to the
second, which should come from `eval --swap-streams` (z_H written into the [L] slot, z_L into the [H] slot):
  accuracy vs SLOT label near the original accuracy   -> the style follows the slot (position / format artefact)
  accuracy vs SLOT label near 0 (labels flipped)      -> the style follows the vector (stream-specific content)
  near 50%                                            -> neither (the swap destroys the difference)
"""

import argparse
import json
import re

import numpy as np


def strip_prefix(t: str) -> str:
    return re.sub(r'^\s*Marked token:\s*"[^"]*"\.?\s*', "", t)


def load(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("L_field") and r.get("H_field")]
    texts = [strip_prefix(r[f]) for r in rows for f in ("L_field", "H_field")]
    return texts, np.array([0, 1] * len(rows)), np.repeat(np.arange(len(rows)), 2)


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("--apply", default=None, help="second dump (e.g. from eval --swap-streams) to classify")
    args = p.parse_args()

    texts, y, g = load(args.samples)
    print(f"{args.samples}: {len(texts) // 2} rows")
    accs = []
    for tr, te in GroupKFold(5).split(texts, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = v.fit_transform([texts[i] for i in tr])
        accs.append(LogisticRegression(max_iter=2000).fit(X, y[tr]).score(v.transform([texts[i] for i in te]), y[te]))
    print(f"  cross-validated L-vs-H accuracy: {np.mean(accs):.3f}")
    if args.apply:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        clf = LogisticRegression(max_iter=2000).fit(v.fit_transform(texts), y)
        t2, y2, _ = load(args.apply)
        pred = clf.predict(v.transform(t2))
        print(f"{args.apply}: {len(t2) // 2} rows")
        print(f"  accuracy against the SLOT label (L field = 0, H field = 1): {(pred == y2).mean():.3f}")
        print(f"  share of L-slot texts classified as H: {(pred[y2 == 0] == 1).mean():.3f}; "
              f"share of H-slot texts classified as L: {(pred[y2 == 1] == 0).mean():.3f}")


if __name__ == "__main__":
    main()
