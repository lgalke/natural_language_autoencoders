"""Are the two teacher explanations of a row exchangeable? (L/H style confound check)

    python -m nla.hrm.pair_check --parquet splits/ar_sft_explained.parquet --limit 3000

AV-SFT assigns the two samples (api_explanation_0/1) to the L and H fields at random, and AR-SFT pairs sample 0 with the
L head and sample 1 with the H head. If the two samples are independent draws, nothing about a sample's style should
depend on whether it is sample 0 or 1, nor on the other sample's style. Reports:
  openings    most common first-3-word openings of sample 0 and of sample 1 (should look alike)
  same-open   P(both samples of a row open the same way) observed vs. expected for independent samples
              (expected = sum_o p0(o) * p1(o) from the marginals; observed below expected = anti-correlated pairs)
  classifier  cross-validated accuracy for "is this sample 0 or 1" (needs scikit-learn; ~50% if exchangeable)
"""

import argparse
import collections

import pyarrow.parquet as pq


def opening(text: str, n: int = 3) -> str:
    return " ".join((text or "").split()[:n]).lower()


def same_open_stats(e0: list[str], e1: list[str]) -> tuple[float, float]:
    """(observed P(same opening), expected under independence from the two marginals)."""
    o0, o1 = [opening(t) for t in e0], [opening(t) for t in e1]
    n = len(o0)
    observed = sum(a == b for a, b in zip(o0, o1, strict=True)) / n
    c0, c1 = collections.Counter(o0), collections.Counter(o1)
    expected = sum(c0[o] * c1[o] for o in c0) / (n * n)
    return observed, expected


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="explained parquet with api_explanation_0 and api_explanation_1")
    p.add_argument("--limit", type=int, default=3000)
    args = p.parse_args()

    t = pq.read_table(args.parquet, columns=["api_explanation_0", "api_explanation_1"]).slice(0, args.limit)
    e0, e1 = t.column("api_explanation_0").to_pylist(), t.column("api_explanation_1").to_pylist()
    keep = [i for i in range(len(e0)) if e0[i] and e1[i]]
    e0, e1 = [e0[i] for i in keep], [e1[i] for i in keep]
    print(f"{len(e0)} rows")
    for name, e in (("sample 0", e0), ("sample 1", e1)):
        print(f"  {name} openings: {collections.Counter(opening(x) for x in e).most_common(5)}")
    obs, exp = same_open_stats(e0, e1)
    print(f"  same opening: observed {obs:.3f}, expected if independent {exp:.3f}")

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
    except ImportError:
        print("  (scikit-learn not installed: skipping the classifier)")
        return
    texts = [x for pair in zip(e0, e1, strict=True) for x in pair]
    y = np.array([0, 1] * len(e0))
    g = np.repeat(np.arange(len(e0)), 2)
    accs = []
    for tr, te in GroupKFold(5).split(texts, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = v.fit_transform([texts[i] for i in tr])
        accs.append(LogisticRegression(max_iter=2000).fit(X, y[tr]).score(v.transform([texts[i] for i in te]), y[te]))
    print(f"  classifier 'sample 0 or 1': {np.mean(accs):.3f} (5 folds, rows grouped; ~0.5 = exchangeable)")


if __name__ == "__main__":
    main()
