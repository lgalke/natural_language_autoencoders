"""Plots for the talk, from `nla.hrm.bootstrap` results and RL logs. Light "slide" theme, matplotlib only.

    python -m nla.hrm.bootstrap --dump E7=... E8=... --out results.json
    python -m nla.hrm.plots bars    --results results.json --metrics fve_sum fve_L fve_H --split iid --subset non-last \\
                                    --models E7 E8 --reference --out figs/fve_iid.png
    python -m nla.hrm.plots bars    --results results.json --metrics tok_L tok_H --split iid --subset non-last \\
                                    --models E7 SplitSFT --out figs/token_fields.png
    python -m nla.hrm.plots dissociation --results results.json --model E11 --split iid --subset non-last --out figs/dissoc.png
    python -m nla.hrm.plots judge   --results results.json --model E7 --out figs/judge.png
    python -m nla.hrm.plots curves  --log E8=rl_ver.log E9=rl_split.log --keys fve_sum kl --out figs/curves.png
    python -m nla.hrm.plots curves  --blocks E7=e7_blocks.txt --keys fve_sum --out figs/e7_blocks.png
    python -m nla.hrm.plots progress --out figs/token_progress.png

Design (dataviz skill): categorical slots 1 to 4 in fixed order (blue, orange, aqua, yellow; validated, adjacent-pair CVD
separation 9.1); a legend whenever there are two or more series; hairline grid; thin marks; no dual axes. Aqua and yellow are
below 3:1 contrast on the light surface, so every figure's numbers are also available as a table (`nla.hrm.bootstrap` prints
it); the error bars are 95% intervals over ROWS only (they ignore training-run variance). Colour follows the model name in
the order given by --models / --log, so keep the same order across figures.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8b8a85"

LABELS = {"fve_sum": "sum", "fve_L": "L", "fve_H": "H", "tok_L": "L field / call", "tok_H": "H field / call",
          "tok_first": "first quoted field", "ground_span": "quoted spans grounded", "ground_name": "names grounded"}

# Cross-stream ridge baselines (FVE of one stream, or of the sum, from the TRUE other stream; docs/okf/experiments.md,
# `nla.hrm.baselines --limit 100`): (split, subset) -> {metric: [values]}; for the sum both directions are given.
REFERENCE = {
    ("iid", "all"): {"fve_L": [0.144], "fve_H": [0.283], "fve_sum": [0.205, 0.383]},
    ("iid", "non-last"): {"fve_L": [0.088], "fve_H": [0.186], "fve_sum": [0.139, 0.317]},
    ("ood", "all"): {"fve_L": [-0.009], "fve_H": [0.271], "fve_sum": [0.091, 0.281]},
    ("ood", "non-last"): {"fve_L": [-0.096], "fve_H": [0.180], "fve_sum": [0.006, 0.203]},
}
CHANCE_TOKEN = 0.05  # frequent non-last token guess (E5 note)


def _style(ax, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)


def _legend(ax, offset: float = -0.12) -> None:
    """Horizontal legend below the axes (never over the marks or error bars)."""
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=max(len(handles), 1),
              labelcolor=INK2)


def _save(fig, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


def cmd_bars(a) -> None:
    res = json.load(open(a.results))["models"]
    models = a.models or list(res)
    metrics = a.metrics
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    n = len(models)
    width = min(0.22, 0.04 * len(metrics), 0.9 / n - 0.02)  # about 30 px thick at slide size
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        cell = res[m][a.split][a.subset]
        est = np.array([cell[k]["est"] for k in metrics])
        lo = np.array([cell[k]["lo"] for k in metrics])
        hi = np.array([cell[k]["hi"] for k in metrics])
        pos = x + (i - (n - 1) / 2) * (width + 0.02)
        ax.bar(pos, est, width=width, color=SERIES[i], label=m, zorder=3)
        ax.errorbar(pos, est, yerr=[est - lo, hi - est], fmt="none", ecolor=INK, elinewidth=1.2, capsize=0, zorder=4)
    is_fve = metrics[0].startswith("fve")
    if a.reference and is_fve:
        ref = REFERENCE[(a.split, a.subset)]
        for j, k in enumerate(metrics):
            vals = ref.get(k)
            if vals:
                ax.hlines(vals, j - 0.42, j + 0.42, colors=NEUTRAL, linewidth=1.6, zorder=2,
                          label="cross-stream ridge baseline" if j == 0 else None)
    if not is_fve and metrics[0].startswith("tok"):
        ax.axhline(CHANCE_TOKEN, color=NEUTRAL, linewidth=1.2, zorder=2)
        ax.text(0.5 if len(metrics) > 1 else 0.0, CHANCE_TOKEN + 0.012, "chance (about 5%)", color=INK2, fontsize=9, ha="center")
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=2)
    ax.set_xticks(x, [LABELS.get(k, k) for k in metrics])
    title = a.title or f"{'FVE' if is_fve else 'Marked-token accuracy'}: {a.split}, {a.subset} rows (95% CI over rows)"
    _style(ax, "FVE (vs. mean predictor)" if is_fve else "share correct", title)
    _legend(ax)
    _save(fig, a.out)


def cmd_dissociation(a) -> None:
    """2 x 2: per-call accuracy on two facts (marked token, position fifth): L call vs H call, with chance levels."""
    cell = json.load(open(a.results))["models"][a.model][a.split][a.subset]
    groups = [("marked token", "tok", CHANCE_TOKEN), ("position (fifth of the prompt)", "pos", 0.20)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    width = 0.2
    for gi, (_, key, chance) in enumerate(groups):
        for ci, (call, color) in enumerate((("L", SERIES[0]), ("H", SERIES[1]))):
            c = cell[f"{key}_{call}"]
            x = gi + (ci - 0.5) * (width + 0.03)
            ax.bar(x, c["est"], width=width, color=color, label=f"{call} call" if gi == 0 else None, zorder=3)
            ax.errorbar(x, c["est"], yerr=[[c["est"] - c["lo"]], [c["hi"] - c["est"]]], fmt="none", ecolor=INK,
                        elinewidth=1.2, capsize=0, zorder=4)
        ax.hlines(chance, gi - 0.3, gi + 0.3, colors=NEUTRAL, linewidth=1.6, zorder=2,
                  label="chance" if gi == 0 else None)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    _style(ax, "share correct", a.title or f"Per-call accuracy, {a.split}, {a.subset} rows ({a.model}; 95% CI over rows)")
    _legend(ax)
    _save(fig, a.out)


def cmd_judge(a) -> None:
    res = json.load(open(a.results))["judge"][a.model]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
    for ax, site, name in zip(axes, ("primary", "secondary_zH"), ("patch the sum (H input)", "patch z_H only"), strict=True):
        splits = [s for s in ("iid", "ood") if s in res]
        x = np.arange(len(splits))
        series = [("reconstruction", "real_kl", SERIES[0]), ("another row's reconstruction", "shuffled_kl", SERIES[1]),
                  ("mean vector (ablation)", "mean_ablation_kl", NEUTRAL)]
        present = [s for s in series if all(s[1] in res[sp].get(site, {}) for sp in splits)]
        w = 0.22
        for i, (label, key, color) in enumerate(present):
            est = np.array([res[sp][site][key]["est"] for sp in splits])
            lo = np.array([res[sp][site][key]["lo"] for sp in splits])
            hi = np.array([res[sp][site][key]["hi"] for sp in splits])
            pos = x + (i - (len(present) - 1) / 2) * (w + 0.02)
            ax.bar(pos, est, width=w, color=color, label=label, zorder=3)
            ax.errorbar(pos, est, yerr=[est - lo, hi - est], fmt="none", ecolor=INK, elinewidth=1.2, capsize=0, zorder=4)
        ax.set_xticks(x, splits)
        _style(ax, "KL(clean || patched), nats" if site == "primary" else "", name)
        _legend(ax)
    _save(fig, a.out)


_KV = re.compile(r"(\w+)=(-?\d+\.?\d*(?:e-?\d+)?)")


def _parse_log(path: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in open(path, errors="ignore"):
        m = re.search(r"step=(\d+) ", line)
        if m:
            rows[int(m.group(1))] = {k: float(v) for k, v in _KV.findall(line) if k != "step"}
    return rows


def _blocks(rows: dict[int, dict[str, float]], size: int = 50) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for step, kv in rows.items():
        for k, v in kv.items():
            out.setdefault(k, {}).setdefault((step - 1) // size, []).append(v)
    return {k: [((b + 0.5) * size, float(np.mean(v))) for b, v in sorted(d.items())] for k, d in out.items()}


def _parse_blocks_file(path: str) -> dict[str, list[tuple[int, float]]]:
    """Pasted lines like `steps 51-100: mean fve_sum=0.318` or `steps 1-50: fve_sum 0.324  token_ok 0.515 ...`."""
    out: dict[str, list[tuple[int, float]]] = {}
    for line in open(path):
        m = re.match(r"\s*steps (\d+)-(\d+):(.*)", line)
        if not m:
            continue
        mid = (int(m[1]) + int(m[2])) / 2
        for k, v in re.findall(r"(\w+)[ =]+(-?\d+\.?\d*)", m[3].replace("mean", " ")):
            out.setdefault(k, []).append((mid, float(v)))
    return out


def cmd_curves(a) -> None:
    runs: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for s in a.log or []:
        name, path = s.split("=", 1)
        runs[name] = _blocks(_parse_log(path), a.block)
    for s in a.blocks or []:
        name, path = s.split("=", 1)
        runs[name] = _parse_blocks_file(path)
    keys = a.keys
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        for i, (name, data) in enumerate(runs.items()):
            if key in data:
                xs, ys = zip(*data[key], strict=True)
                ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=name, solid_capstyle="round")
        _style(ax, key, f"{key} per {a.block}-step block (training rollouts)")
        ax.set_xlabel("RL step", color=INK2, fontsize=11)
        if len(runs) > 1:
            _legend(ax)
    _save(fig, a.out)


# Held-out marked-token accuracy on non-last positions, n=87 (iid) / 88 (ood), from docs/okf/experiments.md.
PROGRESS = [("E3\nlog-reward RL", 0.0, 0.0), ("E4\ntoken-prefix SFT", 0.22, 0.09), ("E5\n+ RL", 0.19, 0.08),
            ("E6\ntoken loss weight 8", 0.49, 0.30), ("E7\n+ 1300 RL steps", 0.51, 0.33)]


def cmd_progress(a) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(PROGRESS))
    for i, (label, col) in enumerate((("iid", 1), ("ood", 2))):
        ys = [p[col] for p in PROGRESS]
        ax.plot(x, ys, color=SERIES[i], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    ax.axhline(CHANCE_TOKEN, color=NEUTRAL, linewidth=1.2)
    ax.text(len(PROGRESS) - 0.6, CHANCE_TOKEN + 0.015, "chance (about 5%)", color=INK2, fontsize=9, ha="right")
    ax.set_xticks(x, [p[0] for p in PROGRESS])
    _style(ax, "marked token correct", a.title or "Reading the marked token, by design iteration (non-last positions)")
    ax.set_ylim(0, 0.6)
    _legend(ax, offset=-0.22)
    _save(fig, a.out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bars")
    b.add_argument("--results", required=True)
    b.add_argument("--metrics", nargs="+", required=True)
    b.add_argument("--split", default="iid")
    b.add_argument("--subset", default="non-last", choices=["all", "non-last"])
    b.add_argument("--models", nargs="+", default=None)
    b.add_argument("--reference", action="store_true", help="FVE only: draw the cross-stream ridge baselines")
    b.add_argument("--title", default=None)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_bars)
    d = sub.add_parser("dissociation")
    d.add_argument("--results", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--split", default="iid")
    d.add_argument("--subset", default="non-last", choices=["all", "non-last"])
    d.add_argument("--title", default=None)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_dissociation)
    j = sub.add_parser("judge")
    j.add_argument("--results", required=True)
    j.add_argument("--model", required=True)
    j.add_argument("--out", required=True)
    j.set_defaults(fn=cmd_judge)
    c = sub.add_parser("curves")
    c.add_argument("--log", nargs="*", help="NAME=train_rl log (lines with step=N ... key=value)")
    c.add_argument("--blocks", nargs="*", help="NAME=file with pasted 'steps a-b: ...' block-mean lines")
    c.add_argument("--keys", nargs="+", default=["fve_sum"])
    c.add_argument("--block", type=int, default=50)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_curves)
    g = sub.add_parser("progress")
    g.add_argument("--title", default=None)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_progress)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
