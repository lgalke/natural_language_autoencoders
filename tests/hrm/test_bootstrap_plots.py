import json

import numpy as np

from nla.hrm.bootstrap import dump_results, judge_results, paired_results
from nla.hrm.plots import _blocks, _parse_blocks_file, _parse_log


def _rows(n=60, shift=0.0):
    rng = np.random.default_rng(0)
    return [{"split": "iid", "dataset": "d", "is_last_prompt_pos": i % 10 == 0, "context_marked": f"a ⟦t{i}⟧ b",
             "L_field": 'Marked token: "t1". x', "H_field": 'Marked token: "t1". y',
             "marked_token_quote_correct": bool(i % 2),
             "fve": {"fve_sum": 0.2 + shift + float(rng.normal(0, 0.05)), "fve_L": 0.1, "fve_H": 0.3},
             "grounding": {"quoted_total": 2, "quoted_ok": 1, "caps_total": 0, "caps_ok": 0}} for i in range(n)]


def test_point_estimate_is_the_mean_and_ci_brackets_it():
    rows = _rows()
    res = dump_results(rows, n_boot=500, seed=0)["iid"]["all"]
    mean = np.mean([r["fve"]["fve_sum"] for r in rows])
    assert abs(res["fve_sum"]["est"] - mean) < 1e-9
    assert res["fve_sum"]["lo"] < mean < res["fve_sum"]["hi"]
    assert abs(res["ground_span"]["est"] - 0.5) < 1e-9


def test_paired_difference_recovers_the_shift_with_a_tight_interval():
    a, b = _rows(shift=0.0), _rows(shift=0.07)
    d = paired_results(a, b, n_boot=500, seed=0)["iid"]["all"]["fve_sum"]
    assert abs(d["est"] - 0.07) < 1e-9 and d["hi"] - d["lo"] < 1e-9 + 0.02


def test_judge_ratio_of_means(tmp_path):
    pred, mean = [1.0, 2.0, 3.0], [4.0, 4.0, 4.0]
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"judge": {"iid": {"primary": {"kl_pred_at_patch": pred, "kl_mean_ablation_at_patch": mean}}}}))
    e = judge_results(str(path), n_boot=200, seed=0)["iid"]["primary"]
    assert abs(e["real_removed"]["est"] - 0.5) < 1e-9 and abs(e["beats_mean_share"]["est"] - 1.0) < 1e-9


def test_log_and_block_parsers(tmp_path):
    log = tmp_path / "rl.log"
    log.write_text("step=5 kl=0.01 fve_sum=0.2 malformed=0/128\nnoise\nstep=60 kl=0.02 fve_sum=0.4 malformed=1/128\n")
    blocks = _blocks(_parse_log(str(log)), 50)
    assert [round(v, 3) for _, v in blocks["fve_sum"]] == [0.2, 0.4]
    f = tmp_path / "b.txt"
    f.write_text("steps 1-50: mean fve_sum=-0.405\nsteps 51-100: fve_sum 0.324  token_ok 0.5  malformed 0.5/128\n")
    parsed = _parse_blocks_file(str(f))
    assert parsed["fve_sum"] == [(25.5, -0.405), (75.5, 0.324)] and parsed["token_ok"] == [(75.5, 0.5)]
