"""Mimir patch-back judge — the PRIMARY faithfulness metric. Eval-only, never
called from train_rl.py's per-step loop (see docs/hrm.md and the module
docstring of train_rl.py): run periodically as a standalone check + in the
final evaluation.

Re-renders each eval row's EXACT original Mimir prompt (from `prompt_ids`,
not a re-tokenization of `context_marked` — see stage0_hrm.py's schema
docstring), re-extracts z_L/z_H at the fixed hook site, and asserts they
match the stored gold vectors (catches renderer/hook drift between
extraction time and judge time before trusting anything downstream).

Two patches:
  PRIMARY:   replace H_in@2[b,pos] (= z_L+z_H) with ẑ_L+ẑ_H, measure
             KL(clean || patched) at the patched position (also reported at
             the last-prompt-token position and averaged over the prompt).
  SECONDARY: replace H_out@1[b,pos] (= z_H) with ẑ_H alone and let cycle 2
             run normally — z_H persists through cycle 2 (re-added at every
             L step), so this is the causal test of the H component alone;
             the primary patch (on the SUM) is blind to the L/H split.

References for both: the TRUE vector gives KL≈0 (asserted <1e-4 — a
correctness check on the patch mechanism itself, not a result). A
mean-ablation baseline (replace with the train-set mean s or mean z_H) gives
"fraction of KL recovered" = 1 - KL(pred)/KL(mean).
"""

import argparse
import json

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nla.hrm.devices import default_device, default_dtype
from nla.hrm.mimir import HrmStreamCapture, load_mimir


def _pad_batch(prompt_ids_list: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in prompt_ids_list)
    input_ids = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(prompt_ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(prompt_ids_list):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
    return input_ids.to(device), attn.to(device)


class HrmH2Patcher:
    """Like `HrmStreamCapture` but can OVERWRITE H's 2nd-cycle input (the
    primary patch site) or H's 1st-cycle output (the secondary patch site,
    z_H) at one (batch_row, position) pair per row, before the rest of the
    forward pass continues. `set_patch_h2/set_patch_h1` set the values;
    `clear` removes them (so the SAME model instance can run clean and
    patched passes back-to-back without re-registering hooks)."""

    def __init__(self, model):
        self.inner = model.model
        self.L_cycles = self.inner.config.L_cycles
        self._h_counter = 0
        self._patch_h2: tuple[torch.Tensor, torch.Tensor] | None = None  # (positions[B], values[B,d])
        self._patch_h1: tuple[torch.Tensor, torch.Tensor] | None = None
        self._handles = [
            self.inner.register_forward_pre_hook(self._on_pass_start),
            self.inner.H_module.register_forward_pre_hook(self._on_H_pre),
            self.inner.H_module.register_forward_hook(self._on_H_post),
        ]

    def set_patch_h2(self, positions: torch.Tensor, values: torch.Tensor) -> None:
        self._patch_h2 = (positions, values)

    def set_patch_h1(self, positions: torch.Tensor, values: torch.Tensor) -> None:
        self._patch_h1 = (positions, values)

    def clear(self) -> None:
        self._patch_h2 = None
        self._patch_h1 = None

    def _on_pass_start(self, _module, _args):
        self._h_counter = 0

    def _on_H_pre(self, _module, args):
        h = self._h_counter + 1
        if h == 2 and self._patch_h2 is not None:
            positions, values = self._patch_h2
            hidden = args[0].clone()
            b_idx = torch.arange(hidden.shape[0])
            hidden[b_idx, positions] = values.to(hidden.dtype)
            return (hidden,) + tuple(args[1:])
        return None

    def _on_H_post(self, _module, _inputs, output):
        h = self._h_counter + 1
        result = None
        if h == 1 and self._patch_h1 is not None:
            positions, values = self._patch_h1
            hidden = output.clone()
            b_idx = torch.arange(hidden.shape[0])
            hidden[b_idx, positions] = values.to(hidden.dtype)
            result = hidden
        self._h_counter += 1
        return result

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self) -> "HrmH2Patcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _kl(la: torch.Tensor, lb: torch.Tensor) -> torch.Tensor:
    """KL(softmax(la) || softmax(lb)) per row; la, lb: [n, vocab] float32."""
    return (F.softmax(la, dim=-1) * (F.log_softmax(la, dim=-1) - F.log_softmax(lb, dim=-1))).sum(-1)


@torch.no_grad()
def _judge_batch(
    model, tokenizer, rows: list[dict], z_L_hat: torch.Tensor, z_H_hat: torch.Tensor,
    device: str, mean_s: torch.Tensor | None, mean_zH: torch.Tensor | None,
) -> dict:
    """One padded batch. Memory note: we run the inner HrmTextModel and apply the LM head ONLY at the
    two positions per row we need (patch position, last prompt token). Full [rows, seq, vocab=262144]
    logits (what model(...) builds) are tens of GB for long prompts and used to OOM."""
    n = len(rows)
    pad_id = tokenizer.pad_token_id
    input_ids, attn = _pad_batch([r["prompt_ids"] for r in rows], pad_id, device)
    token_type_ids = attn.clone()
    positions = torch.tensor([r["position"] for r in rows], dtype=torch.long, device=device)
    last_positions = torch.tensor([r["prompt_len"] - 1 for r in rows], dtype=torch.long, device=device)
    z_L_gold = torch.tensor([r["z_L"] for r in rows], dtype=torch.float32, device=device)
    z_H_gold = torch.tensor([r["z_H"] for r in rows], dtype=torch.float32, device=device)
    s_gold = z_L_gold + z_H_gold
    s_hat = z_L_hat.to(device) + z_H_hat.to(device)
    b_all = torch.arange(n, device=device)

    def _forward(want_capture: bool = False):
        # Any active HrmH2Patcher patch is applied via ITS OWN already-registered hooks; the capture is
        # only for the call-count sanity check (6 L / 2 H fires) and to read back recomputed vectors.
        cap = HrmStreamCapture(model)
        h = model.model(input_ids=input_ids, attention_mask=attn, token_type_ids=token_type_ids,
                        use_cache=False).last_hidden_state
        cap.verify_call_counts()
        cap.close()
        out = (model.lm_head(h[b_all, positions]).float(), model.lm_head(h[b_all, last_positions]).float())
        return (out, cap) if want_capture else out

    (lp_clean, ll_clean), clean_cap = _forward(want_capture=True)
    s_recomputed = clean_cap.h_in_target[b_all, positions]   # H_in@2 as computed in THIS pass
    zH_recomputed = clean_cap.z_H[b_all, positions]
    bf16 = next(model.parameters()).dtype != torch.float32
    tol_soft = 5e-2 if bf16 else 1e-4   # stored-vs-recomputed drift: bf16 rounding differs between batch compositions

    results: dict = {}
    with HrmH2Patcher(model) as patcher:
        # (a) mechanism check: patching back the vector RECOMPUTED in this very pass must
        # reproduce the clean logits exactly — this is what tests the patch hook itself.
        patcher.set_patch_h2(positions, s_recomputed)
        kl_mech = _kl(lp_clean, _forward()[0])
        assert kl_mech.max().item() < 1e-4, (
            f"patching back the recomputed s did not reproduce clean logits (max KL={kl_mech.max().item():.2e}) — "
            f"patch mechanism or hook-site assumption is wrong, distrust everything below."
        )
        # (b) noise floor: the STORED gold s came from the extraction run (different batch
        # composition/padding), so in bf16 it differs from the recomputed one by rounding noise.
        # Reconstruction KLs must be read against this floor.
        patcher.set_patch_h2(positions, s_gold)
        kl_true = _kl(lp_clean, _forward()[0])
        if kl_true.max().item() > tol_soft:
            print(f"  [judge] WARNING: stored gold s drifts from recomputed (max KL={kl_true.max().item():.2e} > {tol_soft:.0e}); "
                  f"stored vectors may come from a different dtype/renderer than this judge run.")
        assert kl_true.max().item() < 0.25, (
            f"stored gold s is far from the recomputed one (max KL={kl_true.max().item():.2e}): wrong prompt_ids/"
            f"position columns or an extraction/judge dtype mismatch."
        )

        patcher.set_patch_h2(positions, s_hat)
        lp_pred, ll_pred = _forward()
        kl_pred_at_patch = _kl(lp_clean, lp_pred)
        kl_pred_at_last = _kl(ll_clean, ll_pred)
        results["primary"] = {
            "kl_mechanism_max": kl_mech.max().item(),  # recomputed-vector patch, expect ~0
            "kl_true_at_patch_max": kl_true.max().item(),  # stored-gold noise floor (bf16 rounding)
            "kl_true_at_patch": kl_true.tolist(),
            "kl_pred_at_patch": kl_pred_at_patch.tolist(),
            "kl_pred_at_last": kl_pred_at_last.tolist(),
        }
        if mean_s is not None:
            patcher.set_patch_h2(positions, mean_s.to(device).unsqueeze(0).expand(n, -1))
            kl_mean = _kl(lp_clean, _forward()[0])
            results["primary"]["kl_mean_ablation_at_patch"] = kl_mean.tolist()
            results["primary"]["frac_kl_recovered"] = [
                1.0 - (p / m) if m > 1e-8 else float("nan")
                for p, m in zip(kl_pred_at_patch.tolist(), kl_mean.tolist(), strict=True)
            ]

        # --- secondary: patch z_H alone at H_out@1, let cycle 2 run normally ---
        patcher.clear()
        patcher.set_patch_h1(positions, zH_recomputed)
        kl_mech_h = _kl(lp_clean, _forward()[0])
        assert kl_mech_h.max().item() < 1e-4, (
            f"patching back the recomputed z_H did not reproduce clean logits (max KL={kl_mech_h.max().item():.2e})"
        )
        patcher.set_patch_h1(positions, z_H_gold)
        kl_true_h = _kl(lp_clean, _forward()[0])
        patcher.set_patch_h1(positions, z_H_hat.to(device))
        kl_pred_h = _kl(lp_clean, _forward()[0])
        results["secondary_zH"] = {
            "kl_mechanism_max": kl_mech_h.max().item(),
            "kl_true_at_patch_max": kl_true_h.max().item(),
            "kl_pred_at_patch": kl_pred_h.tolist(),
        }
        if mean_zH is not None:
            patcher.set_patch_h1(positions, mean_zH.to(device).unsqueeze(0).expand(n, -1))
            kl_mean_h = _kl(lp_clean, _forward()[0])
            results["secondary_zH"]["kl_mean_ablation_at_patch"] = kl_mean_h.tolist()
            results["secondary_zH"]["frac_kl_recovered"] = [
                1.0 - (p / m) if m > 1e-8 else float("nan")
                for p, m in zip(kl_pred_h.tolist(), kl_mean_h.tolist(), strict=True)
            ]
    return results


def run_judge(
    model, tokenizer, rows: list[dict], z_L_hat: torch.Tensor, z_H_hat: torch.Tensor,
    device: str, mean_s: torch.Tensor | None = None, mean_zH: torch.Tensor | None = None,
    max_batch_tokens: int = 16384, max_rows: int = 16,
) -> dict:
    """rows: parquet rows with prompt_ids/position/prompt_len/z_L/z_H (GOLD).
    z_L_hat/z_H_hat: reconstructed streams, [N, d_model], SAME row order.
    Rows are processed in length-sorted chunks under a token budget (attention memory is quadratic in
    length); per-row lists come back in the ORIGINAL order, `*_max` entries are maxima over chunks."""
    N = len(rows)
    order = sorted(range(N), key=lambda i: len(rows[i]["prompt_ids"]))
    chunks: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        length = len(rows[i]["prompt_ids"])
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * length > max_batch_tokens):
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)

    merged: dict = {}
    for idx in chunks:
        part = _judge_batch(model, tokenizer, [rows[i] for i in idx], z_L_hat[idx], z_H_hat[idx],
                            device, mean_s, mean_zH)
        for sec, vals in part.items():
            m = merged.setdefault(sec, {})
            for k, v in vals.items():
                if isinstance(v, list):
                    full = m.setdefault(k, [None] * N)
                    for j, i in enumerate(idx):
                        full[i] = v[j]
                else:
                    m[k] = max(m.get(k, v), v)
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-parquet", required=True, help="judge_subset.parquet (or eval_iid/eval_ood) from split.py")
    p.add_argument("--reconstructions", required=True,
                    help="JSON {row_index: {z_L_hat: [...], z_H_hat: [...]}} — produced by eval.py's rollout step")
    p.add_argument("--base-model", default=None, help="defaults to nla.hrm.mimir.DEFAULT_MIMIR")
    p.add_argument("--device", default=default_device())
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default=default_dtype())
    p.add_argument("--mean-s-json", default=None, help="{'mean_s': [...], 'mean_zH': [...]} — train-set means")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from nla.hrm.mimir import DEFAULT_MIMIR
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, tokenizer = load_mimir(args.base_model or DEFAULT_MIMIR, device=args.device, torch_dtype=dtype)

    t = pq.read_table(args.eval_parquet)
    rows = t.to_pylist()
    recon = json.load(open(args.reconstructions))
    idx = [int(k) for k in recon]
    rows = [rows[i] for i in idx]
    z_L_hat = torch.tensor([recon[str(i)]["z_L_hat"] for i in idx], dtype=torch.float32)
    z_H_hat = torch.tensor([recon[str(i)]["z_H_hat"] for i in idx], dtype=torch.float32)

    mean_s = mean_zH = None
    if args.mean_s_json:
        m = json.load(open(args.mean_s_json))
        mean_s = torch.tensor(m["mean_s"], dtype=torch.float32)
        mean_zH = torch.tensor(m["mean_zH"], dtype=torch.float32)

    results = run_judge(model, tokenizer, rows, z_L_hat, z_H_hat, args.device, mean_s, mean_zH)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    import statistics
    print(f"n={len(rows)}")
    print(f"primary KL(clean||patched) at patch pos: mean={statistics.mean(results['primary']['kl_pred_at_patch']):.4f}")
    print(f"primary KL at last-prompt-token: mean={statistics.mean(results['primary']['kl_pred_at_last']):.4f}")
    if "frac_kl_recovered" in results["primary"]:
        vals = [v for v in results["primary"]["frac_kl_recovered"] if v == v]  # drop NaN
        if vals:
            print(f"primary fraction KL recovered vs mean-ablation: mean={statistics.mean(vals):.3f}")
    print(f"secondary (z_H alone) KL at patch pos: mean={statistics.mean(results['secondary_zH']['kl_pred_at_patch']):.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
