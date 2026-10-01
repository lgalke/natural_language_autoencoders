## 2026-10-02
* **Planned**: [data collection v2](/plans/data-collection-v2.md); **Code**: `nla.hrm.data_report` for the composition of an extraction.
* **Measured**: token probe ceiling, shuffled-vector control and NLL gap; **Code**: `probe_check`, `nll_check`, `build --prefix-token`; see [token probe](/observations/token-probe.md), [token-prefix targets](/decisions/token-prefix-targets.md).
* **Run**: RL run 2 (700 steps, log-reward, batch 128 rollouts) finished; see [RL run 2](/observations/rl-run-2-log-reward.md). Held-out evaluation pending.
* **Code**: judge OOM fix (LM head only at needed positions, chunked), `eval` geometry diagnostics, `--shuffle-vectors` control, marked-token quote accuracy; see [pitfalls](/observations/pitfalls.md).

## 2026-09-30
* **Fixed**: `split.py` now writes the `judge_subset` sidecar; generated marker cache untracked; `eval --limit N` added; `preflight.py`, `AGENTS.md` and `opencode.json` added for handover to an opencode session on the cluster.
* **Added**: this bundle (decisions, observations, open questions, methods notes).
* **Verified**: hidden states of prompts in a padded batch equal those run alone (max difference 3.8e-6, CPU float32); see [PrefixLM](/decisions/prefixlm-rendering.md).
* **Code**: `infer.py`, per-step FVE and sample dumps in `train_rl.py`, `eval --dump-samples`, multiple values per `--eval-parquet` flag.
* **Code**: RL micro-batching and response-only logits after a 95 GB OOM; see [pitfalls](/observations/pitfalls.md).
* **Run**: first 200-step RL run completed; see [RL run 1](/observations/rl-run-1.md).
* **Data**: SFT explanations generated with GLM-5.3 (9191 of 9324 av_sft rows kept); see [teacher](/decisions/explanation-teacher.md).
* **Code**: stage0 length-sorted batching and `logits_to_keep=1` after a 105 GiB OOM; corpus builder fixes; GPU/bf16 defaults; OpenAI-compatible and UCloud providers.

## 2026-09-29
* **Created**: `nla/hrm/` on the `hrm` branch with data generation, model, SFT, RL, judge and evaluation; see [overview](/overview.md).
* **Decided**: the initial design choices in [decisions](/index.md).
