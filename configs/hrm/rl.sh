#!/bin/bash
# RL: the GRPO-style loop, simultaneous av (policy) + ar (online critic).
# This is where L-vs-H differentiation actually happens — see docs/hrm.md.
# --sanity runs the real-vs-shuffled-vectors check before training starts.
set -euo pipefail

: "${RL_PARQUET:?set RL_PARQUET (e.g. \$OUT_DIR/rl.parquet)}"
: "${AV_SFT_CKPT:?set AV_SFT_CKPT — ar_sft.sh's SAVE_DIR}"
: "${AR_SFT_CKPT:?set AR_SFT_CKPT — av_sft.sh's SAVE_DIR}"
: "${SAVE_DIR:?set SAVE_DIR for RL checkpoints}"
VERBALIZER_MODEL="${VERBALIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

${PYTHON:-python} -m nla.hrm.train_rl \
    --rl-parquet "$RL_PARQUET" \
    --eval-parquet "${EVAL_PARQUET:-}" \
    --av-sft-ckpt "$AV_SFT_CKPT" \
    --ar-sft-ckpt "$AR_SFT_CKPT" \
    --verbalizer-model "$VERBALIZER_MODEL" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --group-size "${GROUP_SIZE:-8}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-300}" \
    --steps "${STEPS:-500}" \
    --policy-lr "${POLICY_LR:-1e-5}" \
    --ar-lr "${AR_LR:-1e-4}" \
    --kl-beta "${KL_BETA:-0.01}" \
    --w-sum "${W_SUM:-1.0}" \
    --w-comp "${W_COMP:-0.25}" \
    --eval-every "${EVAL_EVERY:-25}" \
    --save-every "${SAVE_EVERY:-50}" \
    --sanity \
    --output "$SAVE_DIR" \
    "$@"
