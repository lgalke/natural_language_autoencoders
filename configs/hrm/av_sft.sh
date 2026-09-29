#!/bin/bash
# AV-SFT: format-only warm-up of the 'av' LoRA adapter + injection adapters.
# Stops early once --target-format-rate is hit on eval. See docs/hrm.md
# "SFT is format-only" — this does NOT teach L-vs-H content.
set -euo pipefail

: "${AV_SFT_PARQUET:?set AV_SFT_PARQUET (e.g. $OUT_DIR/av_sft.parquet)}"
: "${SAVE_DIR:?set SAVE_DIR for the av checkpoint}"
: "${NORM_STATS_JSON:?set NORM_STATS_JSON — norm_stats.py's output, gives the injection-scale init}"
VERBALIZER_MODEL="${VERBALIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

${PYTHON:-python} -m nla.hrm.train_av_sft \
    --train-parquet "$AV_SFT_PARQUET" \
    --norm-stats-json "$NORM_STATS_JSON" \
    --verbalizer-model "$VERBALIZER_MODEL" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" \
    --batch-size "${BATCH_SIZE:-16}" \
    --epochs "${EPOCHS:-3}" \
    --lr "${LR:-1e-4}" \
    --target-format-rate "${TARGET_FORMAT_RATE:-0.99}" \
    --eval-every "${EVAL_EVERY:-100}" \
    --output "$SAVE_DIR" \
    "$@"
