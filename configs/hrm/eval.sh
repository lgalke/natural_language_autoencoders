#!/bin/bash
# Full eval report (+ optional Mimir patch-back judge) for a trained
# checkpoint. Pass the SAME dir for AV_CKPT/AR_CKPT when evaluating an
# train_rl.py checkpoint (it saves both adapters into one directory).
set -euo pipefail

: "${EVAL_PARQUET:?set EVAL_PARQUET, e.g. 'iid=\$OUT_DIR/eval_iid.parquet ood=\$OUT_DIR/eval_ood.parquet'}"
: "${AV_CKPT:?set AV_CKPT}"
: "${AR_CKPT:?set AR_CKPT}"
: "${OUTPUT:?set OUTPUT for the report JSON}"
VERBALIZER_MODEL="${VERBALIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# shellcheck disable=SC2086
${PYTHON:-python} -m nla.hrm.eval \
    $(for spec in $EVAL_PARQUET; do echo --eval-parquet "$spec"; done) \
    --av-ckpt "$AV_CKPT" \
    --ar-ckpt "$AR_CKPT" \
    --verbalizer-model "$VERBALIZER_MODEL" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-300}" \
    --norm-stats-json "${NORM_STATS_JSON:-}" \
    --run-judge \
    --base-model "${BASE_MODEL:-danish-foundation-models/DFM-Mimir-v1.5}" \
    --output "$OUTPUT" \
    "$@"
