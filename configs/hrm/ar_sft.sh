#!/bin/bash
# AR-SFT: format-only warm-up of the 'ar' LoRA adapter + ReconHeads. See
# docs/hrm.md "SFT is format-only" — this does NOT teach L-vs-H content.
set -euo pipefail

: "${AR_SFT_PARQUET:?set AR_SFT_PARQUET (e.g. $OUT_DIR/ar_sft.parquet)}"
: "${SAVE_DIR:?set SAVE_DIR for the ar checkpoint}"
VERBALIZER_MODEL="${VERBALIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

${PYTHON:-python} -m nla.hrm.train_ar_sft \
    --train-parquet "$AR_SFT_PARQUET" \
    --verbalizer-model "$VERBALIZER_MODEL" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --epochs "${EPOCHS:-3}" \
    --lr "${LR:-1e-4}" \
    --output "$SAVE_DIR" \
    "$@"
