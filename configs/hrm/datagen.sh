#!/bin/bash
# HRM data-gen pipeline: corpus -> base -> diagnostics -> split -> norm_stats
# -> explain -> build. See docs/hrm.md for the full design.
#
# Explanations use nla.hrm.testing.FakeCompletionProvider by default (no API
# key needed) so this script is safe to smoke-test end-to-end; set
# PROVIDER_CLS=nla.datagen.providers.AnthropicProvider (+ ANTHROPIC_API_KEY)
# for a real run.
set -euo pipefail

: "${OUT_DIR:?set OUT_DIR for pipeline outputs}"
BASE_MODEL="${BASE_MODEL:-danish-foundation-models/DFM-Mimir-v1.5}"
VERBALIZER_MODEL="${VERBALIZER_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:-musr}"
PROVIDER_CLS="${PROVIDER_CLS:-nla.hrm.testing.FakeCompletionProvider}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"
mkdir -p "$OUT_DIR"

echo "[1/7] corpus"
$PYTHON -m nla.hrm.build_prompt_corpus --output "$OUT_DIR/corpus.jsonl" "$@"

echo "[2/7] extract (stage0)"
$PYTHON -m nla.hrm.stage0_hrm --corpus "$OUT_DIR/corpus.jsonl" --base-model "$BASE_MODEL" \
    --device "$DEVICE" --output "$OUT_DIR/base.parquet"

echo "[3/7] diagnostics"
$PYTHON -m nla.hrm.diagnostics --input "$OUT_DIR/base.parquet"

echo "[4/7] split"
$PYTHON -m nla.hrm.split --base "$OUT_DIR/base.parquet" --holdout-dataset "$HOLDOUT_DATASET" \
    --output-dir "$OUT_DIR/splits"

echo "[5/7] norm_stats"
$PYTHON -m nla.hrm.norm_stats --train-parquet "$OUT_DIR/splits/av_sft.parquet" \
    --train-parquet "$OUT_DIR/splits/ar_sft.parquet" --train-parquet "$OUT_DIR/splits/rl.parquet" \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/norm_stats.json"

echo "[6/7] explain"
$PYTHON -m nla.hrm.explain --input "$OUT_DIR/splits/av_sft.parquet" --provider-cls "$PROVIDER_CLS" \
    --output "$OUT_DIR/splits/av_sft_explained.parquet"
$PYTHON -m nla.hrm.explain --input "$OUT_DIR/splits/ar_sft.parquet" --provider-cls "$PROVIDER_CLS" \
    --output "$OUT_DIR/splits/ar_sft_explained.parquet"

echo "[7/7] build"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/av_sft_explained.parquet" --stage av_sft \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/av_sft.parquet"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/ar_sft_explained.parquet" --stage ar_sft \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/ar_sft.parquet"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/rl.parquet" --stage rl \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/rl.parquet"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/eval_iid.parquet" --stage rl \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/eval_iid.parquet"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/eval_ood.parquet" --stage rl \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/eval_ood.parquet"
$PYTHON -m nla.hrm.build --input "$OUT_DIR/splits/judge_subset.parquet" --stage rl \
    --verbalizer-model "$VERBALIZER_MODEL" --output "$OUT_DIR/judge_subset.parquet"

echo "done -> $OUT_DIR"
