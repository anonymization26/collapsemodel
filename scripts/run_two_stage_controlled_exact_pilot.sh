#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
FEATURE_DIR="${FEATURE_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_controlled_exact_corrected_v3}"
ENCODERS="${ENCODERS:-resnet50 vit_b16 clip_b32 dinov2_b14}"
SEEDS="${SEEDS:-20260910 20260911 20260912}"
N_RANDOM="${N_RANDOM:-100}"

cd "$PROJECT_ROOT"
STAGE1_BLAS_THREADS="${STAGE1_BLAS_THREADS:-16}"
export OPENBLAS_NUM_THREADS="$STAGE1_BLAS_THREADS"
export OMP_NUM_THREADS="$STAGE1_BLAS_THREADS"
export MKL_NUM_THREADS="$STAGE1_BLAS_THREADS"
mkdir -p "$RESULT_ROOT/logs"

echo "START_UTC=$(date -u +%FT%TZ) ENCODERS=$ENCODERS SEEDS=$SEEDS N_RANDOM=$N_RANDOM"
pids=()
for encoder in $ENCODERS; do
  python3 scripts/two_stage_summary_only_controlled.py \
    --feature-dir "$FEATURE_DIR" \
    --out-dir "$RESULT_ROOT" \
    --encoder "$encoder" \
    --top-k 20 \
    --construction-seeds $SEEDS \
    --n-random "$N_RANDOM" \
    --include-full-rank \
    --output-prefix summary_exact \
    > "$RESULT_ROOT/logs/${encoder}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

python3 scripts/merge_two_stage_multiencoder_controlled.py \
  --input-dir "$RESULT_ROOT" \
  --out-dir "$RESULT_ROOT/aggregate" \
  --prefix summary_exact \
  --primary-method rank_l_gram
echo "DONE_UTC=$(date -u +%FT%TZ)"
