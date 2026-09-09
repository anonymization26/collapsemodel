#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
FEATURE_DIR="${FEATURE_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_controlled_rank_sweep_corrected_v3}"
RANKS="${RANKS:-10 20 50}"
ENCODERS="${ENCODERS:-resnet50 vit_b16 clip_b32 dinov2_b14}"
N_RANDOM="${N_RANDOM:-0}"
INCLUDE_FULL_RANK="${INCLUDE_FULL_RANK:-0}"

cd "$PROJECT_ROOT"
STAGE1_BLAS_THREADS="${STAGE1_BLAS_THREADS:-16}"
export OPENBLAS_NUM_THREADS="$STAGE1_BLAS_THREADS"
export OMP_NUM_THREADS="$STAGE1_BLAS_THREADS"
export MKL_NUM_THREADS="$STAGE1_BLAS_THREADS"
mkdir -p "$RESULT_ROOT"

echo "START_UTC=$(date -u +%FT%TZ) RANKS=$RANKS ENCODERS=$ENCODERS"
for rank in $RANKS; do
  rank_dir="$RESULT_ROOT/L${rank}"
  mkdir -p "$rank_dir/logs"
  pids=()
  for encoder in $ENCODERS; do
    command=(
      python3 scripts/two_stage_summary_only_controlled.py
      --feature-dir "$FEATURE_DIR"
      --out-dir "$rank_dir"
      --encoder "$encoder"
      --top-k "$rank"
      --n-random "$N_RANDOM"
      --output-prefix summary_stopping
    )
    if [[ "$INCLUDE_FULL_RANK" == "1" ]]; then
      command+=(--include-full-rank)
    fi
    "${command[@]}" > "$rank_dir/logs/${encoder}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  python3 scripts/merge_two_stage_multiencoder_controlled.py \
    --input-dir "$rank_dir" \
    --out-dir "$rank_dir/aggregate" \
    --prefix summary_stopping \
    --primary-method rank_l_gram
  echo "RANK_DONE=$rank UTC=$(date -u +%FT%TZ)"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
