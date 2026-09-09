#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 ENCODER [...]" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
SOURCE_DIR="${SOURCE_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_natural_shortlist_corrected_v3}"
STAGE1_SAMPLES="${STAGE1_SAMPLES:-500}"

cd "$PROJECT_ROOT"
STAGE1_BLAS_THREADS="${STAGE1_BLAS_THREADS:-8}"
export OPENBLAS_NUM_THREADS="$STAGE1_BLAS_THREADS"
export OMP_NUM_THREADS="$STAGE1_BLAS_THREADS"
export MKL_NUM_THREADS="$STAGE1_BLAS_THREADS"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$RESULT_ROOT"

echo "START_UTC=$(date -u +%FT%TZ) ENCODERS=$*"
for encoder in "$@"; do
  out_dir="$RESULT_ROOT/$encoder"
  mkdir -p "$out_dir"
  python3 scripts/two_stage_natural_shortlist.py screen \
    --source-dir "$SOURCE_DIR" \
    --out-dir "$out_dir" \
    --encoder "$encoder" \
    --cache-samples 5000 \
    --stage1-samples "$STAGE1_SAMPLES" \
    --sample-seed 20260905 \
    --source-weighting equal_source \
    --top-k 20 \
    --shortlist-sizes 3 5 10 \
    > "$out_dir/screen.log" 2>&1
  echo "SCREEN_DONE=$encoder UTC=$(date -u +%FT%TZ)"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
