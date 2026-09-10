#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-target-conditioned}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/target_conditioned}"
RUN_ID="${RUN_ID:-pilot_v1}"
LOG_DIR="$RESULT_ROOT/logs/$RUN_ID"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

echo "START_UTC=$(date -u +%FT%TZ) RUN_ID=$RUN_ID PROJECT_ROOT=$PROJECT_ROOT"

python3 -m unittest \
  tests.test_target_conditioned \
  tests.test_target_conditioned_e1 \
  -v \
  > "$LOG_DIR/tests.log" 2>&1

python3 scripts/run_target_conditioned_e0.py \
  --out-dir "$RESULT_ROOT/e0_theory_validation/$RUN_ID" \
  > "$LOG_DIR/e0.log" 2>&1

python3 scripts/run_target_conditioned_e1.py \
  --out-dir "$RESULT_ROOT/e1_synthetic/$RUN_ID" \
  --seeds 20260910 20260911 20260912 20260913 20260914 \
  --dimensions 16 32 \
  --candidate-counts 8 12 \
  --budgets 1 3 \
  --target-samples 32 128 \
  --source-samples 64 \
  --target-test-samples 512 \
  --sketch-ranks 2 4 8 \
  --shift-levels 0.0 0.5 1.0 \
  --random-repeats 20 \
  > "$LOG_DIR/e1.log" 2>&1

echo "DONE_UTC=$(date -u +%FT%TZ) RUN_ID=$RUN_ID"
