#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: $0 <e1|h5> <seed> [run-id]}"
SEED="${2:?usage: $0 <e1|h5> <seed> [run-id]}"
RUN_ID="${3:-expanded_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-target-conditioned-v2}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/target_conditioned}"
LOG_DIR="$RESULT_ROOT/logs/$RUN_ID"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

echo "START_UTC=$(date -u +%FT%TZ) MODE=$MODE SEED=$SEED RUN_ID=$RUN_ID"

case "$MODE" in
  e1)
    python3 scripts/run_target_conditioned_e1.py \
      --out-dir "$RESULT_ROOT/e1_synthetic/$RUN_ID/shards/seed_$SEED" \
      --seeds "$SEED" \
      --dimensions 16 32 64 \
      --candidate-counts 8 12 16 \
      --budgets 1 3 \
      --target-samples 16 32 128 512 \
      --target-rank-fractions 0.125 0.25 0.5 \
      --source-samples 64 \
      --target-test-samples 512 \
      --sketch-ranks 2 4 8 16 \
      --shift-levels 0.0 0.5 1.0 \
      --random-repeats 20
    ;;
  h5)
    python3 scripts/run_target_conditioned_h5.py \
      --out-dir "$RESULT_ROOT/e5_sketch_certificate/$RUN_ID/shards/seed_$SEED" \
      --seeds "$SEED" \
      --dimensions 16 32 64 \
      --candidate-counts 8 16 \
      --budgets 1 3 \
      --target-samples 32 128 \
      --target-rank-fractions 0.125 0.25 0.5 \
      --source-samples 64 \
      --sketch-ranks 1 2 4 8 16 32
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

echo "DONE_UTC=$(date -u +%FT%TZ) MODE=$MODE SEED=$SEED RUN_ID=$RUN_ID"
