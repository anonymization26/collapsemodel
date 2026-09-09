#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ENCODER" >&2
  exit 2
fi

ENCODER="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
SOURCE_DIR="${SOURCE_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
BASE_RESULTS="${BASE_RESULTS:-/data/Paper06/results/two_stage_natural_shortlist_corrected_v3}"
UTILITY_ROOT="${UTILITY_ROOT:-/data/Paper06/results/two_stage_repeated_cv_utility_corrected_v3/folds_5_seeds_20260905_20260906_20260907}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_repeated_cv_source_lodo_corrected_v3}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "START_UTC=$(date -u +%FT%TZ) ENCODER=$ENCODER"
echo "SCRIPT_SHA256=$(sha256sum scripts/two_stage_source_lodo.py | cut -d' ' -f1)"
python3 scripts/two_stage_source_lodo.py \
  --source-dir "$SOURCE_DIR" \
  --adaptation-csv "$UTILITY_ROOT/$ENCODER/adaptation_results.csv" \
  --adaptation-manifest "$BASE_RESULTS/$ENCODER/${ENCODER}_screening_manifest.json" \
  --out-dir "$RESULT_ROOT" \
  --encoder "$ENCODER" \
  --cache-samples 5000 \
  --stage1-samples 1000 \
  --sample-seed 20260905 \
  --top-k 20 \
  --shortlist-sizes 3 5 10 \
  --n-random 100 \
  --random-seed 20260905 \
  --tie-tolerance 1e-12
echo "DONE_UTC=$(date -u +%FT%TZ)"
