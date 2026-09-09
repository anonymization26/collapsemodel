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
ADAPTATION_ROOT="${ADAPTATION_ROOT:-/data/Paper06/results/two_stage_repeated_cv_utility_corrected_v3/folds_5_seeds_20260905_20260906_20260907}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_cv_lodo_corrected_v3}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "START_UTC=$(date -u +%FT%TZ) ENCODER=$ENCODER"
echo "SCRIPT_SHA256=$(sha256sum scripts/two_stage_natural_shortlist.py | cut -d' ' -f1)"
for domain in medical digits_characters general_vision rendered_text; do
  case "$domain" in
    medical) shortlist_sizes=(3 5) ;;
    digits_characters|general_vision) shortlist_sizes=(3 5 8) ;;
    rendered_text) shortlist_sizes=(3 5 10) ;;
  esac
  out_dir="$RESULT_ROOT/exclude_${domain}/$ENCODER"
  mkdir -p "$out_dir"
  echo "FOLD_START=$domain UTC=$(date -u +%FT%TZ)"
  python3 scripts/two_stage_natural_shortlist.py screen \
    --source-dir "$SOURCE_DIR" \
    --out-dir "$out_dir" \
    --encoder "$ENCODER" \
    --cache-samples 5000 \
    --stage1-samples 1000 \
    --sample-seed 20260905 \
    --top-k 20 \
    --exclude-source-domains "$domain" \
    --shortlist-sizes "${shortlist_sizes[@]}" \
    > "$out_dir/screen.log" 2>&1
  python3 scripts/two_stage_natural_shortlist.py summarize \
    --manifest "$out_dir/${ENCODER}_screening_manifest.json" \
    --adaptation-manifest "$BASE_RESULTS/$ENCODER/${ENCODER}_screening_manifest.json" \
    --adaptation-csv "$ADAPTATION_ROOT/$ENCODER/adaptation_results.csv" \
    --out-dir "$out_dir" \
    --n-random 100 \
    --random-seed 20260905 \
    --tie-tolerance 1e-12 \
    > "$out_dir/summarize.log" 2>&1
  echo "FOLD_DONE=$domain UTC=$(date -u +%FT%TZ)"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
