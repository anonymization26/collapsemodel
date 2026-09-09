#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 NPU ENCODER:TARGET_SPLIT_SEED [...]" >&2
  exit 2
fi

NPU="$1"
shift

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
SOURCE_DIR="${SOURCE_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
TARGET_DIR="${TARGET_DIR:-/data/Paper06/features/two_stage_target_splits_n5000}"
BASE_RESULTS="${BASE_RESULTS:-/data/Paper06/results/two_stage_natural_shortlist_corrected_v3}"
RESULT_ROOT="${RESULT_ROOT:-/data/Paper06/results/two_stage_split_sensitivity_corrected_v3}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
mkdir -p "$RESULT_ROOT"

echo "START_UTC=$(date -u +%FT%TZ) NPU=$NPU JOBS=$*"
echo "SCRIPT_SHA256=$(sha256sum scripts/two_stage_natural_shortlist.py | cut -d' ' -f1)"
for spec in "$@"; do
  encoder="${spec%%:*}"
  target_split_seed="${spec#*:}"
  if [[ "$encoder" == "$target_split_seed" ]]; then
    echo "invalid job '$spec'; expected ENCODER:TARGET_SPLIT_SEED" >&2
    exit 2
  fi
  manifest="$BASE_RESULTS/$encoder/${encoder}_screening_manifest.json"
  out_dir="$RESULT_ROOT/split_${target_split_seed}/$encoder"
  mkdir -p "$out_dir"
  echo "JOB_START=$spec UTC=$(date -u +%FT%TZ)"
  python3 -u scripts/two_stage_natural_shortlist.py adapt \
    --manifest "$manifest" \
    --source-dir "$SOURCE_DIR" \
    --target-dir "$TARGET_DIR" \
    --output "$out_dir/adaptation_results.csv" \
    --encoder "$encoder" \
    --npu "$NPU" \
    --cache-samples 5000 \
    --adapter-samples 500 \
    --source-sample-seed 20260905 \
    --target-split-seed "$target_split_seed" \
    --seed-start 0 \
    --n-seeds 5 \
    --width 256 \
    --steps 600 \
    --batch-size 128 \
    --ridge 0.01 \
    --shard-index 0 \
    --shard-count 1 \
    > "$out_dir/adapt.log" 2>&1
  python3 scripts/two_stage_natural_shortlist.py summarize \
    --manifest "$manifest" \
    --adaptation-csv "$out_dir/adaptation_results.csv" \
    --out-dir "$out_dir" \
    --n-random 100 \
    --random-seed 20260905 \
    --tie-tolerance 1e-12 \
    > "$out_dir/summarize.log" 2>&1
  python3 -u scripts/two_stage_natural_shortlist.py heldout \
    --selection-manifest "$out_dir/${encoder}_shortlist_manifest.json" \
    --target-dir "$TARGET_DIR" \
    --output "$out_dir/${encoder}_heldout_results.csv" \
    --encoder "$encoder" \
    --npu "$NPU" \
    > "$out_dir/heldout.log" 2>&1
  echo "JOB_DONE=$spec UTC=$(date -u +%FT%TZ) ROWS=$(wc -l < "$out_dir/adaptation_results.csv")"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
