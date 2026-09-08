#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 NPU VARIANT [VARIANT ...]" >&2
  exit 2
fi

NPU="$1"
shift

PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
ARROW_ROOT="${ARROW_ROOT:-/data/67/paper14/hf_cache/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/Paper06/features/two_stage_source_splits_n5000}"
SAMPLES="${SAMPLES:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "START_UTC=$(date -u +%FT%TZ) NPU=$NPU VARIANTS=$*"
for variant in "$@"; do
  echo "VARIANT_START=$variant UTC=$(date -u +%FT%TZ)"
  python3 -u scripts/extract_two_stage_source_features_npu.py \
    --variant "$variant" \
    --arrow-root "$ARROW_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --npu "$NPU" \
    --samples "$SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS"
  echo "VARIANT_DONE=$variant UTC=$(date -u +%FT%TZ)"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
