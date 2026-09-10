#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 NPU DATASET_ROOT MANIFEST_DIR CACHE_ROOT VARIANT [VARIANT ...]" >&2
  exit 2
fi

NPU="$1"
DATASET_ROOT="$2"
MANIFEST_DIR="$3"
CACHE_ROOT="$4"
shift 4
VARIANTS=("$@")
PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-target-conditioned}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-4}"

for path in "$DATASET_ROOT" "$MANIFEST_DIR" "$CACHE_ROOT"; do
  if [[ "$path" != /data/* ]]; then
    echo "E2 data artifacts must be stored below /data: $path" >&2
    exit 2
  fi
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT/logs"

echo "START_UTC=$(date -u +%FT%TZ) NPU=$NPU VARIANTS=${VARIANTS[*]}"
for variant in "${VARIANTS[@]}"; do
  output_dir="$CACHE_ROOT/$variant"
  log_path="$CACHE_ROOT/logs/$variant.log"
  mkdir -p "$output_dir"
  python3 -u scripts/extract_target_conditioned_e2_features_npu.py \
    --variant "$variant" \
    --dataset-root "$DATASET_ROOT" \
    --manifest-dir "$MANIFEST_DIR" \
    --output-dir "$output_dir" \
    --npu "$NPU" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    2>&1 | tee "$log_path"
done
echo "DONE_UTC=$(date -u +%FT%TZ) NPU=$NPU"
