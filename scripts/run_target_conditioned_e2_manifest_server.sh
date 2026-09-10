#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 DATASET_ROOT RECEIPT MANIFEST_DIR SOURCE_ARTIFACT [SOURCE_ARTIFACT ...]" >&2
  exit 2
fi

DATASET_ROOT="$1"
RECEIPT="$2"
MANIFEST_DIR="$3"
shift 3
SOURCE_ARTIFACTS=("$@")
PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-target-conditioned}"

for path in "$DATASET_ROOT" "$RECEIPT" "$MANIFEST_DIR" "${SOURCE_ARTIFACTS[@]}"; do
  if [[ "$path" != /data/* ]]; then
    echo "E2 data artifacts must be stored below /data: $path" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
mkdir -p "$MANIFEST_DIR"

arguments=()
for artifact in "${SOURCE_ARTIFACTS[@]}"; do
  arguments+=(--source-artifact "$artifact")
done

python3 -u scripts/build_target_conditioned_e2_manifest.py \
  --dataset-root "$DATASET_ROOT" \
  --receipt "$RECEIPT" \
  --output-dir "$MANIFEST_DIR" \
  --blocks-per-source-domain 4 \
  --split-seed target-conditioned-e2-v1 \
  --verify-images \
  "${arguments[@]}"

python3 -u scripts/validate_target_conditioned_e2_artifacts.py manifest \
  --manifest-dir "$MANIFEST_DIR" \
  --dataset-root "$DATASET_ROOT" \
  --verify-images \
  "${arguments[@]}"
