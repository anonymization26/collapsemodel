#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 DATASET ENCODER MANIFEST_ROOT FEATURE_ROOT OUTPUT_ROOT [CONFIG]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=$1
ENCODER=$2
MANIFEST_ROOT=$3
FEATURE_ROOT=$4
OUTPUT_ROOT=$5
CONFIG=${6:-"$ROOT/code/configs/target_conditioned_e2/experiment_v1.json"}

if [[ -z ${COLLAPSEMODEL_GIT_REVISION:-} && -f "$ROOT/SOURCE_REVISION" ]]; then
  export COLLAPSEMODEL_GIT_REVISION
  COLLAPSEMODEL_GIT_REVISION=$(tr -d '[:space:]' < "$ROOT/SOURCE_REVISION")
fi

MANIFEST_DIR="$MANIFEST_ROOT/$DATASET"
FEATURE_DIR="$FEATURE_ROOT/$DATASET/$ENCODER"
OUT_DIR="$OUTPUT_ROOT/$DATASET/$ENCODER"
SELECTION="$OUT_DIR/selection.json"

mkdir -p "$OUT_DIR"

python3 "$ROOT/scripts/run_target_conditioned_e2.py" select \
  --manifest-dir "$MANIFEST_DIR" \
  --features "$FEATURE_DIR/features.npz" \
  --metadata "$FEATURE_DIR/metadata.json" \
  --config "$CONFIG" \
  --output "$SELECTION"

python3 "$ROOT/scripts/run_target_conditioned_e2.py" evaluate \
  --manifest-dir "$MANIFEST_DIR" \
  --features "$FEATURE_DIR/features.npz" \
  --metadata "$FEATURE_DIR/metadata.json" \
  --config "$CONFIG" \
  --selection "$SELECTION" \
  --out-dir "$OUT_DIR"
