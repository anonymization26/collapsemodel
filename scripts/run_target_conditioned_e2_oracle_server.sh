#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 DATASET ENCODER MANIFEST_ROOT FEATURE_ROOT PARENT_E2_ROOT OUTPUT_ROOT [ORACLE_CONFIG]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET=$1
ENCODER=$2
MANIFEST_ROOT=$3
FEATURE_ROOT=$4
PARENT_E2_ROOT=$5
OUTPUT_ROOT=$6
ORACLE_CONFIG=${7:-"$ROOT/code/configs/target_conditioned_e2/oracle_headroom_v1.json"}
PARENT_CONFIG="$ROOT/code/configs/target_conditioned_e2/experiment_v1.json"

if [[ -z ${COLLAPSEMODEL_GIT_REVISION:-} && -f "$ROOT/SOURCE_REVISION" ]]; then
  export COLLAPSEMODEL_GIT_REVISION
  COLLAPSEMODEL_GIT_REVISION=$(tr -d '[:space:]' < "$ROOT/SOURCE_REVISION")
fi

python3 "$ROOT/scripts/run_target_conditioned_e2_oracle.py" \
  --manifest-dir "$MANIFEST_ROOT/$DATASET" \
  --features "$FEATURE_ROOT/$DATASET/$ENCODER/features.npz" \
  --metadata "$FEATURE_ROOT/$DATASET/$ENCODER/metadata.json" \
  --parent-config "$PARENT_CONFIG" \
  --oracle-config "$ORACLE_CONFIG" \
  --parent-run-dir "$PARENT_E2_ROOT/$DATASET/$ENCODER" \
  --out-dir "$OUTPUT_ROOT/$DATASET/$ENCODER"
