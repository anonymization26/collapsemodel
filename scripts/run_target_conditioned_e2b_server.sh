#!/usr/bin/env bash
set -eo pipefail

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -r "$ASCEND_ENV" ]]; then
  printf 'Ascend environment file is not readable: %s\n' "$ASCEND_ENV" >&2
  exit 2
fi
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PYTHONPATH="${PYTHONPATH:-}"
source "$ASCEND_ENV"
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${E2B_WORK_ROOT:?set E2B_WORK_ROOT on a large filesystem}"
NPU_CLIP="${NPU_CLIP:-4}"
NPU_RESNET="${NPU_RESNET:-4}"
NPU_DINO="${NPU_DINO:-6}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
CONFIG="$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json"
DOWNLOADS="$WORK_ROOT/downloads"
DATASET="$WORK_ROOT/dataset"
RESULTS="$WORK_ROOT/results"
MANIFEST="$RESULTS/data_manifest"
ANCHOR="$RESULTS/features/clip_b32"
CANDIDATES="$RESULTS/candidates/candidates.json"
REVISION="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || cat "$ROOT/SOURCE_REVISION")"

mkdir -p "$WORK_ROOT/logs" "$RESULTS/features" "$RESULTS/screen" \
  "$RESULTS/validation" "$RESULTS/test_audit" "$RESULTS/summary"

stage="${1:-all}"

if [[ "$stage" == "download" || "$stage" == "all" ]]; then
  python3 "$ROOT/scripts/prepare_target_conditioned_e2b_domainnet.py" download \
    --config "$CONFIG" --download-dir "$DOWNLOADS"
fi

if [[ "$stage" == "manifest" || "$stage" == "all" ]]; then
  python3 "$ROOT/scripts/prepare_target_conditioned_e2b_domainnet.py" build \
    --config "$CONFIG" --download-dir "$DOWNLOADS" \
    --extracted-root "$DATASET" --output-dir "$MANIFEST"
fi

if [[ "$stage" == "features" || "$stage" == "all" ]]; then
  SOURCE_GIT_REVISION="$REVISION" python3 \
    "$ROOT/scripts/extract_target_conditioned_e2b_features_npu.py" \
    --variant clip_b32 --dataset-root "$DATASET" --manifest-dir "$MANIFEST" \
    --config "$CONFIG" --output-dir "$ANCHOR" --npu "$NPU_CLIP" \
    --batch-size "$BATCH_SIZE" --workers "$WORKERS"

  mkdir -p "$(dirname "$CANDIDATES")"
  python3 "$ROOT/scripts/build_target_conditioned_e2b_candidates.py" \
    --manifest-dir "$MANIFEST" --config "$CONFIG" \
    --anchor-features "$ANCHOR/features.npz" \
    --anchor-metadata "$ANCHOR/metadata.json" --output "$CANDIDATES"

  SOURCE_GIT_REVISION="$REVISION" python3 \
    "$ROOT/scripts/extract_target_conditioned_e2b_features_npu.py" \
    --variant resnet50 --dataset-root "$DATASET" --manifest-dir "$MANIFEST" \
    --config "$CONFIG" --output-dir "$RESULTS/features/resnet50" \
    --npu "$NPU_RESNET" --batch-size "$BATCH_SIZE" --workers "$WORKERS" &
  resnet_pid=$!
  SOURCE_GIT_REVISION="$REVISION" python3 \
    "$ROOT/scripts/extract_target_conditioned_e2b_features_npu.py" \
    --variant dinov2_b14 --dataset-root "$DATASET" --manifest-dir "$MANIFEST" \
    --config "$CONFIG" --output-dir "$RESULTS/features/dinov2_b14" \
    --npu "$NPU_DINO" --batch-size "$BATCH_SIZE" --workers "$WORKERS" &
  dino_pid=$!
  wait "$resnet_pid"
  wait "$dino_pid"
fi

run_stage() {
  local command="$1"
  local encoder="$2"
  local output="$3"
  shift 3
  SOURCE_GIT_REVISION="$REVISION" python3 \
    "$ROOT/scripts/run_target_conditioned_e2b_shortlist.py" "$command" \
    --manifest-dir "$MANIFEST" --config "$CONFIG" --candidates "$CANDIDATES" \
    --anchor-features "$ANCHOR/features.npz" \
    --anchor-metadata "$ANCHOR/metadata.json" \
    --features "$RESULTS/features/$encoder/features.npz" \
    --metadata "$RESULTS/features/$encoder/metadata.json" \
    --encoder "$encoder" --output-dir "$output" "$@"
}

if [[ "$stage" == "screen" || "$stage" == "all" ]]; then
  for encoder in resnet50 dinov2_b14; do
    run_stage screen "$encoder" "$RESULTS/screen/$encoder"
  done
fi

if [[ "$stage" == "validate" || "$stage" == "all" ]]; then
  for encoder in resnet50 dinov2_b14; do
    run_stage validate "$encoder" "$RESULTS/validation/$encoder" \
      --screen-dir "$RESULTS/screen/$encoder"
  done
fi

if [[ "$stage" == "test-audit" || "$stage" == "all" ]]; then
  for encoder in resnet50 dinov2_b14; do
    run_stage test-audit "$encoder" "$RESULTS/test_audit/$encoder" \
      --screen-dir "$RESULTS/screen/$encoder" \
      --validation-dir "$RESULTS/validation/$encoder"
  done
  python3 "$ROOT/scripts/summarize_target_conditioned_e2b.py" \
    --config "$CONFIG" \
    --audit-dir "$RESULTS/test_audit/resnet50" \
    --audit-dir "$RESULTS/test_audit/dinov2_b14" \
    --output-dir "$RESULTS/summary"
fi
