#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${E2B_WORK_ROOT:-$ROOT}"
PYTHON="${E2B_PYTHON:-$WORK_ROOT/.venv/bin/python}"
FROZEN="$ROOT/results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1"
CONFIG="$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json"
export COLLAPSE_WEIGHT_DIR="$WORK_ROOT/weights"
export COLLAPSE_DINO_REPOSITORY="$WORK_ROOT/dinov2_repository"
export COLLAPSE_DINO_CHECKPOINT="$WORK_ROOT/weights/dinov2_vitb14_pretrain.pth"
export TORCH_HOME="$WORK_ROOT/torch_cache"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK_ROOT/logs"

case "${1:-check}" in
  check)
    "$PYTHON" "$ROOT/scripts/check_target_conditioned_e2b_cuda.py" \
      --reference-dir "$FROZEN/features" --output "$WORK_ROOT/environment_check.json"
    ;;
  restore)
    "$PYTHON" -u "$ROOT/scripts/materialize_target_conditioned_e2b_parquet.py" \
      --manifest-dir "$FROZEN/data_manifest" --dataset-root "$WORK_ROOT/dataset" \
      --work-dir "$WORK_ROOT/downloads" --endpoint "${E2B_DATA_ENDPOINT:-https://hf-mirror.com}" \
      --workers "${E2B_DOWNLOAD_WORKERS:-12}" --chunk-mib 32
    ;;
  features)
    : "${SOURCE_GIT_REVISION:?set the base Git revision; source file hashes record local patches}"
    variants=(resnet50 dinov2_b14 clip_b32)
    pids=()
    for gpu in 0 1 2; do
      variant="${variants[$gpu]}"
      "$PYTHON" -u "$ROOT/scripts/extract_target_conditioned_e2b_features.py" \
        --variant "$variant" --device "cuda:$gpu" --batch-size 64 --workers 4 \
        --dataset-root "$WORK_ROOT/dataset" --manifest-dir "$FROZEN/data_manifest" \
        --config "$CONFIG" --expected-metadata "$FROZEN/features/$variant/metadata.json" \
        --output-dir "$WORK_ROOT/reconstructed_features/$variant" \
        > "$WORK_ROOT/logs/extract-$variant.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "$pid" || status=1
    done
    exit "$status"
    ;;
  bundles)
    for encoder in resnet50 dinov2_b14; do
      "$PYTHON" "$ROOT/scripts/prepare_target_conditioned_e2b_stage_bundles.py" \
        --manifest-dir "$FROZEN/data_manifest" --config "$CONFIG" \
        --candidates "$FROZEN/candidates/candidates.json" \
        --features "$WORK_ROOT/reconstructed_features/$encoder/features.npz" \
        --metadata "$WORK_ROOT/reconstructed_features/$encoder/metadata.json" \
        --encoder "$encoder" --output-dir "$WORK_ROOT/stage_bundles/$encoder"
    done
    ;;
  *)
    printf 'Usage: %s {check|restore|features|bundles}\n' "$0" >&2
    exit 2
    ;;
esac
