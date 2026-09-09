#!/usr/bin/env bash
set -eo pipefail

NPU="${NPU:-4}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/67/collapsemodel-two-stage}"
ARROW_ROOT="${ARROW_ROOT:-/data/67/paper14/hf_cache/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/Paper06/features/two_stage_source_splits_n5000_unlabeled}"
VARIANTS="${VARIANTS:-resnet50 vit_b16 clip_b32 dinov2_b14}"
DATASETS="${DATASETS:-bloodmnist breastmnist cifar10 cifar100 cifar100_coarse dermamnist fashion_mnist mnist octmnist organamnist organcmnist organsmnist pathmnist pneumoniamnist rendered_sst2 retinamnist stl10 svhn tiny_imagenet tissuemnist usps}"
SAMPLES="${SAMPLES:-5000}"
FORCE="${FORCE:-0}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "START_UTC=$(date -u +%FT%TZ) NPU=$NPU VARIANTS=$VARIANTS DATASETS=$DATASETS"
force_args=()
if [[ "$FORCE" == "1" ]]; then
  force_args+=(--force)
fi
for variant in $VARIANTS; do
  python3 -u scripts/extract_two_stage_source_features_npu.py \
    --variant "$variant" \
    --datasets $DATASETS \
    --arrow-root "$ARROW_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --npu "$NPU" \
    --samples "$SAMPLES" \
    --sample-seed 20260905 \
    --sampling unlabeled_random \
    --batch-size 64 \
    --workers 4 \
    "${force_args[@]}"
  echo "VARIANT_DONE=$variant UTC=$(date -u +%FT%TZ)"
done
echo "DONE_UTC=$(date -u +%FT%TZ)"
