#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${E2B_WORK_ROOT:-$ROOT}"
PYTHON="${E2B_PYTHON:-$WORK/.venv/bin/python}"
OUTPUT="${E2B_REVISION_OUTPUT:?set a new revision output directory}"
: "${SOURCE_GIT_REVISION:?set the base commit; source hashes also record local additions}"
CONFIG="$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"] == "ready"; assert r["sample_count"] == 46080' \
  "$WORK/preparation_report.json"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
printf 'running\n' > "$OUTPUT/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed exit=%s\n" "$rc" > "$OUTPUT/status.txt"; fi' EXIT
cd "$ROOT"
sha256sum "$CONFIG" "$WORK/preparation_report.json" \
  scripts/run_target_conditioned_e2b_revision.sh scripts/run_target_conditioned_e2b_isolated.py \
  scripts/run_target_conditioned_e2b_shortlist.py scripts/summarize_target_conditioned_e2b.py \
  code/metrics/e2b_stage_bundle.py code/metrics/target_conditioned_e2b.py \
  > "$OUTPUT/input_sha256.txt"
printf '%s\n' "$SOURCE_GIT_REVISION" > "$OUTPUT/base_git_revision.txt"

run_encoder() {
  local encoder="$1"
  for stage in screen validate test-audit; do
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$encoder" "$stage"
    printf '%s\n' "$stage" > "$OUTPUT/$encoder.status.txt"
    local folder="$stage"
    local extra=()
    if [[ "$stage" == "validate" ]]; then
      folder=validation
      extra=(--screen-dir "$OUTPUT/screen/$encoder")
    elif [[ "$stage" == "test-audit" ]]; then
      folder=test_audit
      extra=(--screen-dir "$OUTPUT/screen/$encoder" --validation-dir "$OUTPUT/validation/$encoder")
    fi
    "$PYTHON" -u scripts/run_target_conditioned_e2b_isolated.py "$stage" \
      --stage-bundle "$WORK/stage_bundles/$encoder/$stage" --config "$CONFIG" \
      --encoder "$encoder" --output-dir "$OUTPUT/$folder/$encoder" "${extra[@]}" || return $?
  done
  printf 'complete\n' > "$OUTPUT/$encoder.status.txt"
}

pids=()
for encoder in resnet50 dinov2_b14; do
  run_encoder "$encoder" > "$OUTPUT/logs/$encoder.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi
"$PYTHON" scripts/summarize_target_conditioned_e2b.py --config "$CONFIG" \
  --audit-dir "$OUTPUT/test_audit/resnet50" --audit-dir "$OUTPUT/test_audit/dinov2_b14" \
  --output-dir "$OUTPUT/summary"
printf 'complete\n' > "$OUTPUT/status.txt"
