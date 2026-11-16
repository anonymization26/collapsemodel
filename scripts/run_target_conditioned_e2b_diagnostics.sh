#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${E2B_WORK_ROOT:-$ROOT}"
PYTHON="${E2B_PYTHON:-$WORK/.venv/bin/python}"
UPSTREAM="${E2B_REVISION_OUTPUT:?set the completed R1 output directory}"
OUTPUT="${E2B_DIAGNOSTICS_OUTPUT:?set a new diagnostics output directory}"
: "${SOURCE_GIT_REVISION:?set the source base commit}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
[[ "$(<"$UPSTREAM/status.txt")" == "complete" ]]
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
mkdir "$OUTPUT/logs"
printf 'running\n' > "$OUTPUT/status.txt"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "failed exit=%s\n" "$rc" > "$OUTPUT/status.txt"; fi' EXIT
cd "$ROOT"
sha256sum scripts/run_target_conditioned_e2b_diagnostics.sh \
  scripts/run_target_conditioned_e2b_diagnostics.py \
  code/configs/target_conditioned_e2b/diagnostics_v1.json > "$OUTPUT/launch_sha256.txt"
pids=()
for encoder in resnet50 dinov2_b14; do
  "$PYTHON" -u scripts/run_target_conditioned_e2b_diagnostics.py run \
    --config "$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json" \
    --diagnostics-config "$ROOT/code/configs/target_conditioned_e2b/diagnostics_v1.json" \
    --stage-bundle "$WORK/stage_bundles/$encoder/test-audit" --encoder "$encoder" \
    --screen-dir "$UPSTREAM/screen/$encoder" --validation-dir "$UPSTREAM/validation/$encoder" \
    --audit-dir "$UPSTREAM/test_audit/$encoder" --output-dir "$OUTPUT/$encoder" \
    > "$OUTPUT/logs/$encoder.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" -ne 0 ]]; then exit 1; fi
"$PYTHON" scripts/run_target_conditioned_e2b_diagnostics.py verify-pairing \
  --diagnostics-dir "$OUTPUT/resnet50" --diagnostics-dir "$OUTPUT/dinov2_b14" \
  --output "$OUTPUT/pairing_verification.json"
printf 'complete\n' > "$OUTPUT/status.txt"
