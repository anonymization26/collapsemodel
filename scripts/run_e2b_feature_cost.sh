#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${E2B_WORK_ROOT:?set E2B_WORK_ROOT}"
OUT="${E2B_DECISION_OUTPUT:?set E2B_DECISION_OUTPUT}"
PID="${E2B_WAIT_PID:?set the decision-job PID}"
PYTHON="${E2B_PYTHON:-$WORK/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
deadline=$((SECONDS + 7200))
while kill -0 "$PID" 2>/dev/null; do
  if [[ -f "$OUT/timing/resnet50/timing.json" && -f "$OUT/timing/dinov2_b14/timing.json" ]]; then
    break
  fi
  if (( SECONDS >= deadline )); then
    printf 'Timed out waiting for isolated decision measurements.\n' >&2
    exit 1
  fi
  sleep 10
done
for encoder in resnet50 dinov2_b14; do
  test -f "$OUT/timing/$encoder/timing.json"
done
if [[ -n "${E2B_RIDGE_CORRECTION_SCRIPT:-}" ]]; then
  for encoder in resnet50 dinov2_b14; do
    "$PYTHON" -u "$E2B_RIDGE_CORRECTION_SCRIPT" --work-root "$WORK" \
      --base-config "$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json" \
      --readout-protocol "$ROOT/code/configs/target_conditioned_e2b/readout_projection_v1.json" \
      --protocol "$OUT/protocol.json" --encoder "$encoder" --readouts ridge \
      --output-root "$OUT/timing_ridge_float64/$encoder"
  done
fi
for encoder in resnet50 dinov2_b14; do
  "$PYTHON" -u "$ROOT/scripts/profile_e2b_feature_cost.py" --work-root "$WORK" \
    --encoder "$encoder" --output-root "$OUT/feature_cost/$encoder"
done
