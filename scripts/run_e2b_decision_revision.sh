#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${E2B_WORK_ROOT:?set E2B_WORK_ROOT}"
OUT="${E2B_DECISION_OUTPUT:?set a new E2B_DECISION_OUTPUT}"
PYTHON="${E2B_PYTHON:-$WORK/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir "$OUT"
BASE="$ROOT/code/configs/target_conditioned_e2b/domainnet_v1.json"
READOUT="$ROOT/code/configs/target_conditioned_e2b/readout_projection_v1.json"
PROTOCOL="$ROOT/code/configs/target_conditioned_e2b/decision_cost_v1.json"
cp "$PROTOCOL" "$OUT/protocol.json"
for encoder in resnet50 dinov2_b14; do
  for construction in original hash_partition; do
    for stage in screen validate test-audit; do
      "$PYTHON" -u "$ROOT/scripts/run_e2b_decision_challenge.py" "$stage" \
        --bundle "$WORK/stage_bundles/$encoder/$stage" --base-config "$BASE" \
        --readout-protocol "$READOUT" --protocol "$PROTOCOL" --encoder "$encoder" \
        --construction "$construction" --output-root "$OUT/$encoder/$construction"
    done
  done
done
# Timing is deliberately serial, after scientific runs have released the CPU.
for encoder in resnet50 dinov2_b14; do
  "$PYTHON" -u "$ROOT/scripts/time_e2b_decisions.py" --work-root "$WORK" \
    --base-config "$BASE" --readout-protocol "$READOUT" --protocol "$PROTOCOL" \
    --encoder "$encoder" --output-root "$OUT/timing/$encoder"
done
