# Reproduction notes

Server project: `/home/ubuntu/caozheng/collapsemodel_rebuttal_20261117`.
Interpreter: `.venv-exp/bin/python` (same as the fresh campaign).
Data: `/mnt/datadisk0/datasets/collapsemodel_20261117/fresh`.
Baseline quality audit: `runs/`; new outputs: `linear_runs/`.

Never overwrite either archived output directory. The campaign driver fails on
an incomplete pre-existing selection directory rather than silently replacing it.

The new selection policy needs only the 15D incidence rows and queried validation
losses; it does not use GPU operations. Set `CUDA_VISIBLE_DEVICES=0`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before running.
No passwords or SSH credentials are required by the code or included in archives.

From the project root, for a *new* reproduction output folder:

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p linear_reproduction
.venv-exp/bin/python linear_bai_20261117/check_policies.py linear_reproduction/implementation_checks.json
.venv-exp/bin/python linear_bai_20261117/run_comparison.py design --out linear_reproduction/designs.json
.venv-exp/bin/python linear_bai_20261117/run_comparison.py select \
  --old runs/domainnet_clipart_hash_d128_20261117 \
  --out linear_reproduction/domainnet_clipart_hash_d128_20261117 \
  --design linear_reproduction/designs.json
.venv-exp/bin/python linear_bai_20261117/run_comparison.py audit \
  --old runs/domainnet_clipart_hash_d128_20261117 \
  --unit /mnt/datadisk0/datasets/collapsemodel_20261117/fresh/domainnet_clipart_hash_d128_20261117 \
  --out linear_reproduction/domainnet_clipart_hash_d128_20261117
.venv-exp/bin/python linear_bai_20261117/run_comparison.py timing \
  --unit /mnt/datadisk0/datasets/collapsemodel_20261117/fresh/domainnet_clipart_hash_d128_20261117 \
  --out linear_reproduction/domainnet_clipart_hash_d128_20261117
```

Repeat the select phase for all 56 predefined units BEFORE beginning the audit
phase if reproducing the full campaign. Run timing sequentially, without your
other compute jobs, only after selection/audit work has finished. Allocation
sequences can be shared for quality replay; the timing function recomputes them.

The report scripts are postprocessing only. `summarize_comparison.py OUTPUT_ROOT`
accepts an explicit output root. `build_report.py OUTPUT_ROOT` requires the final
verification record and all 189 timing rows; it refuses a partial delivery.
`verify_outputs.py` is configured for the recorded server project tree and checks
the 56 model hashes in addition to decisions, budgets and timing reference IDs.

The source archive also contains `fresh_20261117/`, including raw-data preparation,
feature extraction and original full-model fit code. Those frozen sources and
data manifests identify the prior fresh campaign; this follow-up deliberately
reuses that campaign's immutable models for quality comparisons. Do not describe
the follow-up's replay as a new full training campaign or independent test set.
