# Fresh rebuttal experiments (2026-11-17)

This directory implements the user's request to restart from new data, after
learning the historical feature caches were on another server. No prior cache
or prior outcome is an input. The base repository is Rebuttal commit
`51fce47f26c6f447edc89808bd5fe280e18cba71`.

Read `Protocol.md` and `ExecutionDetails.md` before interpreting any result.
Completed results are under `../runs/fresh-20261117/Results.md`.

## Server layout

- Code/environment: `/home/ubuntu/caozheng/collapsemodel_rebuttal_20261117`
- New raw data/features: `/mnt/datadisk0/datasets/collapsemodel_20261117`
- Python: project `.venv-exp/bin/python`
- Only physical GPU 0 is authorized; GPU commands use
  `CUDA_VISIBLE_DEVICES=0` and `flock /home/ubuntu/caozheng/gpu0.lock`.
- CPU quality auditing uses at most four independent workers, BLAS=1 each.
  Independent timing runs are serial with respect to campaign compute stages.
- The environment reuses an existing installed torch runtime read-only via a
  `.pth` file; all experimental data, features and fitted readouts are new.

No password is saved in project files. The experiment driver does not publish
results, push git changes, or install cron jobs.

## Execution

Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` for CPU work.
Use `python check_core.py` to verify budgets, exact-additive identification and
class probability alignment. Prepare Optdigits with `prepare_optdigits.py`.
For each unit, `run_fresh.py select` must finish and produce a selection lock
before `run_fresh.py audit`. `run_fresh.py timing` independently refits methods.
`run_ridge.py` performs calibration/selection on disjoint validation halves,
writes selection locks, then audits test predictions.

For DomainNet, `prepare_domainnet.py` downloads the official split files and
CRC-checked selected ZIP entries. `domainnet_features.py extract` computes new
features, and its `units` phase freezes projected source pools and target roles.
`run_campaign.py` sequences these stages, runs every predeclared unit, and records
failures in `runs/campaign_status.json`. Existing partial scientific outputs
require investigation; they are not automatically discarded or silently retried.
`cold_images.py` independently executes the no-feature-cache image pipeline.

After runs finish, use `audit_outputs.py --root runs` and
`summarize.py --root runs`. Never infer completion from a launched process.
The running server workflow also uses `finalize_campaign.py`: it waits for the
campaign, measures a separate cost breakdown, verifies artifacts, and writes
the final report and figures. Check both `campaign_status.json` and
`finalization_status.json`; the former can finish before final verification.

## Interpretation

Quality audits reuse all 455 candidate fits across policies to compare their
locked query trajectories. This is not a deployment time measurement. Shared
seeds and artificial pools do not create independent datasets. Keep unfavorable
pool constructions, per-target results and model-mismatch diagnostics.

Raw checkpoints and fitted weights are retained on the server and in the local
run folder when synchronized. They need not be committed to git. Record the
actual runtime versions and source hashes in `environment.json`.
