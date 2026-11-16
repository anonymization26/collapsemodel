# Feature Geometry for Data-Pool Screening: Utility, Stability, and Limits

English-only reproduction supplement, 2026-09-26.

This copy includes the expanded Related Work and Appendix D on recent
data-selection methods. It retains the minimal reproduction scope: external
reference PDFs and later exploratory results not used by the paper are not added.
The abstract and main text clarify coefficient uncertainty, target feature energy,
and the distinction between shortlist omission and quality lost during validation
selection. The formulas and experimental evidence are unchanged.

Section 3.1 now presents the complete Target A-opt screening and validation
procedure in Algorithm 1, including inputs, scores, shortlists, fitting, and final
selection. The detailed effective-rank background is retained in Appendix A.
The algorithm matches the existing implementation; experimental records,
hyperparameters, results, and conclusions are unchanged.

The algorithm name is consistently Target A-opt in the manuscript, abstract,
algorithm, tables, and figures. The standard literature term A-optimal design
and the implementation/result identifier `target_a` are retained. The English
abstract has 196 words; the builder and checker enforce the display name.

The method starts on page 3 with normal heading spacing on page 2. The main text
still ends on page 9. The official style, font size, baseline spacing, and margins
are unchanged. The checker also rejects severe underfull-box spacing warnings;
visual inspection remains necessary for smaller warnings and float placement.

## Contents

- `paper/iclr2027/`: English PDF, LaTeX, bibliography, official style, figures,
  tables, and English-only asset builder/checker.
- `code/metrics/`, `code/configs/`, `scripts/`: implementations, frozen protocols,
  data preparation, feature extraction, fitting, diagnostics, and aggregation.
- `tests/`: 95 relevant numerical/protocol/aggregation tests.
- `data/experiment_records.tar.gz`: compressed experimental records,
  sample manifests, candidate definitions, feature metadata, and timing records.
- `data/records_manifest.json`: relative paths, byte counts, and SHA-256 values
  for every archived record.
- `data/anonymization_manifest.json`: metadata-only changes and verification scope.
- `reproduce.py`: integrity verification, safe extraction, and statistical replay.

Chinese manuscripts and translations, third-party paper PDFs, revision plans,
unrelated legacy tools/results, later Target A-opt/MMD complementarity exploration,
Git history, caches, and build logs are excluded. Scientific routines and
experimental values are unchanged. Build/check plumbing and title/abstract
metadata were restricted to English; launcher defaults use portable paths.
Private repository revisions are replaced by consistent anonymous version
labels, with dependent integrity hashes recalculated. These labels are not Git
commits. The public dataset revision and dataset/model fingerprints are retained.
The authors retain the original records separately; they are not in this release.

## Verify and Unpack

From this directory, using Python 3.12:

```bash
python3 reproduce.py verify
python3 reproduce.py unpack
```

Verification and extraction need only the standard library. Extraction restores
the `results/target_conditioned/` paths and distributed record bytes; it refuses
to overwrite a differing existing record. About 408 MB of disk space is needed
for the expanded records. Compression reduces storage/transfer size, not the
underlying evidence. Keep the distributed archive and unpack in a working copy.

## Recompute the Main Analyses

Use an isolated environment, then run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py stats --output recomputed
```

The output directory must not already exist. This replays six analyses from
frozen records, without feature extraction or model fitting:

| Analysis | Archived inputs under `results/target_conditioned/` |
| --- | --- |
| Primary shortlist comparison | `e2b_fixed_cost_shortlist/domainnet_v1/` |
| Projection/readout sensitivity | `revision_20260917/revision_results/e2b_readout_projection_v1/` |
| Final-loss budget curves | Same readout records and `code/configs/target_conditioned_e2b/decision_cost_v1.json` |
| Candidate-construction challenge | `revision_20260922/e2b_decision_cost_v1/` |
| Domain-level matching with random ties | Same construction records |
| Independent quality/cost accounting | Same construction records, timing, and feature profiles |

The earlier PACS/Office-Home selection/oracle records, shared-model and
conditional-shift simulation tables, E0 checks, and sketch-certificate records
are retained with their corresponding scripts. They support the manuscript's
earlier experiments and appendices, not a new claim of empirical superiority.
Historical floating-point timing records needed by provenance checks remain
marked as such; the corrected aggregation script selects the valid measurements.

## Rebuild the English Paper

After unpacking, use the environment above plus TeX Live, `latexmk`, pdfLaTeX,
the standard `algorithm` and `algpseudocode` packages, and Poppler (`pdfinfo`,
`pdftotext`). From `paper/iclr2027/`:

```bash
../../.venv/bin/python build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
../../.venv/bin/python verify_draft.py
```

Keep `.latexmkrc` and the official style files. The checker uses fresh compiler
logs, so compile before running it. No Chinese fonts or Chinese manuscript files
are required. Rebuilding may change PDF timestamps; the statistical tables and
English manuscript text should remain unchanged.

## Run the Tests

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s tests
```

The included tests use temporary synthetic fixtures. They do not require real
dataset images or accelerator access. Some provenance helpers print a no-Git
message; the archive intentionally contains no `.git` directory.

## Re-run Training or Feature Extraction

Raw image datasets, model weights, full feature caches, fitted model NPZ files,
and per-example diagnostic NPZ files were not in the local source snapshot and
are not distributed. Therefore this archive supports offline statistical replay
after dependencies are installed, but is **not** an offline end-to-end training
bundle. Image-level diagnostics and fresh training require regenerating or
providing those external artifacts. Dataset licenses remain applicable.

The frozen data source/version, sample IDs, image hashes, role splits, and
candidate membership are preserved in the archive. Reuse them rather than
resampling after seeing test results. The feature protocol is in
`code/configs/target_conditioned_e2b/domainnet_v1.json`; readout/projection and
decision-cost settings are in the adjacent JSON files.

For CUDA extraction, use the recorded PyTorch 2.5.1+cu124 / torchvision
0.20.1+cu124 environment with `code/requirements-e2b-cuda.txt`. Provide CLIP
`weights/ViT-B-32.pt`, DINOv2 `weights/dinov2_vitb14_pretrain.pth`, a matching local
`dinov2_repository/`, and the required torchvision weight cache under a new work
directory. Set `E2B_WORK_ROOT`, `E2B_PYTHON`, and `SOURCE_GIT_REVISION` explicitly.
For this no-Git release, use the following 40-character package fingerprint for
`SOURCE_GIT_REVISION`, whose legacy name is retained by the preparation scripts:

```bash
export SOURCE_GIT_REVISION="$(python3 -c 'import hashlib; from pathlib import Path; print(hashlib.sha1(Path("PACKAGE_MANIFEST.json").read_bytes()).hexdigest())')"
```

This is a content label, not a Git commit or the revision of the historical runs.
`PACKAGE_MANIFEST.json` records SHA-256 checksums of the distributed files.
Do not attach a private repository hash to results intended for anonymous release.

`scripts/prepare_target_conditioned_e2b_cuda.sh` provides `check`, `restore`,
`features`, and `bundles` stages. Its feature stage uses GPUs 0, 1, and 2; for
fewer devices, invoke `extract_target_conditioned_e2b_features.py --help` and
extract serially. `run_target_conditioned_e2b_isolated.py` enforces the ordered
`screen`, `validate`, and `test-audit` interfaces. The readout, construction,
timing, and feature-cost scripts expose their arguments with `--help`.
Always write new results outside the archived result directories. Other hosts
must set documented paths rather than reuse historical absolute provenance paths.
Cross-device feature extraction is not promised to be bitwise identical.

## Validation of This Reduced Package

- All 1,162 archived records are integrity-checked. Only version labels and
  dependent integrity metadata were adapted; experimental values are unchanged.
- Six statistical analyses completed; all 19 generated CSV tables matched the
  recorded originals byte-for-byte.
- All 95 included tests passed; no test module was skipped in that run.
- English assets regenerated and the manuscript compiled successfully.
  The draft checker passed, with Algorithm 1 on page 3 and the main text ending
  on page 9 (21 total).
- Numerical routines, experimental settings, and English body text match the
  current manuscript. The appendix differs only in three artifact-ID prefixes
  that now identify the anonymous records and the description of the English-only
  abstract builder. Configuration version labels are anonymized; launcher path
  defaults and integrity metadata are adapted for distribution. The recompiled
  PDF text matches the current manuscript after these packaging substitutions.
- Real-data training and accelerator execution were not rerun during packaging.

The development repository remains unchanged. This release changes packaging
and identity-related metadata, not experimental methods or scientific conclusions.
The manuscript's AI Use Statement is retained without changes.
