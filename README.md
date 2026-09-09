# The Collapse Model: Label-Free Data-Pool Compression via Spectral Geometry

Anonymous NeurIPS 2026 submission.

---

## Overview

This repository studies label-free pre-screening of frozen-representation data pools. The
mathematical core has four explicitly separated levels:

1. exact additive Gram identities for a fixed, pool-independent preprocessing protocol;
2. an information-limit result showing that marginal spectra and principal angles do not
   determine the merged spectrum in general;
3. exact local formulas under a stated block-compatibility condition; and
4. a composable rank-*L* PSD Gram sketch with deterministic log-effective-rank error intervals.

The original four-summary Collapse formula is retained only as a theory-inspired empirical
ablation. It is not a generally correct merged-rank formula and is not the primary algorithm.
The primary computational candidate is `rank_l_gram`, which directly scores the sum of transmitted
low-rank PSD factors and records tail bounds and per-step certificates.

## Evidence status

Most older result directories are a historical pilot and audit baseline. They show no stable
advantage of DPP-subspace or Legacy Collapse over rank-only in the natural-pool Stage-2 protocol.
Those historical runs used label-stratified source caches and a confounded global overlap AUROC,
so they must not be cited as validation of the corrected method.

The earlier `corrected_v2` rerun has been invalidated: it mixed a truncated marginal scalar with
the direct Gram sketch and predates the complete encoder/index provenance gate. It must not be
cited. A clean `corrected_v3` rerun uses exact source-local marginal scalars for the legacy
ablation, label-independent subsets, model-state and preprocessing hashes, complete shard
validation, and numerical-tail-aware certificates for `rank_l_gram`.

The four-encoder rank sweep is complete. Across ranks 10, 20, and 50, all 2,880 selected-prefix
intervals contain the exact score, but none of 7,200 evaluated greedy steps is certified; the
bounds are valid but operationally vacuous at these ranks. DPP is at least as competitive as
`rank_l_gram` in this controlled setting. A separate $L=20$ exact-reference pilot covers 288
configurations with Random-100: rank-$L$, DPP, and Full-Gram greedy improve over rank-only by
3.65%, 5.37%, and 5.81%, respectively; rank-$L$ matches none of 288 Full-Gram ordered prefixes
and certifies 0/720 steps. The corrected natural-pool Stage-2 audit also fails its
utility gate: across 28 encoder-target units, the validation-oracle source-supervised adapter is
0.0670 below frozen identity on held-out accuracy (target-cluster 95% CI [-0.0977, -0.0440]). No
method reaches the predeclared 90% top-candidate recall threshold while removing at least half of
the 21 sources. These results support an auditable screening protocol and a negative result, not a
validated practical compressor. The corrections and remaining work are documented in
[`AUDIT_2026-09-08.md`](AUDIT_2026-09-08.md) and tracked in [`PLAN.md`](PLAN.md).

---

## Repository layout

```
paper/                  LaTeX source and compiled PDF
  main.tex              Full paper (Collapse Model + appendices)
  main.pdf              Compiled PDF
  neurips_2026.sty      NeurIPS 2026 style file
  references.bib        Bibliography
  checklist.tex         NeurIPS reproducibility checklist
  figures/              All paper figures (PDF/PNG)

code/                   Reproducibility package
  run_all.py            Orchestrator — runs every experiment end-to-end
  configs/
    repro_config.yaml   Master configuration (paths, seeds, which experiments to run)
  metrics/              Core metric implementations
    reff.py             Effective rank and UID
    subspace_alignment.py  Subspace alignment SA_k
    grassmann.py        Grassmann/principal-angle utilities
    direction_gain.py   Direction-gain helper
  scripts/              One script per paper experiment/table
  results/              Populated by run_all.py (empty at submission time)
  REPRODUCE.md          Full step-by-step reproduction instructions
  requirements.txt      Python dependencies

MATHEMATICAL_FOUNDATION.md  Corrected theorem chain, counterexamples, and claim ledger
IDEA.md                     Two-stage system design
PLAN.md                     Experiment plan and correction checklist
AUDIT_2026-09-08.md         Evidence audit and historical-result limitations
scripts/                    Corrected two-stage screening/adaptation/aggregation scripts
tests/                      Protocol and numerical unit tests
results/                    Versioned historical pilots and explicitly marked corrected reruns
```

---

## Quick start

```bash
# 1. Install dependencies (Python ≥ 3.10 recommended)
pip install -r code/requirements.txt

# 2. Run the protocol and numerical tests
python -m unittest discover -s tests

# 3. Run the original paper pipeline
cd code
python run_all.py

# 4. Run a single original-paper experiment
python run_all.py --only e1a_synthetic

# 5. Dry-run (print plan, no execution)
python run_all.py --dry-run
```

See `code/REPRODUCE.md` for the full step-by-step guide, including how to point the pipeline
at pre-downloaded datasets and how to run on CPU-only machines (all experiments except E1d).
The corrected two-stage pipeline uses locally cached feature archives. `screen` and `summarize`
run on CPU; `adapt` and `heldout` require `torch_npu` and an Ascend host. Source caches used by
Stage-1 must have label-independent sampling metadata unless an explicitly non-strict compatibility
run is requested. Corrected runs follow four ordered phases: label-blind screening,
source-supervised adapter training with target-train cross-validation and hashed checkpoints,
immutable selection-manifest creation, and held-out test evaluation. The `adapt` and `summarize`
phases do not open target-test caches. The audit trained every source adapter offline to estimate
exhaustive shortlist regret; therefore its wall-clock total is not a measured deployment saving.

The corrected workflows use explicit environment variables rather than assuming the paths on the
reported Ascend host:

```bash
# Controlled rank-L sweep and exact-reference pilot (CPU).
PROJECT_ROOT="$PWD" FEATURE_DIR=/path/to/source/features RESULT_ROOT=/path/to/output \
  bash scripts/run_two_stage_controlled_rank_sweep.sh
PROJECT_ROOT="$PWD" FEATURE_DIR=/path/to/source/features RESULT_ROOT=/path/to/output \
  bash scripts/run_two_stage_controlled_exact_pilot.sh

# Natural-pool screening (CPU), then source-supervised validation and held-out evaluation (Ascend).
PROJECT_ROOT="$PWD" SOURCE_DIR=/path/to/source/features RESULT_ROOT=/path/to/screening \
  bash scripts/run_two_stage_corrected_screen.sh resnet50 vit_b16 clip_b32 dinov2_b14
PROJECT_ROOT="$PWD" SOURCE_DIR=/path/to/source/features TARGET_DIR=/path/to/target/features \
  BASE_RESULTS=/path/to/screening RESULT_ROOT=/path/to/stage2 \
  bash scripts/run_two_stage_repeated_cv_utility_npu.sh 4 resnet50

# Run after all four encoder directories are complete.
python3 scripts/summarize_two_stage_cross_encoder.py \
  --results-dir /path/to/stage2 \
  --expected-encoders resnet50 vit_b16 clip_b32 dinov2_b14
```

The committed corrected outputs and their interpretation are under
`results/two_stage_controlled_rank_sweep_corrected_v3/`,
`results/two_stage_controlled_exact_corrected_v3/`,
`results/two_stage_natural_shortlist_corrected_v3/`, and
`results/two_stage_repeated_cv_utility_corrected_v3/`.

---

## Datasets

The original paper pipeline uses the following sources:

| Split | Sources |
|-------|---------|
| 14 vision datasets | torchvision (CIFAR-10/100, STL-10, SVHN, MNIST, Fashion-MNIST, Oxford Pets, Stanford Cars, Food-101, EuroSAT, RESISC-45, DTD, DomainNet, CelebA) + MedMNIST (Camelyon17) |
| 19 SBERT text corpora | HuggingFace Datasets |

The corrected two-stage experiments use auditable local Arrow/cache inputs and record their
SHA-256 hashes. Four frozen visual encoders are evaluated: ResNet-50, ViT-B/16, CLIP ViT-B/32,
and DINOv2. The current natural-pool pilot is domain-imbalanced; broader independent medical,
character, and general-vision candidate/target collections remain required work.

---

## Reproducibility

Original-paper table mappings remain in `code/configs/repro_config.yaml`. Corrected two-stage
runs additionally bind every adaptation CSV to the screening manifest, script, configuration,
encoder state, preprocessing, sampled indices, source/target-train feature hashes, and generated
adapter checkpoints. Resume is accepted only for complete source/seed groups with the same run
fingerprint. Held-out evaluation is accepted only when its input selection manifest and unchanged
validation-only detail file pass their SHA-256 checks; target-test hashes first appear in the
held-out run sidecar.

Exact top-1 recall remains the historical primary metric. Practical tolerance curves and a
paired-bootstrap candidate confidence-set recall are reported alongside it rather than replacing
it post hoc.

---

## Paper

The compiled PDF is at `paper/main.pdf`. The LaTeX source can be recompiled with:

```bash
cd paper
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```
