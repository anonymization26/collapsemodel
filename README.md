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
The deployable candidate is `rank_l_gram`, which directly scores the sum of transmitted
low-rank PSD factors and records tail bounds and per-step certificates.

## Evidence status

The committed result directories are a historical pilot and audit baseline. They show no
stable advantage of DPP-subspace or Legacy Collapse over rank-only in the current natural-pool
Stage-2 protocol. They did not evaluate the corrected `rank_l_gram` implementation, used
label-stratified source caches, and used a confounded global overlap AUROC. Therefore they must
not be cited as validation of the corrected method.

The corrections, affected claims, and required reruns are documented in
[`AUDIT_2026-09-08.md`](AUDIT_2026-09-08.md) and tracked in [`PLAN.md`](PLAN.md). Until those
reruns finish, the empirical status of the primary rank-*L* method is **not yet established**.

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
results/                    Versioned historical pilot outputs (not corrected reruns)
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
run on CPU; only `adapt` requires `torch_npu` and an Ascend host. Source caches used by Stage-1
must have label-independent sampling metadata unless an explicitly non-strict compatibility run
is requested.

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
and all source/target feature hashes. Resume is accepted only for complete source/seed groups
with the same run fingerprint.

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
