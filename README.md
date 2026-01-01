# The Collapse Model: Label-Free Data-Pool Compression via Spectral Geometry

Anonymous NeurIPS 2026 submission.

---

## Overview

This repository contains the paper and reproducibility code for **The Collapse Model**,
a closed-form method that predicts the merged effective rank of two feature-matrix pools
from three per-dataset summary statistics (energy ratio γ, subspace alignment α, intrinsic
dimension UID) without requiring label access or cross-dataset feature sharing.

**Key results:**
- Pearson *r* = 0.97 on 427 real feature-matrix pairs (5 encoder families, 2 modalities)
- R² = 0.95 with calibration; superadditivity-threshold diagnosis distinguishes the model from *r*_A + *r*_B
- Greedy compression retains > 100 % of full-pool effective rank with ≤ 1 % regret vs. exhaustive search
- Scales to *M* = 43 pools at 14× compression; outperforms Facility Location / *k*-Medoids by +7–+14 pp on downstream linear-probe accuracy

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
```

---

## Quick start

```bash
# 1. Install dependencies (Python ≥ 3.10 recommended)
pip install -r code/requirements.txt

# 2. Run all experiments (≈ 8 h on one GPU node)
cd code
python run_all.py

# 3. Run a single experiment
python run_all.py --only e1a_synthetic

# 4. Dry-run (print plan, no execution)
python run_all.py --dry-run
```

See `code/REPRODUCE.md` for the full step-by-step guide, including how to point the pipeline
at pre-downloaded datasets and how to run on CPU-only machines (all experiments except E1d).

---

## Datasets

All datasets are downloaded automatically by the scripts on first run:

| Split | Sources |
|-------|---------|
| 14 vision datasets | torchvision (CIFAR-10/100, STL-10, SVHN, MNIST, Fashion-MNIST, Oxford Pets, Stanford Cars, Food-101, EuroSAT, RESISC-45, DTD, DomainNet, CelebA) + MedMNIST (Camelyon17) |
| 19 SBERT text corpora | HuggingFace Datasets |

Pre-trained encoder weights (ResNet-50, ViT-B/16, CLIP ViT-B/32, DINO ViT-S/16, SBERT)
are downloaded automatically by `timm`, `open_clip_torch`, and `sentence-transformers`.

---

## Reproducibility

Every number reported in the paper is produced by one of the scripts in `code/scripts/`.
The mapping from script → paper table/figure is listed in `code/configs/repro_config.yaml`
under each experiment's `description` field.

Bootstrap confidence intervals use *B* = 10 000 resamples (random seed 42).
All results are written as CSV/JSON to `code/results/` and can be verified against the
numbers in `paper/main.tex`.

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
