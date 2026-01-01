# Reproducing the Collapse Model paper

This package contains everything a NeurIPS reviewer needs to reproduce
**every table and figure in the paper**.

```
repro/
├── REPRODUCE.md        ← you are here
├── requirements.txt    ← Python dependencies (pinned)
├── run_all.py          ← orchestrator (driver)
├── configs/
│   └── repro_config.yaml      ← which experiments to run, paths, seeds
├── scripts/            ← all paper experiment scripts (self-contained)
├── metrics/            ← core measurement library (SA_k, r_eff, …)
├── data_cache/         ← (created on first run) features & raw data
├── results/            ← (created on first run) CSV/JSON outputs
└── figures/            ← (created on first run) regenerated figures
```

---

## TL;DR — fastest path

```bash
# 1. Install (one-time)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run all paper experiments
python run_all.py
```

End-to-end runtime: **≈8 hours** on a single GPU (A100 / 4090 / 3090).
Each step writes a CSV/JSON under `results/` and a per-step log under
`results/_logs/<step>.log`.

---

## The 4 most paper-relevant tables (under 2 hours total)

If full reproduction is too long, a reviewer can verify the headline
claims with these four:

| Paper table | Script | Time | What it shows |
|---|---|---|---|
| Tab 1 (E1a synthetic) | `experiment_e1_theorem.py` | 2 min | Closed-form formula is exact under derivation assumptions ($R^2{=}0.9994$) |
| Tab 5 (M=43 scale) | `run_e8_scale_m50.py` + `run_e9_fair_lp.py` | 90 min | Compression retains $>112\%$ of full-pool $\reff$ at $14\times$, beats Random $95\%$ CI on LP |
| Tab 6 (modern baselines) | `run_p1a_modern_baselines.py` | 90 min | Vendi/DomainDiverse/Centroid-FF comparison; two-signal decomposition |
| Tab p2e_hybrid | `run_p2e_hybrid.py` | 90 min | Cen→Col strictly dominates pure Collapse on both LP and retention at $k{=}10$ |

Run only these:

```bash
python run_all.py --only e1a_synthetic,feature_extraction_m43,e8_scale_m43,e9_fair_lp,p1a_modern_baselines,p2e_hybrid
```

---

## What runs by default

`configs/repro_config.yaml` enables 15 experiments (see the file for the
full list). Each one maps to one or more paper tables/figures. Disable
any by setting `enabled: false` or skipping at the CLI:

```bash
python run_all.py --skip e1d_clip_dino,p1a_modern_baselines
python run_all.py --only e1a_synthetic,ablation_256
```

Use `--dry-run` to see the plan without running:

```bash
python run_all.py --dry-run
```

---

## Hardware and runtime expectations

The paper's experiments fall into three classes:

| Class | Examples | Hardware | Runtime |
|---|---|---|---|
| **Closed-form / pure linear-algebra** | E1a synthetic, channel independence, ablation_256 | CPU, ≤8 GB RAM | seconds–minutes |
| **Real-data validation, no LP** | E1b/E1c/E1d, bimodal angle table | CPU + 1 GPU for feature extraction | 15–60 min |
| **LP-bound compression** | M=43 fair LP, P1-A modern baselines, P2-E hybrid | 1 GPU recommended | 60–90 min each |

A reviewer with a single mid-range GPU should expect ≈8 h end-to-end.
On CPU only, the LP steps will run but multiply by ~3–5×.

The `compute.device` field in the YAML accepts `auto | cuda | cpu` and
is forwarded to the scripts as the env var `COLLAPSE_DEVICE`.

---

## Datasets

All datasets used by the paper are publicly available:

* **Vision** (CIFAR-10/100, STL-10, SVHN, MNIST, Fashion-MNIST,
  BloodMNIST, DermaMNIST, PathMNIST): downloaded automatically via
  `torchvision` and `medmnist` on first run.
* **ImageNet semantic subsets** (animal / vehicle / artifact / food /
  scene): require ImageNet-1K validation images; reviewers should set
  `COLLAPSE_DATA_DIR=/path/to/imagenet` before running. Subset
  construction (WordNet semantic hierarchy) is in
  `scripts/extract_features_m50.py`.
* **Text corpora** (AG News, 20 Newsgroups, DBpedia, IMDB, Yelp): pulled
  from HuggingFace `datasets` automatically.

If a dataset cannot be auto-downloaded, the script will exit with a
clear error message naming the missing path.

---

## Environment overrides

Three environment variables are honoured by every script:

| Variable | Default | Purpose |
|---|---|---|
| `COLLAPSE_REPRO_ROOT` | `<repro/>` | All relative paths resolve from here. Set this to put outputs elsewhere. |
| `COLLAPSE_DATA_DIR`   | `<repro>/data_cache` | Where datasets are downloaded to / loaded from. |
| `RANDOM_SEED`        | `42` (from YAML) | Master RNG seed for all stochastic steps. |

```bash
export COLLAPSE_DATA_DIR=/scratch/imagenet/val
export COLLAPSE_REPRO_ROOT=/scratch/collapse-repro
python run_all.py
```

---

## Verifying outputs against paper numbers

After all experiments complete, run the verification script:

```bash
python scripts/verify_claims.py
```

It loads each CSV under `results/`, computes the headline numbers
quoted in the paper (e.g. `Pearson r = 0.97 on 427 pairs`,
`retention = 117.4% at k=7 on M=43`), and prints **PASS** / **FAIL**
against expected values within a 1% tolerance.

A reviewer should see:

```
PASS  E1a synthetic         R² = 0.9994 (expected 0.9994 ± 0.001)
PASS  E1d 427 pairs         Pearson r = 0.970 (expected 0.97 ± 0.01)
PASS  M=43 scale k=3        retention = 112.4% (expected 112.4 ± 1.0)
PASS  M=43 fair LP k=3      Collapse 0.7550 vs Rand 95%CI [0.7446,0.7508]
PASS  P1-A modern baseline  k=5 DomainDiverse LP = 0.7634 (expected 0.7634)
PASS  P2-E Cen→Col k=10     LP = 0.7602 (expected 0.7602 ± 0.003)
…
```

Any FAIL prints the observed-vs-expected delta and the path to the
underlying CSV; please report these as reviewer comments rather than
re-running the experiment.

---

## Troubleshooting

* **`ModuleNotFoundError: No module named 'config'`** — orchestrator
  prepends `scripts/` to `PYTHONPATH`. If running a script directly,
  do so from `repro/scripts/` or add it to your `PYTHONPATH`.
* **CUDA OOM during E1d** — set `compute.device: cpu` in the YAML and
  expect ~3× slower extraction. The closed-form formula itself is
  CPU-only.
* **Dataset download blocked** — pre-stage datasets under
  `COLLAPSE_DATA_DIR` and set `data_dir` in the YAML to that path; all
  scripts skip the download if files already exist.
* **A single experiment failed** — `run_all.py` continues past failures
  and reports them in the summary; re-run only that one with
  `python run_all.py --only <name>`. The full log is at
  `results/_logs/<name>.log`.

---

## Citing the paper

If you use any of this code, please cite:

```bibtex
@inproceedings{collapse2026,
  title  = {The Collapse Model: Label-Free Data-Pool Compression via Spectral Geometry},
  author = {Anonymous},
  booktitle = {NeurIPS 2026},
  year   = {2026},
}
```
