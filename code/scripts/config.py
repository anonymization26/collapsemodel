"""
Collapse Model: predicting effective rank of merged feature matrices
Global experiment configuration

Core question: how does r_eff change when two feature matrices H_A and H_B are merged into [H_A; H_B]?
Core answer:   Collapse Model closed-form formula, based on eigenvalue analysis of 2x2 Gram blocks.
Three-channel decomposition: gamma (Sigma energy ratio) x alpha (V directional alignment SA_k) x UID (U sample geometry)
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
# repro/scripts/config.py -> ROOT = repro/  (reproducibility package root)
# Can be overridden via environment variables COLLAPSE_REPRO_ROOT / COLLAPSE_DATA_DIR
import os
ROOT       = Path(os.environ.get(
    "COLLAPSE_REPRO_ROOT",
    Path(__file__).resolve().parent.parent
)).resolve()
DATA_DIR   = Path(os.environ.get("COLLAPSE_DATA_DIR", "/home/data")).resolve()
RESULT_DIR = ROOT / "results"
FIG_DIR    = ROOT / "figures"

for d in [RESULT_DIR, FIG_DIR]:
    d.mkdir(exist_ok=True)
try:
    DATA_DIR.mkdir(exist_ok=True)
except OSError:
    pass  # /home/data may not be creatable in local environments

# ── General parameters ────────────────────────────────────────
BATCH_SIZE  = 256
N_SAMPLES   = 2000    # feature samples per dataset (use 500 for mini experiments)
SEED        = 42

# ── Subspace dimension k (used for SA_k computation) ─────────
K_DEFAULT   = 20
K_VALUES    = (10, 20, 50)   # multi-scale ablation

# ── Quick debug mode ──────────────────────────────────────────
DEBUG_MODE  = False
if DEBUG_MODE:
    N_SAMPLES = 300

# ═══════════════════════════════════════════════════════════════
# E1: Collapse Model validation (synthetic + real dataset pairs)
#
# Closed-form formula: r_hat = r_dom^{r*} * r_sub^{1-r*} / (r*^{r*} * (1-r*)^{1-r*})
# where r* is determined by the 2x2 Gram block discriminant D = sqrt((1-gamma^2)^2 + 4*gamma^2*alpha)
#
# Three behavior regimes:
#   (1) Superadditive (small alpha, balanced gamma): r_merged > max(r_A, r_B)
#   (2) Direction collapse (alpha -> 1): subspace overlap causes redundancy
#   (3) Energy collapse (gamma >> 1 or gamma << 1): one side dominates
# ═══════════════════════════════════════════════════════════════

# Synthetic experiment grid
# alpha densified near 0 -> precisely captures the superadditivity boundary
# gamma covers energy collapse threshold (gamma* ~= 3.0)
E1_ALPHA_GRID  = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
E1_GAMMA_GRID  = [0.1, 0.33, 0.5, 1.0, 2.0, 3.0, 10.0]
E1_SYNTH_D     = 512   # feature dimensionality
E1_SYNTH_K     = 50    # subspace dimension (determines r_eff magnitude)
E1_SYNTH_N     = 1000  # samples per matrix
E1_N_REPEAT    = 5     # repetitions per (alpha, gamma) point

# Real dataset pairs (covering all three behavior regimes)
E1_REAL_PAIRS = [
    ("cifar10",  "svhn"),    # expected direction collapse (same small-image domain, high SA_k)
    ("cifar10",  "stl10"),   # expected direction collapse (STL-10 is a hi-res version of CIFAR-10)
    ("svhn",     "stl10"),   # expected superadditive (street digits vs objects, large domain gap)
]
E1_PROBES = ["resnet50", "vit_b16"]

# ═══════════════════════════════════════════════════════════════
# E_downstream: correlation validation between r_eff and downstream LP accuracy
# 10 probes x 5 datasets, per-dataset Spearman rho
# ═══════════════════════════════════════════════════════════════

E2_PROBES_MINI = ["resnet50", "vit_b16"]
E2_PROBES_FULL = [
    # CNN family
    "resnet50", "resnet101", "efficientnet_b4", "convnext_tiny", "densenet121",
    # ViT family
    "vit_b16", "vit_l16", "deit_b", "swin_tiny", "beit_base",
    # Hybrid architectures
    "regnet_y", "maxvit_tiny",
]

E2_TARGET_DATASETS_MINI = ["cifar10", "stl10", "svhn"]
E2_TARGET_DATASETS_FULL = [
    "cifar10", "cifar100", "stl10", "svhn",
    "mnist", "fashion_mnist", "kmnist",
    "eurosat", "oxford_pets", "flowers102", "food101", "dtd",
]

# Linear probe training configuration
E2_LP_LR          = 1e-3
E2_LP_EPOCHS      = 50
E2_LP_BATCH_SIZE  = 256

# ═══════════════════════════════════════════════════════════════
# E3: Delta_dir dataset selection (directional-gain criterion Delta_dir = 1 - SA_k)
# ═══════════════════════════════════════════════════════════════

E3_BASE_DATASET     = "cifar10"
E3_CANDIDATE_POOL   = [
    "stl10", "svhn", "mnist", "fashion_mnist",
    "kmnist", "eurosat", "cifar100",
]
E3_PROBE_FOR_DELTA  = "resnet50"
E3_N_ROUNDS         = len(E3_CANDIDATE_POOL)

# ═══════════════════════════════════════════════════════════════
# E4: Cross-probe consistency validation
# Verifies that Collapse Model predictions are consistent across different encoder architectures
# ═══════════════════════════════════════════════════════════════

E4_SOURCE_DOMAIN    = "real"
E4_TARGET_DOMAINS   = ["clipart", "infograph", "painting", "quickdraw", "sketch"]
E4_LAMBDA_VALUES    = [0.01, 0.1, 1.0]
E4_EPOCHS           = 30
E4_K_ALIGN          = 50

# ═══════════════════════════════════════════════════════════════
# E5: Pool selection experiment (Collapse Model Greedy)
# M=10 candidate datasets, budget K in {1,2,3,5}
# Comparison: exhaustive search, Collapse Greedy, Max-reff, Delta_dir Greedy, Random
# ═══════════════════════════════════════════════════════════════

PROBES_MINI = ["resnet50"]
PROBES_FULL = E2_PROBES_FULL

# ── Dataset semantic domain labels (used for visualization grouping) ──
DATASET_DOMAIN = {
    "mnist":         "handwritten",
    "kmnist":        "handwritten",
    "fashion_mnist": "handwritten",
    "cifar10":       "natural",
    "cifar100":      "natural",
    "stl10":         "natural",
    "svhn":          "digit_scene",
    "eurosat":       "satellite",
    "oxford_pets":   "finegrained",
    "flowers102":    "finegrained",
    "food101":       "finegrained",
    "dtd":           "texture",
    "bloodmnist":    "medical",
    "dermamnist":    "medical",
    "pathmnist":     "medical",
}

DOMAIN_COLORS = {
    "handwritten":  "#4C72B0",
    "natural":      "#55A868",
    "digit_scene":  "#8172B2",
    "satellite":    "#64B5CD",
    "finegrained":  "#937860",
    "texture":      "#C44E52",
    "medical":      "#DD8452",
}
