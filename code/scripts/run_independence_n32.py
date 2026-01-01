"""
Three-channel independence test -- extended to n=32 (EU7 multi-target data)
============================================================================
Previously tested three-channel independence with eu6b (n=10 STL-10 sources, p=0.069).
Now using eu7_expanded_v2_raw.csv (n=32 pairs, 3 targets x 6 sources x ~probe)
for more reliable statistics.

Output:
  results/eu_independence_n32.csv  -- pairwise Spearman rho and p-values
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"

# ── Load n=32 multi-target data ────────────────────────────────────────
df = pd.read_csv(RESULTS / "eu7_expanded_v2_raw.csv")

# Remove self-loops (source == target; SA_k=1 is a degenerate case)
df = df[df["source"] != df["target"]].copy()

# Both probes available -> concat, then drop duplicate (probe, source, target) entries to avoid
# artificially inflating n; conservatively keep only the ResNet50 row per (source, target).
# If only ResNet50 rows exist, use ResNet50.
probes = df["probe"].unique()
print(f"Probes available: {probes}")
print(f"Total rows (excl. self): {len(df)}")

# Prefer ResNet50 for consistency
df_r50  = df[df["probe"] == "resnet50"].copy()
df_vit  = df[df["probe"] == "vit_b16"].copy()

# Row counts
print(f"  ResNet50 rows: {len(df_r50)}")
print(f"  ViT-B16  rows: {len(df_vit)}")

# Union of both probes (using the mean when 2 rows per (source,target) is more robust,
# but the primary report uses a single probe)
# Primary analysis: use ResNet50 (consistent with E1/E4)
main_df = df_r50.copy()
n = len(main_df)
print(f"\nPrimary analysis using ResNet50, n = {n}")

# ── Three-channel values ─────────────────────────────────────────────
sa   = main_df["sa_k"].values
stg  = main_df["stg"].values
uid  = main_df["uid_mm"].values     # |UID_src - UID_tgt|

# ── Pairwise correlations ────────────────────────────────────────────────
pairs = [
    ("SA_k",    "STG",    sa,  stg),
    ("SA_k",    "UID_mm", sa,  uid),
    ("STG",     "UID_mm", stg, uid),
]

rows = []
print(f"\n{'Channel pair':20s}  {'Spearman rho':>12s}  {'p':>8s}  {'Pearson r':>10s}  {'p':>8s}  {'Independent?'}")
print("-" * 78)
for name_a, name_b, x, y in pairs:
    sp_r, sp_p = spearmanr(x, y)
    pe_r, pe_p = pearsonr(x, y)
    independent = (sp_p > 0.05)
    flag = "yes" if independent else "no"
    print(f"  {name_a:8s} vs {name_b:8s}  "
          f"{sp_r:>+12.4f}  {sp_p:>8.4f}  "
          f"{pe_r:>+10.4f}  {pe_p:>8.4f}  {flag}")
    rows.append({
        "channel_a": name_a, "channel_b": name_b,
        "n": n, "probe": "resnet50",
        "spearman_rho": round(sp_r, 4), "spearman_p": round(sp_p, 5),
        "pearson_r":    round(pe_r, 4), "pearson_p":   round(pe_p, 5),
        "independent":  independent,
    })

# ── ViT-B16 cross-check ──────────────────────────────────────────────
print(f"\nViT-B16 validation (n={len(df_vit)}):")
for name_a, name_b, _, _ in pairs:
    x2 = df_vit[{"SA_k": "sa_k", "STG": "stg", "UID_mm": "uid_mm"}[name_a]].values
    y2 = df_vit[{"SA_k": "sa_k", "STG": "stg", "UID_mm": "uid_mm"}[name_b]].values
    sp_r2, sp_p2 = spearmanr(x2, y2)
    print(f"  {name_a:8s} vs {name_b:8s}  rho={sp_r2:+.4f}  p={sp_p2:.4f}  "
          f"{'independent' if sp_p2 > 0.05 else 'correlated'}")
    rows.append({
        "channel_a": name_a, "channel_b": name_b,
        "n": len(df_vit), "probe": "vit_b16",
        "spearman_rho": round(sp_r2, 4), "spearman_p": round(sp_p2, 5),
        "pearson_r": None, "pearson_p": None,
        "independent": sp_p2 > 0.05,
    })

# ── Save ──────────────────────────────────────────────────────────
out = RESULTS / "eu_independence_n32.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nResults saved: {out}")

# ── Summary conclusions ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Conclusions summary (ResNet50, n=32):")
for row in rows[:3]:
    status = "independent" if row["independent"] else "correlated"
    print(f"  {row['channel_a']:8s} vs {row['channel_b']:8s}: "
          f"rho={row['spearman_rho']:+.4f}, p={row['spearman_p']:.4f}  -> {status}")
print("=" * 60)
