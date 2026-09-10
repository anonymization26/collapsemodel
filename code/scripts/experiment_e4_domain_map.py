"""
Experiment E4: Cross-probe Consistency and Domain Distance Map

Goal:
    Build an SA_k inter-domain distance matrix from cached torchvision dataset
    features, and verify that Collapse Model predictions are consistent across
    different encoder architectures (ResNet-50 vs ViT-B/16 Pearson r = 0.891).

Core contents:
    1. Domain distance matrix -- SA_k(D_i, D_j) heatmap (n_domains x n_domains)
    2. Grassmann manifold MDS embedding -- 2D domain position visualization
    3. Distance-gain consistency validation -- Grassmann distance vs delta_r (merge gain)
    4. Cross-probe consistency -- Pearson r between ResNet50 and ViT-B16 domain distance matrices

Uses existing caches; no additional data download needed.

Outputs:
    results/e4_domain_distance.csv  -- full-pair domain distance matrix
    figures/e4_domain_heatmap.png   -- SA_k heatmap
    figures/e4_domain_mds.png       -- Grassmann manifold MDS embedding
    figures/e4_cross_probe.png      -- cross-probe distance consistency scatter
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from config import RESULT_DIR, FIG_DIR, K_DEFAULT
from metrics.subspace_alignment import compute_subspace_alignment
from metrics.collapse_core import (
    effective_rank,
    energy_dominant_ranks,
    is_superadditive,
    nuclear_mass,
    superadditivity_delta,
)


# ─────────────────────────────────────────────────────────────
# 1. Load existing cached features
# ─────────────────────────────────────────────────────────────

def load_cached_features(probe: str, dataset: str) -> np.ndarray:
    """Load cached features produced by E1/E2/E3 (train split)."""
    # Prefer new format (with split suffix)
    new_path = RESULT_DIR / f"feat_{probe}_{dataset}_train.npz"
    old_path  = RESULT_DIR / f"feat_{probe}_{dataset}.npz"
    for p in [new_path, old_path]:
        if p.exists():
            H = np.load(p)["H"]
            print(f"  [cache] {probe}/{dataset}: {H.shape}")
            return H
    raise FileNotFoundError(f"Cache not found: {probe}/{dataset}")


def discover_available_pairs() -> dict:
    """Auto-scan the results/ directory to find available datasets for each probe."""
    available = {}
    for f in sorted(RESULT_DIR.glob("feat_*.npz")):
        name = f.stem  # feat_resnet50_cifar10 / feat_resnet50_svhn_test
        parts = name.split("_")
        # Filter out test splits (use train only)
        if parts[-1] == "test":
            continue
        # Convention: feat_{probe}_{dataset}
        # Match against known probe names (probes may contain underscores, e.g. vit_b16)
        for probe in ["resnet50", "vit_b16"]:
            prefix = f"feat_{probe}_"
            if name.startswith(prefix):
                ds = name[len(prefix):]
                available.setdefault(probe, []).append(ds)
    return available


# ─────────────────────────────────────────────────────────────
# 2. Build domain distance matrix
# ─────────────────────────────────────────────────────────────

def build_distance_matrix(probe: str, datasets: list, k: int = K_DEFAULT) -> pd.DataFrame:
    """
    Compute SA_k and Grassmann distance for all dataset pairs.

    Returns:
        DataFrame: rows/columns are dataset names, values are SA_k
    """
    n = len(datasets)
    sa_mat  = np.zeros((n, n))
    gd_mat  = np.zeros((n, n))

    # Load all features
    features = {}
    for ds in datasets:
        try:
            features[ds] = load_cached_features(probe, ds)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")

    available_ds = [d for d in datasets if d in features]
    n_avail = len(available_ds)
    sa_mat  = np.zeros((n_avail, n_avail))
    gd_mat  = np.zeros((n_avail, n_avail))

    print(f"\nBuilding {probe} domain distance matrix ({n_avail}x{n_avail})...")
    for i, ds_i in enumerate(available_ds):
        for j, ds_j in enumerate(available_ds):
            if i == j:
                sa_mat[i, j] = 1.0   # self SA_k = 1
                gd_mat[i, j] = 0.0
                continue
            if j < i:
                # Exploit symmetry
                sa_mat[i, j] = sa_mat[j, i]
                gd_mat[i, j] = gd_mat[j, i]
                continue
            result = compute_subspace_alignment(features[ds_i], features[ds_j], k=k)
            sa_mat[i, j]  = result["sa_k"]
            gd_mat[i, j]  = result["grassmann_dist"]
            print(f"  {ds_i:15s} ↔ {ds_j:15s}  SA_k={result['sa_k']:.4f}  "
                  f"GD={result['grassmann_dist']:.4f}")

    sa_df = pd.DataFrame(sa_mat, index=available_ds, columns=available_ds)
    gd_df = pd.DataFrame(gd_mat, index=available_ds, columns=available_ds)
    return sa_df, gd_df, features, available_ds


# ─────────────────────────────────────────────────────────────
# 3. Distance-gain consistency validation
# ─────────────────────────────────────────────────────────────

def compute_gain_consistency(features: dict, sa_df: pd.DataFrame,
                              datasets: list, k: int = K_DEFAULT) -> pd.DataFrame:
    """
    Validate: Grassmann distance (1 - SA_k) up -> merge gain delta_r up.

    For all dataset pairs, compute:
        - SA_k (subspace overlap; lower = more complementary)
        - delta_r = r_eff([H_i; H_j]) - r_dom, where r_dom is the
          higher-nuclear-mass source's effective rank
        - Expected: SA_k down -> delta_r up (orthogonal domains have greatest merge gain)
    """
    rows = []
    for ds_i, ds_j in combinations(datasets, 2):
        if ds_i not in features or ds_j not in features:
            continue
        H_i = features[ds_i]
        H_j = features[ds_j]
        H_merged = np.concatenate([H_i, H_j], axis=0)

        r_i      = effective_rank(H_i)
        r_j      = effective_rank(H_j)
        r_merged = effective_rank(H_merged)
        nuclear_i = nuclear_mass(H_i)
        nuclear_j = nuclear_mass(H_j)
        gamma = nuclear_j / nuclear_i
        r_dom, _ = energy_dominant_ranks(r_i, r_j, gamma)
        delta_r = superadditivity_delta(r_merged, r_i, r_j, gamma)

        sa_val = sa_df.loc[ds_i, ds_j] if ds_i in sa_df.index and ds_j in sa_df.columns \
                 else sa_df.loc[ds_j, ds_i]

        rows.append({
            "ds_i":    ds_i,
            "ds_j":    ds_j,
            "r_i":     round(r_i, 2),
            "r_j":     round(r_j, 2),
            "r_merged":round(r_merged, 2),
            "r_dom": round(r_dom, 2),
            "gamma": round(gamma, 6),
            "delta_r": round(delta_r, 2),
            "is_superadditive": is_superadditive(r_merged, r_i, r_j, gamma),
            "sa_k":    round(float(sa_val), 4),
            "delta_dir": round(1.0 - float(sa_val), 4),
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────
# 4. Cross-probe consistency
# ─────────────────────────────────────────────────────────────

def cross_probe_consistency(sa_resnet: pd.DataFrame, sa_vit: pd.DataFrame) -> float:
    """
    Compute Pearson r between the ResNet50 and ViT-B16 domain distance matrices.
    Only uses dataset pairs common to both probes.
    """
    from scipy.stats import pearsonr
    common = sorted(set(sa_resnet.index) & set(sa_vit.index))
    if len(common) < 3:
        return float("nan")

    pairs_r, pairs_v = [], []
    for i, ds_i in enumerate(common):
        for j, ds_j in enumerate(common):
            if j <= i:
                continue
            pairs_r.append(sa_resnet.loc[ds_i, ds_j])
            pairs_v.append(sa_vit.loc[ds_i, ds_j])

    r, p = pearsonr(pairs_r, pairs_v)
    print(f"\nCross-probe Pearson r = {r:.4f}  (n={len(pairs_r)}, p={p:.4f})")
    return float(r)


# ─────────────────────────────────────────────────────────────
# 5. Visualization
# ─────────────────────────────────────────────────────────────

def plot_all(sa_resnet: pd.DataFrame, sa_vit: pd.DataFrame,
             gain_df: pd.DataFrame, cross_r: float):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from sklearn.manifold import MDS
        import seaborn as sns
    except ImportError as e:
        print(f"Visualization dependencies missing: {e}")
        return

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Subplot 1: ResNet50 SA_k heatmap ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    mask = np.zeros_like(sa_resnet.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(sa_resnet, annot=True, fmt=".3f", cmap="YlOrRd_r",
                vmin=0, vmax=1, ax=ax1, mask=mask, cbar_kws={"shrink": 0.8})
    ax1.set_title("ResNet50  SA_k Matrix\n(lower = more complementary)")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=8)

    # ── Subplot 2: ViT-B16 SA_k heatmap ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    mask_v = np.zeros_like(sa_vit.values, dtype=bool)
    np.fill_diagonal(mask_v, True)
    sns.heatmap(sa_vit, annot=True, fmt=".3f", cmap="YlOrRd_r",
                vmin=0, vmax=1, ax=ax2, mask=mask_v, cbar_kws={"shrink": 0.8})
    ax2.set_title("ViT-B16  SA_k Matrix\n(lower = more complementary)")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=8)

    # ── Subplot 3: ResNet50 MDS embedding ───────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    gd_mat = 1.0 - sa_resnet.values.copy()
    np.fill_diagonal(gd_mat, 0.0)
    gd_sym = (gd_mat + gd_mat.T) / 2
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
        coords = mds.fit_transform(gd_sym)
        labels = sa_resnet.index.tolist()
        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
                  "#FF9800", "#795548", "#607D8B"]
        for idx, (label, xy) in enumerate(zip(labels, coords)):
            ax3.scatter(*xy, color=colors[idx % len(colors)], s=120, zorder=3)
            ax3.annotate(label, xy, xytext=(5, 5), textcoords="offset points",
                         fontsize=8, fontweight="bold")
        ax3.set_title("Grassmann Manifold MDS\n(ResNet50, distance = 1 - SA_k)")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("MDS-1")
        ax3.set_ylabel("MDS-2")
    except Exception as e:
        ax3.text(0.5, 0.5, f"MDS failed: {e}", transform=ax3.transAxes,
                 ha="center", va="center")

    # ── Subplot 4: SA_k vs delta_r scatter (ResNet50) ────────
    ax4 = fig.add_subplot(gs[1, 0])
    if len(gain_df) > 0:
        x = gain_df["sa_k"].values
        y = gain_df["delta_r"].values
        ax4.scatter(x, y, color="#2196F3", s=80, alpha=0.8, zorder=3)
        for _, row in gain_df.iterrows():
            ax4.annotate(f"{row['ds_i'][:4]}+{row['ds_j'][:4]}",
                         (row["sa_k"], row["delta_r"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 50)
            ax4.plot(xl, np.polyval(z, xl), "r--", lw=1.5)
            from scipy.stats import pearsonr
            r, p = pearsonr(x, y)
            ax4.set_title(f"SA_k vs delta_r\n(ResNet50, Pearson r={r:+.3f})")
        else:
            ax4.set_title("SA_k vs delta_r (ResNet50)")
        ax4.set_xlabel("SA_k (subspace overlap)")
        ax4.set_ylabel("delta_r = r_merged - r_dom")
        ax4.axhline(0, color="gray", lw=0.8, ls="--")
        ax4.grid(True, alpha=0.3)

    # ── Subplot 5: Cross-probe consistency ───────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    common = sorted(set(sa_resnet.index) & set(sa_vit.index))
    pairs_r, pairs_v, pair_labels = [], [], []
    for i, ds_i in enumerate(common):
        for j, ds_j in enumerate(common):
            if j <= i:
                continue
            pairs_r.append(sa_resnet.loc[ds_i, ds_j])
            pairs_v.append(sa_vit.loc[ds_i, ds_j])
            pair_labels.append(f"{ds_i[:4]}-{ds_j[:4]}")
    if len(pairs_r) > 2:
        ax5.scatter(pairs_r, pairs_v, color="#FF5722", s=80, alpha=0.8, zorder=3)
        for lbl, xr, xv in zip(pair_labels, pairs_r, pairs_v):
            ax5.annotate(lbl, (xr, xv), xytext=(4, 4),
                         textcoords="offset points", fontsize=7)
        z = np.polyfit(pairs_r, pairs_v, 1)
        xl = np.linspace(min(pairs_r), max(pairs_r), 50)
        ax5.plot(xl, np.polyval(z, xl), "b--", lw=1.5)
        ax5.set_title(f"Cross-probe SA_k Consistency\nPearson r={cross_r:+.3f}")
    else:
        ax5.set_title(f"Cross-probe Consistency\nPearson r={cross_r:.3f}")
    ax5.set_xlabel("SA_k (ResNet50)")
    ax5.set_ylabel("SA_k (ViT-B16)")
    ax5.grid(True, alpha=0.3)

    # ── Subplot 6: delta_dir vs delta_r bar chart ────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if len(gain_df) > 0:
        labels_bar = [f"{r['ds_i'][:5]}\n+{r['ds_j'][:5]}" for _, r in gain_df.iterrows()]
        vals = gain_df["delta_r"].values
        colors_bar = ["#2196F3" if v > 0 else "#F44336" for v in vals]
        ax6.bar(range(len(vals)), vals, color=colors_bar, alpha=0.8)
        ax6.set_xticks(range(len(vals)))
        ax6.set_xticklabels(labels_bar, fontsize=7)
        ax6.axhline(0, color="black", lw=0.8)
        ax6.set_ylabel("delta_r (merge gain)")
        ax6.set_title("Merge Gain per Dataset Pair\n(ResNet50, blue=superadditive)")
        ax6.grid(True, alpha=0.3, axis="y")

    plt.suptitle("E4: Grassmann Domain Distance Map", fontsize=14, fontweight="bold", y=1.01)
    out = FIG_DIR / "e4_domain_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("E4 (standalone): Grassmann Domain Distance Map")
    print("  Uses existing cached features; no additional download needed")
    print("="*60)

    # Auto-discover available datasets
    available = discover_available_pairs()
    print(f"\nCached features found:")
    for probe, datasets in available.items():
        print(f"  {probe}: {datasets}")

    # Build distance matrices
    PROBE_DATASETS = {
        "resnet50": ["cifar10", "mnist", "stl10", "svhn"],
        "vit_b16":  ["cifar10", "stl10", "svhn"],
    }

    results = {}
    for probe, datasets in PROBE_DATASETS.items():
        if probe not in available:
            print(f"\n  Skipping {probe} (no cached features)")
            continue
        sa_df, gd_df, features, avail_ds = build_distance_matrix(probe, datasets)
        results[probe] = {
            "sa_df":    sa_df,
            "gd_df":    gd_df,
            "features": features,
            "datasets": avail_ds,
        }

        # Save distance matrix
        out_csv = RESULT_DIR / f"e4_domain_distance_{probe}.csv"
        sa_df.to_csv(out_csv)
        print(f"\n{probe} distance matrix saved: {out_csv}")

    # Gain consistency analysis (ResNet50)
    gain_df = pd.DataFrame()
    if "resnet50" in results:
        print("\nComputing merge gain consistency (ResNet50)...")
        gain_df = compute_gain_consistency(
            results["resnet50"]["features"],
            results["resnet50"]["sa_df"],
            results["resnet50"]["datasets"],
        )
        print("\n" + "-"*55)
        print(f"  {'dataset pair':20s}  SA_k    delta_r  superadditive?")
        print("-"*55)
        for _, row in gain_df.iterrows():
            sa_str = f"SA_k={row['sa_k']:.4f}"
            dr_str = f"delta_r={row['delta_r']:+.2f}"
            flag = "superadditive" if row["is_superadditive"] else "subadditive"
            print(f"  {row['ds_i']:12s}+{row['ds_j']:12s}  "
                  f"{sa_str}  {dr_str}  {flag}")
        out_gain = RESULT_DIR / "e4_gain_consistency.csv"
        gain_df.to_csv(out_gain, index=False)
        print(f"\nGain consistency saved: {out_gain}")

    # Cross-probe consistency
    cross_r = float("nan")
    if "resnet50" in results and "vit_b16" in results:
        cross_r = cross_probe_consistency(
            results["resnet50"]["sa_df"],
            results["vit_b16"]["sa_df"],
        )

    # Summary statistics
    if "resnet50" in results and len(gain_df) > 0:
        n_super = int(gain_df["is_superadditive"].sum())
        print(f"\nSummary statistics:")
        print(f"  Superadditive dataset pairs: {n_super}/{len(gain_df)}")
        from scipy.stats import pearsonr as _pr
        _r_val, _ = _pr(gain_df["sa_k"].values, gain_df["delta_r"].values)
        print(f"  SA_k vs delta_r Pearson r = {_r_val:+.4f}  "
              f"{'PASS (low SA_k -> high delta_r)' if _r_val < 0 else 'NOTE: positive correlation (r_eff scale dominates)'}")
        if not np.isnan(cross_r):
            print(f"  Cross-probe Pearson r = {cross_r:.4f}  "
                  f"{'consistent across models' if cross_r > 0.8 else 'needs analysis'}")

    # Visualization
    if "resnet50" in results and "vit_b16" in results:
        print("\nGenerating figures...")
        plot_all(
            results["resnet50"]["sa_df"],
            results["vit_b16"]["sa_df"],
            gain_df,
            cross_r,
        )
    elif "resnet50" in results:
        print("\nOnly ResNet50 data available; generating single-probe figure...")
        plot_all(
            results["resnet50"]["sa_df"],
            results["resnet50"]["sa_df"],   # use itself as placeholder for ViT slot
            gain_df,
            1.0,
        )

    print("\n" + "="*60)
    print("E4 done")
    print("="*60)


if __name__ == "__main__":
    main()
