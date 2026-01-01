"""
Experiment E1: Collapse Model Validation

Goal:
    Validate the Collapse Model (Proposition 1): r_eff([H_A; H_B]) is jointly
    determined by alpha (SA_k) and gamma (energy ratio).

Sub-experiments:
    E1a -- Synthetic controlled experiment (precise control of alpha and gamma)
    E1b -- Real dataset pairs (validation against Paper01 counterexamples)

Outputs:
    results/e1a_synth_grid.csv      -- r_eff collapse heatmap data over (alpha, gamma) grid
    results/e1b_real_pairs.csv      -- SA_k vs. collapse amount for real dataset pairs
    results/e1_summary.json         -- R^2 and related statistics
    figures/e1a_heatmap.png         -- heatmap (alpha x gamma -> collapse ratio)
    figures/e1b_scatter.png         -- real datasets: SA_k vs. collapse_ratio

Usage:
    python experiment_e1_theorem.py              # full experiment
    python experiment_e1_theorem.py --mini       # synthetic + 1 dataset pair (CPU-friendly)
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import (
    RESULT_DIR, FIG_DIR, SEED,
    E1_ALPHA_GRID, E1_GAMMA_GRID,
    E1_SYNTH_D, E1_SYNTH_K, E1_SYNTH_N, E1_N_REPEAT,
    E1_REAL_PAIRS, E1_PROBES, N_SAMPLES, BATCH_SIZE, K_DEFAULT,
)
from metrics.reff import effective_rank, predict_reff_collapse, make_controlled_pair, reff_theoretical_prediction
from metrics.subspace_alignment import compute_subspace_alignment


# ─────────────────────────────────────────────────────────────
# E1a: Synthetic controlled experiment
# ─────────────────────────────────────────────────────────────

def run_e1a_synthetic(
    alpha_grid: List[float] = E1_ALPHA_GRID,
    gamma_grid: List[float] = E1_GAMMA_GRID,
    d: int = E1_SYNTH_D,
    k: int = E1_SYNTH_K,
    n: int = E1_SYNTH_N,
    n_repeat: int = E1_N_REPEAT,
) -> pd.DataFrame:
    """
    Synthetic controlled experiment: measure r_eff collapse ratio over an (alpha, gamma) grid.

    Returns:
        DataFrame: one row per (alpha, gamma, repeat) result
    """
    print("\n" + "="*60)
    print("E1a: Synthetic controlled experiment")
    print(f"  Grid: alpha={alpha_grid}  gamma={gamma_grid}")
    print(f"  d={d}, k={k}, n={n}, repeats={n_repeat}")
    print("="*60)

    rows = []
    total = len(alpha_grid) * len(gamma_grid) * n_repeat
    done = 0

    for alpha_target in alpha_grid:
        for gamma in gamma_grid:
            for rep in range(n_repeat):
                seed = SEED + rep * 1000 + int(alpha_target * 100) + int(gamma * 10)
                t0 = time.time()

                H_A, H_B = make_controlled_pair(
                    d=d, k=k, n=n,
                    target_alpha=alpha_target,
                    gamma=gamma,
                    seed=seed,
                )

                result = predict_reff_collapse(H_A, H_B, k=k)
                r_pred = reff_theoretical_prediction(
                    result["r_eff_A"], result["r_eff_B"],
                    result["sa_k"], result["energy_ratio"],
                )
                pred_error = abs(r_pred - result["r_eff_merged"])

                rows.append({
                    "alpha_target":   alpha_target,
                    "gamma_target":   gamma,
                    "repeat":         rep,
                    "alpha_actual":   result["sa_k"],
                    "gamma_actual":   result["energy_ratio"],
                    "r_eff_A":        result["r_eff_A"],
                    "r_eff_B":        result["r_eff_B"],
                    "r_eff_merged":   result["r_eff_merged"],
                    "r_eff_ideal":    result["r_eff_ideal"],
                    "collapse_ratio": result["collapse_ratio"],
                    "delta_r":        result["delta_r"],
                    "nuclear_A":      result["nuclear_A"],
                    "nuclear_B":      result["nuclear_B"],
                    "r_star":         result["r_star"],
                    "r_pred_theory":  r_pred,
                    "pred_error":     pred_error,
                    "dominant_cause": result["dominant_cause"],
                    "elapsed_s":      time.time() - t0,
                })

                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] α={alpha_target:.1f} γ={gamma:.1f} "
                          f"rep={rep}  collapse={result['collapse_ratio']:.3f}  "
                          f"pred_err={pred_error:.2f}")

    df = pd.DataFrame(rows)
    out_path = RESULT_DIR / "e1a_synth_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    # Statistics
    r2 = _compute_r2_synth(df)
    n_superadd = (df["delta_r"] > 0).sum()
    n_total    = len(df)
    print(f"\nSynthetic experiment statistics:")
    print(f"  Theory prediction R^2 = {r2:.4f}  (expected > 0.8)")
    print(f"  SA_k up -> collapse up  Pearson r = "
          f"{pearsonr(df['alpha_actual'], df['collapse_ratio'])[0]:.4f}")
    print(f"  |log gamma| up -> collapse up  Pearson r = "
          f"{pearsonr(np.log(df['gamma_actual'].clip(0.01)), df['collapse_ratio'])[0]:.4f}")
    print(f"  Superadditive (delta_r>0): {n_superadd}/{n_total} = {n_superadd/n_total:.1%}")
    # Analyze superadditive cases at alpha~0
    df_low_alpha = df[df["alpha_actual"] < 0.1]
    if len(df_low_alpha) > 0:
        n_super_low = (df_low_alpha["delta_r"] > 0).sum()
        print(f"  SA_k~0 subset: superadditive {n_super_low}/{len(df_low_alpha)} = "
              f"{n_super_low/len(df_low_alpha):.1%}  "
              f"mean delta_r = {df_low_alpha['delta_r'].mean():.2f}")

    return df


def _compute_r2_synth(df: pd.DataFrame) -> float:
    """Compute R^2 of theoretical prediction vs. measured r_eff_merged."""
    y_true = df["r_eff_merged"].values
    y_pred = df["r_pred_theory"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ─────────────────────────────────────────────────────────────
# E1b: Real dataset pair validation (v2)
# Strategy: prefer loading cached .npz features; extract only if missing
# Covers three behavior regimes: direction collapse / transition / superadditive
# ─────────────────────────────────────────────────────────────

def _load_or_extract_features(probe_name: str, ds_name: str,
                               n_samples: int) -> np.ndarray:
    """Load cached features (.npz) if available; otherwise extract and save."""
    cache = RESULT_DIR / f"feat_{probe_name}_{ds_name}.npz"
    if cache.exists():
        data = np.load(cache)
        H = data["H"]
        print(f"      [cache] {cache.name}  shape={H.shape}")
    else:
        print(f"      [extract] {probe_name}/{ds_name} ...")
        from probe_models import build_probe, get_device
        from datasets import make_loader
        device = get_device()
        probe  = build_probe(probe_name, device=device)
        loader = make_loader(ds_name, split="train",
                             n_samples=n_samples, batch_size=BATCH_SIZE)
        H = probe.extract(loader, desc=f"extract {ds_name}")
        np.savez_compressed(cache, H=H)
        print(f"      [saved] {cache.name}  shape={H.shape}")

    # per-sample L2 normalization (required by the framework)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return (H / np.maximum(norms, 1e-8)).astype(np.float32)


def run_e1b_real_pairs(
    pairs: List = E1_REAL_PAIRS,
    probes: List = None,
    n_samples: int = N_SAMPLES,
    k: int = K_DEFAULT,
) -> pd.DataFrame:
    """
    Real dataset pair experiment (v2): multi-probe, cache-first, covering three behavior regimes.

    Experimental design based on the revised theorem:
        Regime 1 -- superadditive: svhn + stl10  (SA_k~0, different semantic domains)
        Regime 2 -- direction collapse: cifar10 + svhn  (high SA_k, same visual domain)
        Regime 2 -- transition: cifar10 + stl10  (moderate SA_k)

    Multi-probe validation: checks whether the same dataset pair behaves consistently
    across different model feature spaces.
    """
    if probes is None:
        probes = E1_PROBES

    print("\n" + "="*60)
    print(f"E1b: Real dataset pair validation v2")
    print(f"  Probes: {probes}")
    print(f"  Dataset pairs: {pairs}")
    print(f"  k={k},  n_samples={n_samples}")
    print("="*60)

    rows = []

    for probe_name in probes:
        for ds_A, ds_B in pairs:
            print(f"\n  [{probe_name}]  {ds_A} + {ds_B}")
            H_A = _load_or_extract_features(probe_name, ds_A, n_samples)
            H_B = _load_or_extract_features(probe_name, ds_B, n_samples)

            result = predict_reff_collapse(H_A, H_B, k=k)
            r_pred = reff_theoretical_prediction(
                result["r_eff_A"], result["r_eff_B"],
                result["sa_k"], result["energy_ratio"],
            )

            # Determine expected behavior regime
            if result["sa_k"] < 0.2 and 0.5 <= result["energy_ratio"] <= 2.0:
                expected_regime = "superadditive"
            elif result["sa_k"] > 0.5:
                expected_regime = "direction_collapse"
            elif result["energy_ratio"] > 3.0 or result["energy_ratio"] < 0.33:
                expected_regime = "energy_collapse"
            else:
                expected_regime = "transition"

            rows.append({
                "probe":          probe_name,
                "ds_A":           ds_A,
                "ds_B":           ds_B,
                "r_eff_A":        result["r_eff_A"],
                "r_eff_B":        result["r_eff_B"],
                "r_eff_merged":   result["r_eff_merged"],
                "r_eff_ideal":    result["r_eff_ideal"],
                "collapse_ratio": result["collapse_ratio"],
                "delta_r":        result["delta_r"],
                "sa_k":           result["sa_k"],
                "nuclear_A":      result["nuclear_A"],
                "nuclear_B":      result["nuclear_B"],
                "energy_ratio":   result["energy_ratio"],
                "r_star":         result["r_star"],
                "r_pred_theory":  r_pred,
                "pred_error":     abs(r_pred - result["r_eff_merged"]),
                "dominant_cause": result["dominant_cause"],
                "expected_regime": expected_regime,
            })

            delta_r_sign = "superadditive" if result["delta_r"] > 0 else "subadditive"
            match_str    = ("matches expected" if result["dominant_cause"] == expected_regime
                            or (expected_regime == "superadditive" and result["delta_r"] > 0)
                            else "needs analysis")
            print(f"    r_eff: {result['r_eff_A']:.1f} + {result['r_eff_B']:.1f} "
                  f"→ {result['r_eff_merged']:.1f}  (ideal={result['r_eff_ideal']:.1f})")
            print(f"    SA_k={result['sa_k']:.4f}  γ={result['energy_ratio']:.3f}  "
                  f"collapse={result['collapse_ratio']:.3f}  "
                  f"delta_r={result['delta_r']:+.2f} {delta_r_sign}")
            print(f"    ||H_A||_*={result['nuclear_A']:.1f}  ||H_B||_*={result['nuclear_B']:.1f}  "
                  f"r*={result['r_star']:.3f}")
            print(f"    Theory pred: {r_pred:.1f}  error={abs(r_pred - result['r_eff_merged']):.2f}  "
                  f"expected_regime={expected_regime}  {match_str}")

    df = pd.DataFrame(rows)
    out_path = RESULT_DIR / "e1b_real_pairs.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def plot_e1a_heatmap(df: pd.DataFrame):
    """Plot a triptych: collapse_ratio heatmap / delta_r superadditivity heatmap / theory prediction scatter."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Skipping plots (matplotlib/seaborn not installed)")
        return

    pivot_collapse = df.groupby(["alpha_target", "gamma_target"])["collapse_ratio"].mean().unstack()
    pivot_delta    = df.groupby(["alpha_target", "gamma_target"])["delta_r"].mean().unstack()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    alpha_labels = [f"α={a}" for a in sorted(df["alpha_target"].unique())]
    gamma_labels = [f"γ={g}" for g in sorted(df["gamma_target"].unique())]

    # Left: collapse ratio heatmap
    sns.heatmap(pivot_collapse, ax=axes[0], annot=True, fmt=".2f",
                cmap="YlOrRd", vmin=0, vmax=1,
                yticklabels=alpha_labels, xticklabels=gamma_labels)
    axes[0].set_title("Collapse ratio\n(red = severe collapse)")
    axes[0].set_xlabel("Energy ratio gamma (nuclear norm)")
    axes[0].set_ylabel("Subspace alignment alpha = SA_k")

    # Center: delta_r heatmap (positive = superadditive blue, negative = subadditive red)
    vmax_d = float(np.abs(pivot_delta.values).max()) + 1e-3
    sns.heatmap(pivot_delta, ax=axes[1], annot=True, fmt=".1f",
                cmap="RdBu_r", center=0, vmin=-vmax_d, vmax=vmax_d,
                yticklabels=alpha_labels, xticklabels=gamma_labels)
    axes[1].set_title("Delta_r = r_merged - max(r_A,r_B)\n(blue = superadditive, red = subadditive)")
    axes[1].set_xlabel("Energy ratio gamma (nuclear norm)")
    axes[1].set_ylabel("Subspace alignment alpha = SA_k")

    # Right: theory prediction vs. measured scatter
    sc = axes[2].scatter(df["r_eff_merged"], df["r_pred_theory"],
                         c=df["alpha_actual"], cmap="viridis", alpha=0.5, s=20)
    lim = max(df["r_eff_merged"].max(), df["r_pred_theory"].max()) * 1.05
    axes[2].plot([0, lim], [0, lim], "r--", lw=1.5, label="y=x")
    plt.colorbar(sc, ax=axes[2], label="SA_k (alpha)")
    r2 = _compute_r2_synth(df)
    axes[2].set_xlabel("r_eff measured")
    axes[2].set_ylabel("r_eff theory prediction (mixed-entropy formula)")
    axes[2].set_title(f"Theorem prediction accuracy  R^2={r2:.3f}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "e1a_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


def plot_e1b_scatter(df: pd.DataFrame):
    """Plot a real dataset diptych: (a) SA_k vs delta_r  (b) SA_k vs collapse_ratio, colored by probe."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    probes      = df["probe"].unique() if "probe" in df.columns else ["resnet50"]
    pair_labels = (df["ds_A"] + "+" + df["ds_B"]).unique()
    colors      = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, probe in enumerate(probes):
        sub = df[df["probe"] == probe] if "probe" in df.columns else df
        # (a) SA_k vs delta_r (superadditivity / subadditivity)
        axes[0].scatter(sub["sa_k"], sub["delta_r"],
                        s=90, color=colors[i % 10], label=probe, zorder=3)
        for _, row in sub.iterrows():
            axes[0].annotate(f"{row['ds_A']}+{row['ds_B']}",
                             (row["sa_k"], row["delta_r"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=7)
        # (b) SA_k vs collapse_ratio
        axes[1].scatter(sub["sa_k"], sub["collapse_ratio"],
                        s=90, color=colors[i % 10], label=probe, zorder=3)
        for _, row in sub.iterrows():
            axes[1].annotate(f"{row['ds_A']}+{row['ds_B']}",
                             (row["sa_k"], row["collapse_ratio"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=7)

    # Superadditive / subadditive boundary line
    axes[0].axhline(0, color="gray", linestyle="--", lw=1.2, label="delta_r=0")
    axes[0].set_xlabel("SA_k (subspace alignment)")
    axes[0].set_ylabel("delta_r = r_merged - max(r_A,r_B)")
    axes[0].set_title("Superadditivity analysis\n(above = superadditive, below = subadditive)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if len(df) >= 3:
        rho = spearmanr(df["sa_k"], df["collapse_ratio"])[0]
    else:
        rho = float("nan")
    axes[1].set_xlabel("SA_k (subspace alignment)")
    axes[1].set_ylabel("collapse_ratio")
    axes[1].set_title(f"Direction collapse validation  Spearman rho={rho:.3f}")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / "e1b_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out}")


# ─────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────

def save_summary(df_synth: pd.DataFrame, df_real: pd.DataFrame):
    rho_alpha = spearmanr(df_synth["alpha_actual"], df_synth["collapse_ratio"])[0]
    rho_gamma = spearmanr(
        np.log(df_synth["gamma_actual"].clip(0.01)), df_synth["collapse_ratio"])[0]

    n_super = int((df_synth["delta_r"] > 0).sum())
    n_total = int(len(df_synth))
    df_low  = df_synth[df_synth["alpha_actual"] < 0.1]
    n_super_low = int((df_low["delta_r"] > 0).sum()) if len(df_low) > 0 else 0

    summary = {
        "e1a_synthetic": {
            "n_samples":                  n_total,
            "r2_theory_pred":             round(_compute_r2_synth(df_synth), 4),
            "spearman_alpha_collapse":    round(float(rho_alpha), 4),
            "spearman_loggamma_collapse": round(float(rho_gamma), 4),
            "mean_pred_error":            round(float(df_synth["pred_error"].mean()), 4),
            "n_superadditive":            n_super,
            "frac_superadditive":         round(n_super / n_total, 4),
            "n_superadditive_low_alpha":  n_super_low,
            "frac_superadditive_low_alpha": round(n_super_low / max(len(df_low), 1), 4),
        },
        "e1b_real_pairs": df_real.to_dict(orient="records"),
    }

    out = RESULT_DIR / "e1_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {out}")
    return summary


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E1: r_eff collapse theorem validation v2")
    parser.add_argument("--mini",       action="store_true",
                        help="Fast mode: 3x3 synthetic grid + resnet50 only")
    parser.add_argument("--probe",      default=None,
                        help="Single probe (default: use all E1_PROBES)")
    parser.add_argument("--k",          type=int, default=K_DEFAULT, help="Subspace dimension")
    parser.add_argument("--skip-real",  action="store_true",  help="Skip real dataset experiment")
    parser.add_argument("--skip-synth", action="store_true",  help="Skip synthetic experiment")
    args = parser.parse_args()

    t_start = time.time()

    # ── E1a: Synthetic controlled experiment ──────────────────
    if not args.skip_synth:
        if args.mini:
            # mini: alpha covers the superadditivity boundary (incl. 0.05), gamma covers threshold
            df_synth = run_e1a_synthetic(
                alpha_grid=[0.0, 0.05, 0.2, 0.6, 1.0],
                gamma_grid=[0.33, 1.0, 3.0, 10.0],
                d=256, k=30, n=500, n_repeat=3,
            )
        else:
            df_synth = run_e1a_synthetic()  # use full grid from config
        plot_e1a_heatmap(df_synth)
    else:
        csv = RESULT_DIR / "e1a_synth_grid.csv"
        df_synth = pd.read_csv(csv) if csv.exists() else pd.DataFrame()

    # ── E1b: Real dataset pairs (cache-first) ────────────────
    if not args.skip_real:
        # mini mode uses only resnet50; full mode uses all probes
        probes_to_use = [args.probe] if args.probe else (
            ["resnet50"] if args.mini else E1_PROBES
        )
        pairs_to_use = E1_REAL_PAIRS[:1] if args.mini else E1_REAL_PAIRS
        df_real = run_e1b_real_pairs(
            pairs=pairs_to_use,
            probes=probes_to_use,
            k=args.k,
        )
        plot_e1b_scatter(df_real)
    else:
        csv = RESULT_DIR / "e1b_real_pairs.csv"
        df_real = pd.read_csv(csv) if csv.exists() else pd.DataFrame()

    # ── Summary ───────────────────────────────────────────────
    if not df_synth.empty and not df_real.empty:
        summary = save_summary(df_synth, df_real)
        r2 = summary["e1a_synthetic"]["r2_theory_pred"]
        print(f"\n{'='*60}")
        print(f"E1 done  total time: {(time.time()-t_start)/60:.1f} min")
        print(f"  Main theorem R^2 = {r2:.4f}  {'PASS' if r2 > 0.8 else 'FAIL (please check)'}")


if __name__ == "__main__":
    main()
