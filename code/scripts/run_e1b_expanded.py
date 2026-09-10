"""
E1b Collapse-4S Extended Association Study: 6 pairs -> 30 pairs
================================================================
Expands the original 3 pairs x 2 probes = 6 rows to:
  C(6, 2) = 15 unique dataset pairs x 2 probes = 30 rows

Available datasets: cifar10, stl10, svhn, mnist, fashion_mnist, cifar100
Note: NLP datasets (BERT features) are not yet cached; marked as Future Work.

Output:
  results/e1b_expanded_30pairs.csv  -- 30-row expanded results
  results/e1b_expanded_summary.txt  -- R^2 and superadditivity statistics
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from metrics.collapse_core import (
    collapse_4s,
    effective_rank,
    is_superadditive,
    nuclear_mass,
    superadditivity_delta,
)
from metrics.subspace_alignment import compute_subspace_alignment

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

sys.path.insert(0, BASE)

# ──────────────────────────────────────────────
# File map (all cached feature files)
# ──────────────────────────────────────────────

FILE_MAP = {
    ('resnet50', 'cifar10'):       'feat_resnet50_cifar10.npz',
    ('resnet50', 'stl10'):         'feat_resnet50_stl10.npz',
    ('resnet50', 'svhn'):          'feat_resnet50_svhn.npz',
    ('resnet50', 'mnist'):         'feat_resnet50_mnist_train.npz',
    ('resnet50', 'fashion_mnist'): 'feat_resnet50_fashion_mnist_train.npz',
    ('resnet50', 'cifar100'):      'feat_resnet50_cifar100_train.npz',
    ('vit_b16',  'cifar10'):       'feat_vit_b16_cifar10.npz',
    ('vit_b16',  'stl10'):         'feat_vit_b16_stl10.npz',
    ('vit_b16',  'svhn'):          'feat_vit_b16_svhn.npz',
    ('vit_b16',  'mnist'):         'feat_vit_b16_mnist_train.npz',
    ('vit_b16',  'fashion_mnist'): 'feat_vit_b16_fashion_mnist_train.npz',
    ('vit_b16',  'cifar100'):      'feat_vit_b16_cifar100_train.npz',
}

# Dataset semantic domain labels (for result analysis)
DOMAIN_MAP = {
    'cifar10':       'natural',
    'stl10':         'natural',
    'svhn':          'digit_scene',
    'mnist':         'handwritten',
    'fashion_mnist': 'handwritten',
    'cifar100':      'natural',
}

# ──────────────────────────────────────────────
# Core pair computation
# ──────────────────────────────────────────────

def load_features(probe, ds, max_n=2000, seed=42):
    """Load feature matrix and apply per-sample L2 normalization."""
    fname = FILE_MAP.get((probe, ds))
    if fname is None:
        return None
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    H = data['H'].astype(np.float32)
    # Subsample to at most max_n samples (ensures comparability across dataset sizes)
    if len(H) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(H), max_n, replace=False)
        H = H[idx]
    # Per-sample L2 normalization (required precondition for the framework)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H = H / np.maximum(norms, 1e-8)
    return H.astype(np.float32)


def compute_sa_k(H_A, H_B, k=20):
    """Subspace alignment SA_k in [0,1]; higher values indicate more similar directions."""
    return float(compute_subspace_alignment(H_A, H_B, k=k)["sa_k"])


def compute_pair(probe, ds_A, ds_B, k=20):
    """Compute Collapse-4S diagnostics for one measured dataset pair."""
    H_A = load_features(probe, ds_A)
    H_B = load_features(probe, ds_B)
    if H_A is None or H_B is None:
        return None

    r_A = effective_rank(H_A)
    r_B = effective_rank(H_B)

    H_AB = np.concatenate([H_A, H_B], axis=0)
    r_merged = effective_rank(H_AB)

    r_ideal = r_A + r_B
    collapse_ratio = max(0.0, (r_ideal - r_merged) / r_ideal) if r_ideal > 0 else 0.0

    # SA_k (directional alignment alpha used by the summary predictor)
    sa_k = compute_sa_k(H_A, H_B, k=k)

    # Energy ratio gamma = ||H_B||_* / ||H_A||_* (nuclear-norm ratio)
    nuclear_A = nuclear_mass(H_A)
    nuclear_B = nuclear_mass(H_B)
    gamma = nuclear_B / nuclear_A

    prediction = collapse_4s(r_A, r_B, gamma, sa_k)
    r_pred = prediction.prediction
    delta_r = superadditivity_delta(r_merged, r_A, r_B, gamma)
    superadditive = is_superadditive(r_merged, r_A, r_B, gamma)
    pred_error = abs(r_pred - r_merged)

    # Dominant cause
    direction_dominant = sa_k > 0.5
    energy_dominant    = gamma > 3.0 or gamma < 0.33
    if direction_dominant and energy_dominant:
        cause = "both"
    elif direction_dominant:
        cause = "direction"
    elif energy_dominant:
        cause = "energy"
    elif superadditive:
        cause = "superadditive"
    else:
        cause = "none"

    # Expected behavior regime (based on theory)
    if sa_k < 0.2 and 0.5 <= gamma <= 2.0:
        expected_regime = "superadditive"
    elif sa_k > 0.5:
        expected_regime = "direction_collapse"
    elif gamma > 3.0 or gamma < 0.33:
        expected_regime = "energy_collapse"
    else:
        expected_regime = "transition"

    # Semantic domain information
    domain_A = DOMAIN_MAP.get(ds_A, 'unknown')
    domain_B = DOMAIN_MAP.get(ds_B, 'unknown')
    cross_domain = (domain_A != domain_B)

    return {
        'probe':           probe,
        'ds_A':            ds_A,
        'ds_B':            ds_B,
        'domain_A':        domain_A,
        'domain_B':        domain_B,
        'cross_domain':    cross_domain,
        'r_eff_A':         r_A,
        'r_eff_B':         r_B,
        'r_eff_dominant':  prediction.dominant_rank,
        'r_eff_subordinate': prediction.subordinate_rank,
        'r_eff_merged':    r_merged,
        'r_eff_ideal':     r_ideal,
        'collapse_ratio':  collapse_ratio,
        'delta_r':         delta_r,
        'is_superadditive': superadditive,
        'sa_k':            sa_k,
        'nuclear_A':       nuclear_A,
        'nuclear_B':       nuclear_B,
        'energy_ratio':    gamma,
        'q':               prediction.q,
        'r_pred_theory':   r_pred,
        'pred_error':      pred_error,
        'dominant_cause':  cause,
        'expected_regime': expected_regime,
    }


# ──────────────────────────────────────────────
# Evaluation functions
# ──────────────────────────────────────────────

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


# ──────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────

def run():
    print("\n" + "="*70)
    print("E1b Collapse Theorem Extended Validation: 6 pairs -> 30 pairs")
    print("="*70)

    # All C(6,2)=15 pairs from 6 datasets
    datasets = ['cifar10', 'stl10', 'svhn', 'mnist', 'fashion_mnist', 'cifar100']
    pairs    = list(itertools.combinations(datasets, 2))
    probes   = ['resnet50', 'vit_b16']

    print(f"\nExperiment design: {len(datasets)} datasets x C(6,2)={len(pairs)} pairs x {len(probes)} probes")
    print(f"  Max theoretical rows: {len(pairs) * len(probes)}")
    print(f"  Datasets: {datasets}")
    print(f"  Note: NLP datasets not yet cached; marked as Future Work")

    print("\n[1/3] Loading features and computing collapse metrics...\n")
    rows = []
    skipped = []

    for probe in probes:
        for ds_A, ds_B in pairs:
            result = compute_pair(probe, ds_A, ds_B, k=20)
            if result is None:
                skipped.append(f"  SKIP {probe}/{ds_A}+{ds_B} (features missing)")
                continue
            rows.append(result)

            sign = "superadditive" if result['is_superadditive'] else "subadditive"
            cross = "cross-domain" if result['cross_domain'] else "same-domain"
            print(f"  [{probe:9s}] {ds_A:13s}+{ds_B:13s}  "
                  f"r=[{result['r_eff_A']:.0f}+{result['r_eff_B']:.0f}→{result['r_eff_merged']:.0f}]  "
                  f"SA={result['sa_k']:.3f}  γ={result['energy_ratio']:.2f}  "
                  f"err={result['pred_error']:.1f}  {sign}  {cross}")

    if skipped:
        print("\n  Skipped configurations:")
        for s in skipped:
            print(s)

    df = pd.DataFrame(rows)
    n_total = len(df)
    print(f"\nValid data rows: {n_total}")

    # ──────────────────────────────────────────────
    # [2/3] Statistical analysis
    # ──────────────────────────────────────────────
    print("\n[2/3] Statistical analysis...\n")

    # R^2: theoretical prediction vs. measured
    y_true = df['r_eff_merged'].values
    y_pred = df['r_pred_theory'].values
    r2 = compute_r2(y_true, y_pred)

    # Spearman correlation: SA_k vs collapse_ratio
    rho_sa_collapse, p_sa_collapse = spearmanr(df['sa_k'].values, df['collapse_ratio'].values)

    # Pearson r: |log gamma| vs collapse_ratio
    log_gamma = np.log(df['energy_ratio'].values.clip(0.01))
    r_gamma_collapse, _ = pearsonr(np.abs(log_gamma), df['collapse_ratio'].values)

    # Superadditivity statistics
    n_superadd = int(df['is_superadditive'].sum())
    n_subaddive = n_total - n_superadd
    pct_superadd = n_superadd / n_total * 100

    # Cross-domain vs same-domain superadditivity comparison
    cross = df['cross_domain']
    n_super_cross = int(df[cross]['is_superadditive'].sum())
    n_super_same  = int(df[~cross]['is_superadditive'].sum())
    n_cross_total = int(cross.sum())
    n_same_total  = int((~cross).sum())

    # Prediction error statistics
    mean_err = float(df['pred_error'].mean())
    median_err = float(df['pred_error'].median())
    max_err = float(df['pred_error'].max())

    print(f"  {'='*50}")
    print(f"  Theoretical prediction accuracy:")
    print(f"    R^2(theory vs measured r_eff)  = {r2:.4f}  {'pass' if r2 > 0.80 else 'needs analysis'}")
    print(f"    Mean error = {mean_err:.2f}  Median error = {median_err:.2f}  Max error = {max_err:.2f}")
    print()
    print(f"  SA_k vs collapse correlation:")
    print(f"    Spearman rho(SA_k, collapse_ratio) = {rho_sa_collapse:+.4f}  p={p_sa_collapse:.4f}")
    print(f"    Pearson r(|log gamma|, collapse_ratio) = {r_gamma_collapse:+.4f}")
    print()
    print(f"  Superadditive / subadditive distribution:")
    print(f"    Superadditive (tolerance-aware): {n_superadd}/{n_total} = {pct_superadd:.1f}%")
    print(f"    Not superadditive (tolerance-aware): {n_subaddive}/{n_total}")
    print(f"    Cross-domain: {n_super_cross}/{n_cross_total} superadditive "
          f"= {n_super_cross/max(n_cross_total,1)*100:.1f}%  (complementary domains -> superadditive)")
    print(f"    Same-domain: {n_super_same}/{n_same_total} superadditive "
          f"= {n_super_same/max(n_same_total,1)*100:.1f}%  (redundant domains -> subadditive)")
    print(f"  {'='*50}")

    # ──────────────────────────────────────────────
    # Per-regime analysis
    # ──────────────────────────────────────────────
    print("\n  --- Analysis by expected behavior regime ---")
    for regime in ['superadditive', 'transition', 'direction_collapse', 'energy_collapse']:
        sub = df[df['expected_regime'] == regime]
        if len(sub) == 0:
            continue
        actual_super = int(sub['is_superadditive'].sum())
        err_m = sub['pred_error'].mean()
        print(f"  {regime:20s}  n={len(sub):3d}  "
              f"superadd={actual_super}/{len(sub)}  "
              f"mean_err={err_m:.2f}  "
              f"R2_sub={compute_r2(sub['r_eff_merged'].values, sub['r_pred_theory'].values):.3f}")

    # ──────────────────────────────────────────────
    # Dataset comparison by SA_k quartile (high/low contrast)
    # ──────────────────────────────────────────────
    print("\n  --- Grouped by SA_k quartile ---")
    q25, q75 = df['sa_k'].quantile(0.25), df['sa_k'].quantile(0.75)
    low_sa  = df[df['sa_k'] <= q25]
    high_sa = df[df['sa_k'] >= q75]
    print(f"  Low SA_k (<=={q25:.3f}, n={len(low_sa)}):  "
          f"superadd={int(low_sa['is_superadditive'].sum())}/{len(low_sa)}  "
          f"mean_collapse={low_sa['collapse_ratio'].mean():.3f}")
    print(f"  High SA_k (>={q75:.3f}, n={len(high_sa)}): "
          f"superadd={int(high_sa['is_superadditive'].sum())}/{len(high_sa)}  "
          f"mean_collapse={high_sa['collapse_ratio'].mean():.3f}")

    # ──────────────────────────────────────────────
    # [3/3] Comparison with original 6 pairs
    # ──────────────────────────────────────────────
    print("\n[3/3] Comparing with original 6 pairs (e1b_real_pairs.csv)...\n")
    orig_path = os.path.join(RESULTS, 'e1b_real_pairs.csv')
    if os.path.exists(orig_path):
        df_orig = pd.read_csv(orig_path)
        r2_orig = compute_r2(df_orig['r_eff_merged'].values, df_orig['r_pred_theory'].values)
        required = {'r_eff_merged', 'r_eff_A', 'r_eff_B', 'energy_ratio'}
        if not required.issubset(df_orig.columns):
            raise ValueError('original E1b results lack inputs for canonical superadditivity')
        super_orig = sum(
            is_superadditive(
                row['r_eff_merged'], row['r_eff_A'], row['r_eff_B'],
                row['energy_ratio'],
            )
            for _, row in df_orig.iterrows()
        )
        print(f"  Original experiment (n=6):  R^2={r2_orig:.4f}  superadd={super_orig}/{len(df_orig)}")
        print(f"  Extended experiment (n={n_total}): R^2={r2:.4f}  superadd={n_superadd}/{n_total}")
    else:
        print("  (Original file not found, skipping comparison)")

    # ──────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────
    out_csv = os.path.join(RESULTS, 'e1b_expanded_30pairs.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}  ({n_total} rows)")

    summary_lines = [
        "=" * 60,
        "E1b Collapse Theorem Extended Validation: 30-pair result summary",
        "=" * 60,
        f"Valid data rows:           {n_total} / {len(pairs)*len(probes)} (theoretical max)",
        f"",
        f"Theoretical prediction accuracy:",
        f"  R^2(theory vs measured) = {r2:.4f}  {'(pass >0.80)' if r2 > 0.80 else '(needs analysis)'}",
        f"  Mean prediction error   = {mean_err:.2f}",
        f"",
        f"SA_k vs collapse correlation:",
        f"  Spearman rho(SA_k, collapse_ratio) = {rho_sa_collapse:+.4f}  p={p_sa_collapse:.4f}",
        f"  Pearson r(|log gamma|, collapse_ratio) = {r_gamma_collapse:+.4f}",
        f"",
        f"Superadditivity / subadditivity:",
        f"  Total superadditive fraction = {n_superadd}/{n_total} ({pct_superadd:.1f}%)",
        f"  Cross-domain superadditive = {n_super_cross}/{n_cross_total} ({n_super_cross/max(n_cross_total,1)*100:.1f}%)",
        f"  Same-domain superadditive  = {n_super_same}/{n_same_total} ({n_super_same/max(n_same_total,1)*100:.1f}%)",
        f"",
        f"Note: NLP/CV heterogeneous domain pairs are Future Work (requires BERT features)",
    ]
    summary_str = "\n".join(summary_lines)

    out_txt = os.path.join(RESULTS, 'e1b_expanded_summary.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(summary_str)
    print(f"Summary saved: {out_txt}")
    print()
    print(summary_str)

    return df, r2, n_superadd, n_total


if __name__ == "__main__":
    run()
