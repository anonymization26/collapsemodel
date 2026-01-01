"""
E1b Collapse Theorem Extended Validation: 6 pairs -> 30 pairs
==============================================================
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
# Core computation functions (self-contained, no dependency on metrics module)
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


def effective_rank(H):
    """Spectral-entropy effective rank: r_eff = exp(-sum p_i log p_i), p_i = sigma_i / sum sigma_j"""
    N, d = H.shape
    if N <= d:
        G = (H.astype(np.float64) @ H.astype(np.float64).T) / N
    else:
        G = (H.astype(np.float64).T @ H.astype(np.float64)) / N
    eigvals = np.linalg.eigvalsh(G)[::-1]
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals[eigvals > 1e-10]
    if len(eigvals) == 0:
        return 1.0
    sigma = np.sqrt(eigvals)
    p = sigma / sigma.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_sa_k(H_A, H_B, k=20):
    """Subspace alignment SA_k in [0,1]; higher values indicate more similar directions."""
    _, _, Vt_A = np.linalg.svd(H_A.astype(np.float64), full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B.astype(np.float64), full_matrices=False)
    k_a = min(k, Vt_A.shape[0], Vt_B.shape[0])
    M = Vt_A[:k_a] @ Vt_B[:k_a].T
    sv = np.linalg.svd(M, compute_uv=False)
    cos2 = np.clip(sv[:k_a], 0.0, 1.0) ** 2
    return float(np.mean(cos2))


def reff_theoretical_prediction(r_A, r_B, alpha, gamma):
    """Theoretical r_eff prediction based on the mixture-entropy formula (consistent with metrics/reff.py)."""
    alpha_c = float(np.clip(alpha, 0.0, 1.0 - 1e-8))
    gamma_f = float(gamma)
    A    = 1.0 + gamma_f ** 2
    disc = max(A ** 2 - 4.0 * gamma_f ** 2 * (1.0 - alpha_c), 0.0)
    D    = float(np.sqrt(disc))
    amp_plus  = float(np.sqrt((A + D) / 2.0))
    amp_minus = float(np.sqrt(max((A - D) / 2.0, 0.0)))
    r_star = float(np.clip(
        amp_plus / (amp_plus + amp_minus + 1e-12),
        1e-6, 1.0 - 1e-6,
    ))
    H_bin    = -(r_star * np.log(r_star) + (1.0 - r_star) * np.log(1.0 - r_star))
    H_A_spec = float(np.log(max(r_A, 1e-8)))
    H_B_spec = float(np.log(max(r_B, 1e-8)))
    if gamma_f >= 1.0:
        H_spec_plus, H_spec_minus = H_B_spec, H_A_spec
    else:
        H_spec_plus, H_spec_minus = H_A_spec, H_B_spec
    r_pred = float(np.exp(H_bin + r_star * H_spec_plus + (1.0 - r_star) * H_spec_minus))
    return float(np.clip(r_pred, min(r_A, r_B), r_A + r_B))


def compute_pair(probe, ds_A, ds_B, k=20):
    """Compute collapse-theorem metrics for a dataset pair. Returns a dict or None if features are missing."""
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

    # SA_k (directional alignment alpha used in the collapse theorem)
    sa_k = compute_sa_k(H_A, H_B, k=k)

    # Energy ratio gamma = ||H_B||_* / ||H_A||_* (nuclear-norm ratio)
    _, s_A, _ = np.linalg.svd(H_A.astype(np.float64), full_matrices=False)
    _, s_B, _ = np.linalg.svd(H_B.astype(np.float64), full_matrices=False)
    nuclear_A = float(np.sum(s_A))
    nuclear_B = float(np.sum(s_B))
    gamma = nuclear_B / (nuclear_A + 1e-8)
    r_star = nuclear_A / (nuclear_A + nuclear_B + 1e-8)

    # delta_r (superadditive / subadditive determination)
    r_max = max(r_A, r_B)
    delta_r = r_merged - r_max

    # Theoretical prediction
    r_pred = reff_theoretical_prediction(r_A, r_B, sa_k, gamma)
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
    elif delta_r > 0:
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
        'r_eff_merged':    r_merged,
        'r_eff_ideal':     r_ideal,
        'collapse_ratio':  collapse_ratio,
        'delta_r':         delta_r,
        'sa_k':            sa_k,
        'nuclear_A':       nuclear_A,
        'nuclear_B':       nuclear_B,
        'energy_ratio':    gamma,
        'r_star':          r_star,
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

            sign = "superadditive" if result['delta_r'] > 0 else "subadditive"
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
    n_superadd = int((df['delta_r'] > 0).sum())
    n_subaddive = int((df['delta_r'] <= 0).sum())
    pct_superadd = n_superadd / n_total * 100

    # Cross-domain vs same-domain superadditivity comparison
    cross = df['cross_domain']
    n_super_cross = int((df[cross]['delta_r'] > 0).sum())
    n_super_same  = int((df[~cross]['delta_r'] > 0).sum())
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
    print(f"    Superadditive (delta_r>0): {n_superadd}/{n_total} = {pct_superadd:.1f}%")
    print(f"    Subadditive (delta_r<=0): {n_subaddive}/{n_total}")
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
        actual_super = int((sub['delta_r'] > 0).sum())
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
          f"superadd={int((low_sa['delta_r']>0).sum())}/{len(low_sa)}  "
          f"mean_collapse={low_sa['collapse_ratio'].mean():.3f}")
    print(f"  High SA_k (>={q75:.3f}, n={len(high_sa)}): "
          f"superadd={int((high_sa['delta_r']>0).sum())}/{len(high_sa)}  "
          f"mean_collapse={high_sa['collapse_ratio'].mean():.3f}")

    # ──────────────────────────────────────────────
    # [3/3] Comparison with original 6 pairs
    # ──────────────────────────────────────────────
    print("\n[3/3] Comparing with original 6 pairs (e1b_real_pairs.csv)...\n")
    orig_path = os.path.join(RESULTS, 'e1b_real_pairs.csv')
    if os.path.exists(orig_path):
        df_orig = pd.read_csv(orig_path)
        r2_orig = compute_r2(df_orig['r_eff_merged'].values, df_orig['r_pred_theory'].values)
        super_orig = int((df_orig['delta_r'] > 0).sum())
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
