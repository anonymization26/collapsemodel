"""
E1c: Large-scale real-dataset validation (Collapse Model)
=========================================================
Uses all available feature files to construct the largest possible set of dataset pairs,
validating the accuracy of the Collapse Model prediction formula and superadditivity coverage.

Dataset pairs:
  ResNet-50: 9 datasets -> C(9,2)=36 pairs
  ViT-B/16:  6 datasets -> C(6,2)=15 pairs
  SBERT:     4 text domains -> C(4,2)= 6 pairs
  Total: 57 pairs

Core formula (Collapse Model):
  r_merged_pred = r_dom^r* * r_sub^(1-r*) / (r*^r* * (1-r*)^(1-r*))
  Superadditivity threshold: r_dom/r_sub < rho_c = exp(H_b(r*) / (1-r*))

Output:
  results/e1c_expanded_pairs.csv   -- full data for each pair
  results/e1c_expanded_summary.csv -- R^2, RMSE, superadditivity rate (grouped + global)
"""

import os
import sys
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
N_SUB   = 2000  # uniform subsampling to N_SUB samples (consistent with E1b)
K       = 20    # subspace dimension
SEED    = 42

# ──────────────────────────────────────────────────────────────
# Dataset -> feature file mapping
# ──────────────────────────────────────────────────────────────

FEAT_MAP = {
    # ResNet-50 probe
    ('resnet50', 'cifar10'):        'feat_resnet50_cifar10.npz',
    ('resnet50', 'cifar100'):       'feat_resnet50_cifar100_train.npz',
    ('resnet50', 'fashion_mnist'):  'feat_resnet50_fashion_mnist_train.npz',
    ('resnet50', 'mnist'):          'feat_resnet50_mnist_train.npz',
    ('resnet50', 'stl10'):          'feat_resnet50_stl10.npz',
    ('resnet50', 'svhn'):           'feat_resnet50_svhn.npz',
    ('resnet50', 'bloodmnist'):     'feat_resnet50_bloodmnist_train.npz',
    ('resnet50', 'dermamnist'):     'feat_resnet50_dermamnist_train.npz',
    ('resnet50', 'pathmnist'):      'feat_resnet50_pathmnist_train.npz',
    # ViT-B/16 probe
    ('vit_b16',  'cifar10'):        'feat_vit_b16_cifar10.npz',
    ('vit_b16',  'cifar100'):       'feat_vit_b16_cifar100_train.npz',
    ('vit_b16',  'fashion_mnist'):  'feat_vit_b16_fashion_mnist_train.npz',
    ('vit_b16',  'mnist'):          'feat_vit_b16_mnist_train.npz',
    ('vit_b16',  'stl10'):          'feat_vit_b16_stl10.npz',
    ('vit_b16',  'svhn'):           'feat_vit_b16_svhn.npz',
    # SBERT probe (text domain)
    ('sbert',    'comp'):           'feat_sbert_comp_train.npz',
    ('sbert',    'rec'):            'feat_sbert_rec_train.npz',
    ('sbert',    'sci'):            'feat_sbert_sci_train.npz',
    ('sbert',    'talk'):           'feat_sbert_talk_train.npz',
}

# Probe -> dataset list
PROBE_DATASETS = {
    'resnet50': ['cifar10', 'cifar100', 'fashion_mnist', 'mnist',
                 'stl10', 'svhn', 'bloodmnist', 'dermamnist', 'pathmnist'],
    'vit_b16':  ['cifar10', 'cifar100', 'fashion_mnist', 'mnist', 'stl10', 'svhn'],
    'sbert':    ['comp', 'rec', 'sci', 'talk'],
}

# Dataset type labels (used for grouped analysis)
DOMAIN_TYPE = {
    'cifar10': 'natural', 'cifar100': 'natural', 'stl10': 'natural',
    'svhn': 'natural', 'fashion_mnist': 'natural', 'mnist': 'natural',
    'bloodmnist': 'medical', 'dermamnist': 'medical', 'pathmnist': 'medical',
    'comp': 'text', 'rec': 'text', 'sci': 'text', 'talk': 'text',
}


# ──────────────────────────────────────────────────────────────
# Feature loading and preprocessing
# ──────────────────────────────────────────────────────────────

def load_feat(probe, ds, n_sub=N_SUB, seed=SEED):
    key = (probe, ds)
    fname = FEAT_MAP.get(key)
    if fname is None:
        return None
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    H = data['H'].astype(np.float64)
    # L2 normalization
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H = H / np.maximum(norms, 1e-10)
    # Subsampling
    if H.shape[0] > n_sub:
        rng = np.random.default_rng(seed)
        idx = rng.choice(H.shape[0], n_sub, replace=False)
        H = H[idx]
    return H.astype(np.float64)


# ──────────────────────────────────────────────────────────────
# Collapse Model core computation functions
# ──────────────────────────────────────────────────────────────

def reff(H):
    """Effective rank: exp(-sum p_i log p_i), p_i = sigma_i / sum sigma_j (spectral-entropy effective rank, consistent with E1b)"""
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


def nuclear_norm(H):
    """Nuclear norm: sum of singular values."""
    sv = np.linalg.svd(H, compute_uv=False)
    return float(sv.sum())


def sa_k(H_A, H_B, k=K):
    """Subspace alignment: mean squared cosine of top-k principal angles."""
    _, _, Vt_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B, full_matrices=False)
    k_use = min(k, Vt_A.shape[0], Vt_B.shape[0])
    M = Vt_A[:k_use] @ Vt_B[:k_use].T
    sv = np.linalg.svd(M, compute_uv=False)
    cos2 = np.clip(sv[:k_use], 0.0, 1.0) ** 2
    return float(np.mean(cos2))


def h_binary(p):
    """Binary entropy function H_b(p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(-p * np.log(p) - (1 - p) * np.log(1 - p))


def collapse_model_predict(r_A, r_B, gamma, alpha):
    """
    Predict the merged effective rank with the Collapse Model (consistent with E1b reff_theoretical_prediction).

    Args:
      r_A, r_B : individual effective ranks of the two matrices
      gamma    : ||H_B||_* / ||H_A||_* (nuclear-norm ratio)
      alpha    : SA_k (subspace alignment) in [0,1]

    Returns:
      r_merged_pred, r_dom, r_sub, r_star, rho_c
    """
    alpha_c = float(np.clip(alpha, 0.0, 1.0 - 1e-8))
    gamma_f = float(gamma)

    # 2x2 Gram block discriminant (core of the Collapse Model)
    A    = 1.0 + gamma_f ** 2
    disc = max(A ** 2 - 4.0 * gamma_f ** 2 * (1.0 - alpha_c), 0.0)
    D    = float(np.sqrt(disc))
    amp_plus  = float(np.sqrt((A + D) / 2.0))
    amp_minus = float(np.sqrt(max((A - D) / 2.0, 0.0)))

    # r* = dominant amplitude weight (nuclear-norm normalized)
    r_star = float(np.clip(
        amp_plus / (amp_plus + amp_minus + 1e-12),
        1e-6, 1.0 - 1e-6,
    ))

    # Binary entropy H_b(r*)
    H_bin = -(r_star * np.log(r_star) + (1.0 - r_star) * np.log(1.0 - r_star))

    # Spectral-entropy weighting (dominant amplitude corresponds to larger effective rank direction)
    H_A_spec = float(np.log(max(r_A, 1e-8)))
    H_B_spec = float(np.log(max(r_B, 1e-8)))
    if gamma_f >= 1.0:
        H_spec_plus, H_spec_minus = H_B_spec, H_A_spec
    else:
        H_spec_plus, H_spec_minus = H_A_spec, H_B_spec

    r_pred = float(np.exp(H_bin + r_star * H_spec_plus + (1.0 - r_star) * H_spec_minus))
    r_pred = float(np.clip(r_pred, min(r_A, r_B), r_A + r_B))

    # Superadditivity critical ratio rho_c
    r_dom = max(r_A, r_B)
    r_sub = min(r_A, r_B)
    denom = max(1.0 - r_star, 1e-10)
    rho_c = np.exp(H_bin / denom)

    return float(r_pred), float(r_dom), float(r_sub), float(r_star), float(rho_c)


# ──────────────────────────────────────────────────────────────
# Per-pair computation
# ──────────────────────────────────────────────────────────────

def compute_pair(probe, ds_A, ds_B):
    H_A = load_feat(probe, ds_A)
    H_B = load_feat(probe, ds_B)
    if H_A is None or H_B is None:
        return None

    r_A = reff(H_A)
    r_B = reff(H_B)

    # Stack directly (each row already L2-normalized; no re-normalization after merge, consistent with E1b)
    H_merged = np.vstack([H_A, H_B])
    r_merged_true = reff(H_merged)

    alpha = sa_k(H_A, H_B, k=K)

    # gamma = nuclear-norm ratio (sum sigma_B / sum sigma_A), consistent with E1b
    _, s_A, _ = np.linalg.svd(H_A.astype(np.float64), full_matrices=False)
    _, s_B, _ = np.linalg.svd(H_B.astype(np.float64), full_matrices=False)
    nuclear_A = float(np.sum(s_A))
    nuclear_B = float(np.sum(s_B))
    gamma = nuclear_B / max(nuclear_A, 1e-10)

    r_pred, r_dom, r_sub, r_star, rho_c = collapse_model_predict(r_A, r_B, gamma, alpha)

    is_superadditive = (r_merged_true > max(r_A, r_B))
    ratio = r_dom / max(r_sub, 1e-10)
    pred_holds = (ratio < rho_c)

    domain_type_A = DOMAIN_TYPE.get(ds_A, 'unknown')
    domain_type_B = DOMAIN_TYPE.get(ds_B, 'unknown')
    pair_type = (domain_type_A if domain_type_A == domain_type_B
                 else f'{domain_type_A}×{domain_type_B}')

    return {
        'probe':      probe,
        'ds_A':       ds_A,
        'ds_B':       ds_B,
        'pair_type':  pair_type,
        'r_A':        r_A,
        'r_B':        r_B,
        'r_dom':      r_dom,
        'r_sub':      r_sub,
        'r_merged_true': r_merged_true,
        'r_merged_pred': r_pred,
        'alpha':      alpha,
        'gamma':      gamma,
        'nuclear_A':  nuclear_A,
        'nuclear_B':  nuclear_B,
        'r_star':     r_star,
        'rho_c':      rho_c,
        'ratio':      ratio,
        'delta_r':    r_merged_true - max(r_A, r_B),
        'pred_error': abs(r_merged_true - r_pred),
        'pred_error_pct': abs(r_merged_true - r_pred) / max(r_merged_true, 1e-10) * 100,
        'is_superadditive': is_superadditive,
        'pred_holds': pred_holds,        # whether predicted superadditivity matches reality
    }


# ──────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────

def run():
    print('\n' + '='*70)
    print('E1c: Large-scale Collapse Model validation (all available feature files)')
    print('='*70)

    rows = []
    skipped = []

    for probe, datasets in PROBE_DATASETS.items():
        pairs = list(combinations(datasets, 2))
        print(f'\nprobe={probe}, datasets={datasets}, n_pairs={len(pairs)}')
        for ds_A, ds_B in pairs:
            result = compute_pair(probe, ds_A, ds_B)
            if result is None:
                skipped.append((probe, ds_A, ds_B))
                print(f'  SKIP {ds_A} × {ds_B}')
                continue
            rows.append(result)
            status = '✅' if result['is_superadditive'] else '❌'
            pred_ok = '✓' if result['pred_holds'] else '✗'
            print(f'  {ds_A:<15s}×{ds_B:<15s} '
                  f'r_true={result["r_merged_true"]:6.2f} '
                  f'r_pred={result["r_merged_pred"]:6.2f} '
                  f'err={result["pred_error_pct"]:5.1f}% '
                  f'α={result["alpha"]:.3f} γ={result["gamma"]:.2f} '
                  f'{status} {pred_ok}')

    df = pd.DataFrame(rows)
    n = len(df)

    # ── Compute R^2 ──
    ss_res = ((df['r_merged_true'] - df['r_merged_pred'])**2).sum()
    ss_tot = ((df['r_merged_true'] - df['r_merged_true'].mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')
    rmse = np.sqrt(((df['r_merged_true'] - df['r_merged_pred'])**2).mean())
    mae  = (df['r_merged_true'] - df['r_merged_pred']).abs().mean()
    pearson_r, pearson_p = pearsonr(df['r_merged_true'], df['r_merged_pred'])

    n_superadd = df['is_superadditive'].sum()
    n_pred_holds = df['pred_holds'].sum()

    # alpha < 1 subset (exclude fully aligned pairs)
    df_nondeg = df[df['alpha'] < 0.99]
    n_nd = len(df_nondeg)
    n_nd_superadd = df_nondeg['is_superadditive'].sum()

    # gamma > 3 boundary
    df_high_gamma = df[df['gamma'] > 3]
    df_normal = df[df['gamma'] <= 3]

    print('\n' + '='*70)
    print('Global summary')
    print('='*70)
    print(f'  Total pairs          : {n}')
    print(f'  R^2 (pred vs actual) : {r2:.6f}')
    print(f'  RMSE                 : {rmse:.4f}')
    print(f'  MAE                  : {mae:.4f}')
    print(f'  Max error            : {df["pred_error"].max():.4f}')
    print(f'  Pearson r            : {pearson_r:.4f}  p={pearson_p:.6f}')
    print(f'  Superadditive pairs  : {n_superadd}/{n}  ({100*n_superadd/n:.1f}%)')
    print(f'  alpha<0.99 superadd  : {n_nd_superadd}/{n_nd}  ({100*n_nd_superadd/max(n_nd,1):.1f}%)')
    print(f'  gamma>3 pairs        : {len(df_high_gamma)}  '
          f'(of which not superadditive: {(~df_high_gamma["is_superadditive"]).sum()})')

    # Grouped summary
    print('\nBy probe:')
    for probe in df['probe'].unique():
        sub = df[df['probe']==probe]
        ss_res_s = ((sub['r_merged_true'] - sub['r_merged_pred'])**2).sum()
        ss_tot_s = ((sub['r_merged_true'] - sub['r_merged_true'].mean())**2).sum()
        r2_s = 1 - ss_res_s/ss_tot_s if ss_tot_s > 1e-10 else float('nan')
        sa_rate = sub['is_superadditive'].sum() / len(sub) * 100
        print(f'  {probe:<12s}: n={len(sub):2d}  R^2={r2_s:.4f}  superadd={sa_rate:.0f}%')

    print('\nBy pair type:')
    for pt in sorted(df['pair_type'].unique()):
        sub = df[df['pair_type']==pt]
        sa_rate = sub['is_superadditive'].sum() / len(sub) * 100
        print(f'  {pt:<25s}: n={len(sub):2d}  superadd={sa_rate:.0f}%  '
              f'MAE={sub["pred_error"].mean():.3f}')

    # Details for non-superadditive pairs
    df_not_sa = df[~df['is_superadditive']]
    if len(df_not_sa) > 0:
        print(f'\nNon-superadditive pairs ({len(df_not_sa)} pairs):')
        for _, row in df_not_sa.iterrows():
            print(f'  {row["probe"]:<10s} {row["ds_A"]:<15s}x{row["ds_B"]:<15s} '
                  f'ratio={row["ratio"]:.2f}  rho_c={row["rho_c"]:.2f}  '
                  f'alpha={row["alpha"]:.3f}  gamma={row["gamma"]:.2f}')

    # Save
    out_pairs = os.path.join(RESULTS, 'e1c_expanded_pairs.csv')
    df.to_csv(out_pairs, index=False)

    summary = []
    for label, sub in [
        ('global', df),
        ('resnet50', df[df['probe']=='resnet50']),
        ('vit_b16', df[df['probe']=='vit_b16']),
        ('sbert', df[df['probe']=='sbert']),
        ('natural_pairs', df[df['pair_type']=='natural']),
        ('medical_pairs', df[df['pair_type']=='medical']),
        ('text_pairs', df[df['pair_type']=='text']),
        ('cross_domain', df[~df['pair_type'].isin(['natural','medical','text'])]),
        ('alpha<0.99', df[df['alpha']<0.99]),
        ('gamma<=3', df[df['gamma']<=3]),
        ('gamma>3', df[df['gamma']>3]),
    ]:
        if len(sub) == 0:
            continue
        ss_r = ((sub['r_merged_true']-sub['r_merged_pred'])**2).sum()
        ss_t = ((sub['r_merged_true']-sub['r_merged_true'].mean())**2).sum()
        summary.append({
            'group': label, 'n': len(sub),
            'R2': round(1 - ss_r/ss_t if ss_t > 1e-10 else float('nan'), 6),
            'RMSE': round(np.sqrt(((sub['r_merged_true']-sub['r_merged_pred'])**2).mean()), 4),
            'MAE': round((sub['r_merged_true']-sub['r_merged_pred']).abs().mean(), 4),
            'max_err': round(sub['pred_error'].max(), 4),
            'superadd_n': int(sub['is_superadditive'].sum()),
            'superadd_pct': round(sub['is_superadditive'].mean()*100, 1),
        })
    df_summary = pd.DataFrame(summary)
    out_summary = os.path.join(RESULTS, 'e1c_expanded_summary.csv')
    df_summary.to_csv(out_summary, index=False)

    print(f'\nDetails saved: {out_pairs}')
    print(f'Summary saved: {out_summary}')
    print(f'Skipped: {len(skipped)} pairs')

    return df, df_summary


if __name__ == '__main__':
    df, df_summary = run()
    print('\nSummary table:')
    print(df_summary.to_string(index=False))
