"""
E8: Large-Scale Compression at M=34+ (scalability validation)
Runs on remote server after extract_features_m50.py completes.

Uses all available ResNet-50 features to demonstrate Collapse Greedy
scales beyond M=14 (the original pool size).
"""
import os, sys, time, warnings, csv
import numpy as np

from metrics.collapse_core import (
    collapse_4s_predict as cpred,
    effective_rank as reff,
    split_numerical_singular_values,
    stable_singular_values_and_vh,
)
from metrics.subspace_alignment import compute_subspace_alignment

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FEAT_DIR = os.path.join(RESULTS, "p0_features")
SEED = 42
N_SAMPLES = 2000
K_SA = 20

# ============================================================
# Core functions (same as run_compression_v2.py)
# ============================================================

def load(probe, ds, split="train"):
    paths = [
        f"{FEAT_DIR}/{probe}_{ds}_{split}.npz",
        f"{RESULTS}/feat_{probe}_{ds}.npz",
        f"{RESULTS}/feat_{probe}_{ds}_{split}.npz",
    ]
    for p in paths:
        if os.path.exists(p):
            d = np.load(p)
            H = d["H"].astype(np.float64)
            y = d["y"].astype(np.int32) if "y" in d else None
            norms = np.linalg.norm(H, axis=1, keepdims=True)
            H = H / np.maximum(norms, 1e-10)
            rng = np.random.default_rng(SEED)
            if len(H) > N_SAMPLES:
                idx = rng.choice(len(H), N_SAMPLES, replace=False)
                H = H[idx]
                y = y[idx] if y is not None else None
            return H, y
    return None, None


def sa_k(HA, HB, k=K_SA):
    return float(compute_subspace_alignment(HA, HB, k=k)["sa_k"])


# ============================================================
# Compression strategies
# ============================================================

def collapse_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds):
        return list(all_ds)
    # Pre-compute SVD right-singular vectors and spectral norms for candidates
    vt_cache = {}
    snorm_cache = {}
    for d in all_ds:
        s, Vt = stable_singular_values_and_vh(feats[d])
        s, _ = split_numerical_singular_values(s, feats[d].shape)
        vt_cache[d] = Vt[:len(s)]
        snorm_cache[d] = float(s.sum())

    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]
    H_cur = feats[seed].copy()
    r_cur = rc[seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        # SVD of current merged pool — once per greedy step
        s_cur, Vt_cur = stable_singular_values_and_vh(H_cur)
        s_cur, _ = split_numerical_singular_values(s_cur, H_cur.shape)
        Vt_cur = Vt_cur[:len(s_cur)]
        nA = float(s_cur.sum())
        ke_cur = min(K_SA, Vt_cur.shape[0])
        Vt_cur_k = Vt_cur[:ke_cur]

        best_pred, best_d = -1, None
        for d in remaining:
            ke = min(ke_cur, vt_cache[d].shape[0])
            cs = np.linalg.svd(Vt_cur_k[:ke] @ vt_cache[d][:ke].T, compute_uv=False)
            alpha = float(np.mean(np.clip(cs, 0, 1) ** 2))
            pred = cpred(r_cur, rc[d], snorm_cache[d] / nA, alpha)
            if pred > best_pred:
                best_pred, best_d = pred, d
        if best_d is None:
            break
        selected.append(best_d)
        remaining.discard(best_d)
        H_cur = np.concatenate([H_cur, feats[best_d]])
        r_cur = reff(H_cur)
    return selected


def rsum_greedy(feats, rc, all_ds, k):
    return sorted(all_ds, key=lambda d: rc[d], reverse=True)[:k]


def random_select(all_ds, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_ds), k, replace=False)
    return [all_ds[i] for i in idx]


# ============================================================
# Main: discover all available ResNet-50 features and run compression
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  E8: Large-Scale Compression (M=34+)")
    print("=" * 72)

    probe = "resnet50"

    # Discover all available ResNet-50 feature files
    all_ds = set()
    # Pattern 1: p0_features/resnet50_{ds}_train.npz
    if os.path.isdir(FEAT_DIR):
        for f in os.listdir(FEAT_DIR):
            if f.startswith(f"{probe}_") and f.endswith("_train.npz"):
                ds = f[len(f"{probe}_"):-len("_train.npz")]
                all_ds.add(ds)
    # Pattern 2: results/feat_resnet50_{ds}_train.npz or feat_resnet50_{ds}.npz
    for f in os.listdir(RESULTS):
        if f.startswith(f"feat_{probe}_") and f.endswith(".npz"):
            # Remove prefix and suffix
            rest = f[len(f"feat_{probe}_"):]
            if rest.endswith("_train.npz"):
                ds = rest[:-len("_train.npz")]
            elif rest.endswith("_test.npz"):
                continue  # skip test files
            elif rest.endswith("_wl.npz"):
                continue  # skip weighted label variants
            elif rest.endswith(".npz"):
                ds = rest[:-len(".npz")]
            else:
                continue
            all_ds.add(ds)

    # Load all features
    feats = {}
    for ds in sorted(all_ds):
        H, y = load(probe, ds, "train")
        if H is not None:
            feats[ds] = H

    avail = sorted(feats.keys())
    M = len(avail)
    print(f"\n  Discovered M={M} datasets:")
    rc = {}
    for d in avail:
        rc[d] = reff(feats[d])
        print(f"    {d:>25s}: shape={feats[d].shape}, r_eff={rc[d]:.1f}")

    if M < 10:
        print(f"\n  Only M={M} datasets found. Need at least 10 for meaningful compression.")
        print("  Wait for extract_features_m50.py to complete, then re-run.")
        sys.exit(1)

    # Compute full pool r_eff
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"\n  r_eff(full pool, M={M}) = {reff_full:.1f}")

    # Compression at various k values
    K_VALUES = [k for k in [3, 5, 7, 10, 14, 20] if k < M]
    N_SEEDS = 10

    all_rows = []

    for k in K_VALUES:
        ratio = M / k
        print(f"\n  {'='*60}")
        print(f"  k={k} (compression {ratio:.1f}x from M={M})")
        print(f"  {'='*60}")

        # Collapse Greedy
        t0 = time.time()
        sel_collapse = collapse_greedy(feats, rc, avail, k)
        t_collapse = time.time() - t0
        H_c = np.concatenate([feats[d] for d in sel_collapse])
        r_c = reff(H_c)
        ret_c = r_c / reff_full

        # r_sum Greedy
        sel_rsum = rsum_greedy(feats, rc, avail, k)
        H_r = np.concatenate([feats[d] for d in sel_rsum])
        r_r = reff(H_r)
        ret_r = r_r / reff_full

        # Random (10 seeds)
        rand_rets = []
        for seed in range(1, N_SEEDS + 1):
            sel_rand = random_select(avail, k, seed)
            H_rand = np.concatenate([feats[d] for d in sel_rand])
            r_rand = reff(H_rand)
            rand_rets.append(r_rand / reff_full)

        print(f"    Collapse : ret={ret_c*100:.1f}% (r_eff={r_c:.1f}, {t_collapse:.1f}s)")
        print(f"      selected: {sel_collapse}")
        print(f"    r_sum    : ret={ret_r*100:.1f}% (r_eff={r_r:.1f})")
        print(f"      selected: {sel_rsum}")
        print(f"    Random   : ret={np.mean(rand_rets)*100:.1f}% +/- {np.std(rand_rets)*100:.1f}%")

        delta_collapse_random = (ret_c - np.mean(rand_rets)) * 100
        delta_collapse_rsum = (ret_c - ret_r) * 100

        all_rows.append({
            "M": M, "k": k, "ratio": ratio,
            "collapse_reff": r_c, "collapse_ret": ret_c,
            "collapse_time": t_collapse,
            "collapse_selected": str(sel_collapse),
            "rsum_reff": r_r, "rsum_ret": ret_r,
            "random_ret_mean": np.mean(rand_rets),
            "random_ret_std": np.std(rand_rets),
            "reff_full": reff_full,
            "delta_collapse_random_pp": delta_collapse_random,
            "delta_collapse_rsum_pp": delta_collapse_rsum,
        })

        print(f"    Delta(Collapse - Random) = {delta_collapse_random:+.1f}pp")
        print(f"    Delta(Collapse - r_sum)  = {delta_collapse_rsum:+.1f}pp")

    # Save results
    out_csv = os.path.join(RESULTS, "e8_scale_m50_compression.csv")
    if all_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Saved {len(all_rows)} rows to {out_csv}")

    # Grand summary table
    print(f"\n  {'='*72}")
    print(f"  GRAND SUMMARY: M={M} datasets, ResNet-50")
    print(f"  {'='*72}")
    print(f"  {'k':>4s} {'ratio':>6s} {'Collapse':>10s} {'r_sum':>10s} {'Random':>15s} {'Δ(C-R)':>8s}")
    print(f"  {'-'*4:>4s} {'-'*6:>6s} {'-'*10:>10s} {'-'*10:>10s} {'-'*15:>15s} {'-'*8:>8s}")
    for row in all_rows:
        print(f"  {row['k']:>4d} {row['ratio']:>6.1f}x {row['collapse_ret']*100:>9.1f}% "
              f"{row['rsum_ret']*100:>9.1f}% "
              f"{row['random_ret_mean']*100:>6.1f}±{row['random_ret_std']*100:>4.1f}% "
              f"{row['delta_collapse_random_pp']:>+7.1f}pp")

    print("\n  Done.")
