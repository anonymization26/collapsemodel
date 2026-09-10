"""
E9: Fair LP experiment (Option A) — Spectral Subspace Projection.

Tests: "Does compressed pool's top-r principal subspace preserve
target-relevant discriminative directions?"

For each target T and each strategy's k-subset S:
1. Concat S features -> M_S (stacked)
2. SVD -> top-r right singular vectors V_S (d, r)  [r = round(reff(M_S))]
3. Project target: H_T_proj = H_T @ V_S @ V_S.T  (rank-r reconstruction)
4. Train LR on (H_T_train_proj, y_T_train)
5. Test: LP accuracy on H_T_test_proj

This eliminates distractor-class penalty because LR trains only on T's classes.
Directly tests whether the spectral structure of compressed pool preserves
target-discriminative information.
"""
import os, sys, time, warnings, glob
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from metrics.collapse_core import (
    collapse_4s_predict as cpred,
    effective_rank as reff,
    nuclear_mass,
)
from metrics.subspace_alignment import compute_subspace_alignment

warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FEAT_DIR = os.path.join(RESULTS, "p0_features")
SEED = 42; K_SA = 20; N_SAMPLES = 2000

def l2n_sub(H, y=None):
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H = H / np.maximum(norms, 1e-10)
    rng = np.random.default_rng(SEED)
    if len(H) > N_SAMPLES:
        idx = rng.choice(len(H), N_SAMPLES, replace=False)
        H = H[idx]
        if y is not None: y = y[idx]
    return H, y

def discover_datasets(probe):
    train_paths, test_paths = {}, {}
    for f in glob.glob(os.path.join(FEAT_DIR, f"{probe}_*_train.npz")):
        n = os.path.basename(f).replace(f"{probe}_","").replace("_train.npz","")
        train_paths[n] = f
    for f in glob.glob(os.path.join(FEAT_DIR, f"{probe}_*_test.npz")):
        n = os.path.basename(f).replace(f"{probe}_","").replace("_test.npz","")
        test_paths[n] = f
    for f in glob.glob(os.path.join(RESULTS, f"feat_{probe}_*.npz")):
        b = os.path.basename(f).replace(f"feat_{probe}_","").replace(".npz","")
        if b.endswith("_wl"): continue
        if b.endswith("_test"):
            test_paths.setdefault(b[:-5], f)
        elif b.endswith("_train"):
            train_paths.setdefault(b[:-6], f)
        else:
            train_paths.setdefault(b, f)
    return train_paths, test_paths

def load_from(path):
    d = np.load(path)
    H = d["H"].astype(np.float64)
    y = d["y"].astype(np.int32) if "y" in d else None
    return l2n_sub(H, y)

def sa_k(HA, HB, k=K_SA):
    return float(compute_subspace_alignment(HA, HB, k=k)["sa_k"])

def collapse_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds): return list(all_ds)
    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]; H_cur = feats[seed].copy(); r_cur = rc[seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_pred, best_d = -1, None
        nA = nuclear_mass(H_cur)
        for d in remaining:
            alpha = sa_k(H_cur, feats[d])
            nB = nuclear_mass(feats[d])
            pred = cpred(r_cur, rc[d], nB / nA, alpha)
            if pred > best_pred: best_pred, best_d = pred, d
        if best_d is None: break
        selected.append(best_d); remaining.discard(best_d)
        H_cur = np.concatenate([H_cur, feats[best_d]])
        r_cur = reff(H_cur)
    return selected

def maxreff_greedy(rc, all_ds, k):
    return sorted(all_ds, key=lambda d: rc[d], reverse=True)[:k]

def rsum_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds): return list(all_ds)
    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]; r_cur = rc[seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_s, best_d = -1, None
        for d in remaining:
            if r_cur + rc[d] > best_s: best_s, best_d = r_cur + rc[d], d
        if best_d is None: break
        selected.append(best_d); remaining.discard(best_d)
        r_cur = reff(np.concatenate([feats[sd] for sd in selected]))
    return selected

def random_select(all_ds, k, seed):
    rng = np.random.default_rng(seed)
    return list(rng.choice(sorted(all_ds), k, replace=False))

def fair_lp_subspace(H_train, y_train, H_test, y_test, V_r):
    """Fair LP: project target features onto subspace V_r (d × r), then LR.
    V_r: top-r right singular vectors from subset (d × r, columns are unit-norm)
    """
    try:
        # Project to rank-r subspace: P = V_r V_r^T (d × d rank r)
        # H_proj = H @ P  (still d-dim but rank r)
        H_train_p = H_train @ V_r @ V_r.T
        H_test_p = H_test @ V_r @ V_r.T
        sc = StandardScaler()
        Htr = sc.fit_transform(H_train_p)
        Hte = sc.transform(H_test_p)
        clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
        clf.fit(Htr, y_train)
        return float((clf.predict(Hte) == y_test).mean())
    except Exception as e:
        print(f"    LP err: {e}"); return None

if __name__ == "__main__":
    print("=" * 72)
    print("  E9: Fair LP via Spectral Subspace Projection (Option A)")
    print("=" * 72)

    PROBE = "resnet50"
    K_VALUES = [3, 5, 7, 10]
    N_SEEDS = 10

    train_paths, test_paths = discover_datasets(PROBE)
    feats, labels = {}, {}
    for name, path in train_paths.items():
        try:
            H, y = load_from(path)
            feats[name] = H
            if y is not None: labels[name] = y
        except: pass

    feats_test, labels_test = {}, {}
    for name, path in test_paths.items():
        if name in feats:
            try:
                Ht, yt = load_from(path)
                if yt is not None:
                    feats_test[name] = Ht
                    labels_test[name] = yt
            except: pass

    avail = sorted(feats.keys())
    M = len(avail)
    targets_lp = [d for d in avail if d in feats_test and d in labels_test]
    print(f"  M={M} datasets; LP targets: {targets_lp} ({len(targets_lp)})")

    rc = {d: reff(feats[d]) for d in avail}
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"  r_eff(full M={M} pool) = {reff_full:.1f}")

    # Baseline: target features projected onto full-pool top-r subspace
    print("  Computing full-pool SVD...", flush=True)
    t0 = time.time()
    _, _, Vt_full = np.linalg.svd(H_full, full_matrices=False)
    print(f"  Full SVD done ({time.time()-t0:.0f}s)")

    # Identity baseline: no projection (r = 2048 = full rank)
    # Full-pool baseline: top-r of full pool
    # Per-strategy: top-r of strategy's subset

    all_rows = []
    for k in K_VALUES:
        if k >= M: continue
        ratio = M / k
        print(f"\n  ═══ k={k} ({ratio:.1f}x compression) ═══", flush=True)

        strategies = [
            ("Collapse", lambda: collapse_greedy(feats, rc, avail, k)),
            ("r_sum",    lambda: rsum_greedy(feats, rc, avail, k)),
            ("Max_reff", lambda: maxreff_greedy(rc, avail, k)),
        ]
        for seed in range(1, N_SEEDS+1):
            strategies.append((f"Random_s{seed}", lambda s=seed: random_select(avail, k, s)))

        for strat_name, strat_fn in strategies:
            t0 = time.time()
            selected = strat_fn()
            H_sel = np.concatenate([feats[d] for d in selected])
            r_sel = reff(H_sel)
            retention = r_sel / reff_full * 100

            # Take top-r singular vectors where r = round(reff(H_sel))
            r_rank = max(int(round(r_sel)), 1)
            _, _, Vt_sel = np.linalg.svd(H_sel, full_matrices=False)
            V_r = Vt_sel[:r_rank].T  # (d, r) — Vt is (r, d)

            # Per-target LP via projection
            lp_scores = []
            for tgt in targets_lp:
                lp = fair_lp_subspace(
                    feats[tgt], labels[tgt],
                    feats_test[tgt], labels_test[tgt],
                    V_r)
                if lp is not None: lp_scores.append(lp)
            mean_lp = np.mean(lp_scores) if lp_scores else np.nan
            elapsed = time.time() - t0

            all_rows.append({
                "k": k, "ratio": ratio, "strategy": strat_name,
                "reff_sel": r_sel, "retention_pct": retention,
                "r_rank": r_rank, "lp_mean": mean_lp,
                "n_lp_targets": len(lp_scores),
                "elapsed_s": elapsed,
                "selected": "|".join(selected),
            })

        col = [r for r in all_rows if r["k"]==k and r["strategy"]=="Collapse"][0]
        rsum = [r for r in all_rows if r["k"]==k and r["strategy"]=="r_sum"][0]
        mx = [r for r in all_rows if r["k"]==k and r["strategy"]=="Max_reff"][0]
        rands = [r for r in all_rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_ret = np.array([r["retention_pct"] for r in rands])
        rand_lp = np.array([r["lp_mean"] for r in rands if not np.isnan(r["lp_mean"])])
        print(f"    Collapse : ret={col['retention_pct']:.1f}% r_rank={col['r_rank']} LP={col['lp_mean']:.4f}")
        print(f"    r_sum    : ret={rsum['retention_pct']:.1f}% LP={rsum['lp_mean']:.4f}")
        print(f"    Max_reff : ret={mx['retention_pct']:.1f}% LP={mx['lp_mean']:.4f}")
        print(f"    Random   : ret={rand_ret.mean():.1f}±{rand_ret.std():.1f}% LP={rand_lp.mean():.4f}±{rand_lp.std():.4f}")
        print(f"    ΔLP(Col-Rand)={col['lp_mean']-rand_lp.mean():+.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS, "e9_fair_lp_subspace.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved {len(df)} rows to {out}")

    print(f"\n{'='*72}\n  FINAL SUMMARY (Option A: spectral subspace projection)\n{'='*72}")
    for k in K_VALUES:
        if k >= M: continue
        col = [r for r in all_rows if r["k"]==k and r["strategy"]=="Collapse"][0]
        mx = [r for r in all_rows if r["k"]==k and r["strategy"]=="Max_reff"][0]
        rands = [r for r in all_rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_ret = np.array([r["retention_pct"] for r in rands])
        rand_lp = np.array([r["lp_mean"] for r in rands if not np.isnan(r["lp_mean"])])
        print(f"  k={k:>2d} ({col['ratio']:.1f}x, r_rank={col['r_rank']}): "
              f"Col=[ret {col['retention_pct']:.1f}% LP {col['lp_mean']:.4f}]  "
              f"Max=[ret {mx['retention_pct']:.1f}% LP {mx['lp_mean']:.4f}]  "
              f"Rand=[ret {rand_ret.mean():.1f}±{rand_ret.std():.1f}% LP {rand_lp.mean():.4f}±{rand_lp.std():.4f}]  "
              f"ΔLP={col['lp_mean']-rand_lp.mean():+.4f}")
    print("\n  Done.")
