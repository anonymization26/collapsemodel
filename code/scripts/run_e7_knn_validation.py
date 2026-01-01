"""
E7: k-NN Downstream Validation
Part A: Compression scenario -- compare Collapse/r_sum/Random using k-NN accuracy
Part B: r_eff vs k-NN correlation -- do probes with higher r_eff also give better k-NN?

Uses pre-computed features from results/p0_features/*.npz
"""
import os, sys, math, time, warnings, csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
FEAT_DIR = os.path.join(RESULTS, "p0_features")
SEED = 42
N_SAMPLES = 2000
K_SA = 20
KNN_K_VALUES = [5, 20]

# ============================================================
# Shared utilities (from run_compression_v2.py)
# ============================================================

def load(probe, ds, split="train"):
    """Load features with dual-path lookup."""
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


def reff(H):
    N, d = H.shape
    G = (H @ H.T) / N if N <= d else (H.T @ H) / N
    ev = np.linalg.eigvalsh(G)[::-1]
    ev = ev[ev > 1e-10]
    s = np.sqrt(ev)
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def sa_k(HA, HB, k=K_SA):
    _, _, VtA = np.linalg.svd(HA, full_matrices=False)
    _, _, VtB = np.linalg.svd(HB, full_matrices=False)
    ke = min(k, VtA.shape[0], VtB.shape[0])
    cs = np.linalg.svd(VtA[:ke] @ VtB[:ke].T, compute_uv=False)
    return float(np.mean(np.clip(cs, 0, 1) ** 2))


def cpred(rA, rB, gamma, alpha):
    rd = rA if gamma <= 1 else rB
    rs = rB if gamma <= 1 else rA
    A = 1 + gamma**2
    D = math.sqrt(max((1 - gamma**2)**2 + 4 * gamma**2 * alpha, 0))
    cp = math.sqrt((A + D) / 2)
    cm = math.sqrt(max((A - D) / 2, 1e-30))
    r = max(min(cp / (cp + cm), 1 - 1e-15), 1e-15)
    Hb = -r * math.log(r) - (1 - r) * math.log(1 - r)
    return math.exp(Hb + r * math.log(max(rd, 1e-10)) + (1 - r) * math.log(max(rs, 1e-10)))


# ============================================================
# Evaluation functions
# ============================================================

def eval_knn(H_train, y_train, H_test, y_test, k=5):
    """k-NN accuracy after StandardScaler."""
    sc = StandardScaler()
    Htr = sc.fit_transform(H_train)
    Hte = sc.transform(H_test)
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    clf.fit(Htr, y_train)
    preds = clf.predict(Hte)
    return float((preds == y_test).mean())


def eval_lp(H_train, y_train, H_test, y_test):
    """Linear probe accuracy after StandardScaler."""
    sc = StandardScaler()
    Htr = sc.fit_transform(H_train)
    Hte = sc.transform(H_test)
    clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1)
    clf.fit(Htr, y_train)
    preds = clf.predict(Hte)
    return float((preds == y_test).mean())


def eval_target_aug(feats, labels, target, H_test, y_test, subset, method="lp", knn_k=5):
    """
    Evaluate on target dataset using subset as augmenter pool.
    method: 'lp' for LogisticRegression, 'knn' for KNeighborsClassifier
    """
    H_t = feats[target]
    y_t = labels[target]
    nc_t = len(np.unique(y_t))
    H_parts = [H_t]
    y_parts = [y_t.astype(np.int64)]
    offset = nc_t
    for d in subset:
        if d == target:
            continue
        if d in labels:
            H_parts.append(feats[d])
            y_parts.append(labels[d].astype(np.int64) + offset)
            offset += len(np.unique(labels[d]))
    H_train = np.concatenate(H_parts)
    y_train = np.concatenate(y_parts)
    if method == "lp":
        return eval_lp(H_train, y_train, H_test, y_test)
    else:
        return eval_knn(H_train, y_train, H_test, y_test, k=knn_k)


# ============================================================
# Compression strategies (from run_compression_v2.py)
# ============================================================

def collapse_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds):
        return list(all_ds)
    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]
    H_cur = feats[seed].copy()
    r_cur = rc[seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_pred, best_d = -1, None
        nA = np.linalg.svd(H_cur, compute_uv=False).sum()
        for d in remaining:
            alpha = sa_k(H_cur, feats[d])
            nB = np.linalg.svd(feats[d], compute_uv=False).sum()
            pred = cpred(r_cur, rc[d], nB / (nA + 1e-8), alpha)
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
# Part B: r_eff vs k-NN correlation (11 probes x 5 datasets)
# ============================================================

def run_part_b():
    print("\n" + "=" * 72)
    print("  E7 Part B: r_eff vs k-NN / LP Correlation")
    print("  (11 probes x 5 datasets, single-dataset evaluation)")
    print("=" * 72)

    PROBES = [
        "resnet50", "resnet101", "vit_b16", "deit_b", "swin_tiny",
        "beit_base", "convnext_tiny", "densenet121", "efficientnet_b4",
        "maxvit_tiny", "regnet_y",
    ]
    DATASETS = ["cifar10", "cifar100", "stl10", "svhn", "fashion_mnist"]

    rows = []
    for probe in PROBES:
        for ds in DATASETS:
            H_train, y_train = load(probe, ds, "train")
            H_test, y_test = load(probe, ds, "test")
            if H_train is None or H_test is None or y_train is None or y_test is None:
                print(f"  SKIP {probe}/{ds}: missing features")
                continue

            r = reff(H_train)
            lp_acc = eval_lp(H_train, y_train, H_test, y_test)

            knn_accs = {}
            for kk in KNN_K_VALUES:
                knn_accs[kk] = eval_knn(H_train, y_train, H_test, y_test, k=kk)

            row = {
                "probe": probe,
                "dataset": ds,
                "reff": r,
                "lp_acc": lp_acc,
            }
            for kk in KNN_K_VALUES:
                row[f"knn_{kk}_acc"] = knn_accs[kk]
            rows.append(row)
            print(f"  {probe:>20s} / {ds:<15s}  r_eff={r:6.1f}  "
                  f"LP={lp_acc:.4f}  kNN-5={knn_accs[5]:.4f}  kNN-20={knn_accs[20]:.4f}")

    if not rows:
        print("  No data!")
        return

    # Save raw data
    out_csv = os.path.join(RESULTS, "e7_partB_reff_knn_correlation.csv")
    fieldnames = ["probe", "dataset", "reff", "lp_acc"] + [f"knn_{k}_acc" for k in KNN_K_VALUES]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved {len(rows)} rows to {out_csv}")

    # Compute correlations
    r_effs = [r["reff"] for r in rows]
    lp_accs = [r["lp_acc"] for r in rows]

    print(f"\n  === Correlation Summary (n={len(rows)}) ===")
    rho_lp, p_lp = spearmanr(r_effs, lp_accs)
    r_lp, pr_lp = pearsonr(r_effs, lp_accs)
    print(f"  r_eff vs LP:     Spearman rho={rho_lp:.4f} (p={p_lp:.2e})  Pearson r={r_lp:.4f} (p={pr_lp:.2e})")

    for kk in KNN_K_VALUES:
        knn_accs = [r[f"knn_{kk}_acc"] for r in rows]
        rho, p_val = spearmanr(r_effs, knn_accs)
        r_val, pr_val = pearsonr(r_effs, knn_accs)
        print(f"  r_eff vs kNN-{kk}:  Spearman rho={rho:.4f} (p={p_val:.2e})  Pearson r={r_val:.4f} (p={pr_val:.2e})")

    # Per-dataset correlations
    print(f"\n  === Per-dataset Correlations ===")
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if len(ds_rows) < 4:
            continue
        re = [r["reff"] for r in ds_rows]
        la = [r["lp_acc"] for r in ds_rows]
        ka5 = [r["knn_5_acc"] for r in ds_rows]
        ka20 = [r["knn_20_acc"] for r in ds_rows]
        rho_lp_ds, _ = spearmanr(re, la)
        rho_k5_ds, _ = spearmanr(re, ka5)
        rho_k20_ds, _ = spearmanr(re, ka20)
        print(f"  {ds:<15s}  rho(r_eff,LP)={rho_lp_ds:.3f}  "
              f"rho(r_eff,kNN-5)={rho_k5_ds:.3f}  rho(r_eff,kNN-20)={rho_k20_ds:.3f}")

    # LP vs kNN correlation (how similar are the two downstream metrics?)
    print(f"\n  === LP vs kNN Correlation ===")
    for kk in KNN_K_VALUES:
        knn_accs = [r[f"knn_{kk}_acc"] for r in rows]
        rho, p_val = spearmanr(lp_accs, knn_accs)
        print(f"  LP vs kNN-{kk}:  Spearman rho={rho:.4f} (p={p_val:.2e})")

    return rows


# ============================================================
# Part A: Compression scenario with k-NN
# ============================================================

def run_part_a():
    print("\n" + "=" * 72)
    print("  E7 Part A: Compression with k-NN Validation")
    print("  (ResNet-50 pool, LP + kNN evaluation)")
    print("=" * 72)

    # Use all ResNet-50 datasets with train+test available locally
    CANDIDATE_DS = [
        "cifar10", "cifar100", "stl10", "svhn", "fashion_mnist", "dtd",
        "mnist", "bloodmnist", "dermamnist", "pathmnist",
        "imagenet_animal", "imagenet_artifact", "imagenet_food",
        "imagenet_scene", "imagenet_vehicle",
    ]

    probe = "resnet50"
    feats, labels = {}, {}
    feats_test, labels_test = {}, {}

    for ds in CANDIDATE_DS:
        H, y = load(probe, ds, "train")
        if H is not None and y is not None:
            feats[ds] = H
            labels[ds] = y
            Ht, yt = load(probe, ds, "test")
            if Ht is not None and yt is not None:
                feats_test[ds] = Ht
                labels_test[ds] = yt

    avail = sorted(feats.keys())
    M = len(avail)
    print(f"  Available datasets: {M}")
    for d in avail:
        has_test = "test" if d in feats_test else "train-only"
        print(f"    {d}: train={feats[d].shape}, {has_test}")

    if M < 4:
        print("  Too few datasets for meaningful compression. Need SCP from remote.")
        return []

    # Compute r_eff for each dataset
    rc = {d: reff(feats[d]) for d in avail}
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"\n  r_eff(full pool, M={M}): {reff_full:.1f}")

    targets = [d for d in feats_test if d in feats]
    print(f"  Targets for downstream eval: {targets}")

    K_VALUES = [k for k in [3, 5, 7] if k < M]
    N_SEEDS = 10

    all_rows = []

    for k in K_VALUES:
        ratio = M / k
        print(f"\n  === k={k} (compression {ratio:.1f}x) ===")

        strategies = [
            ("Collapse", lambda: collapse_greedy(feats, rc, avail, k)),
            ("r_sum", lambda: rsum_greedy(feats, rc, avail, k)),
        ]
        for seed in range(1, N_SEEDS + 1):
            strategies.append((f"Random_s{seed}", lambda s=seed: random_select(avail, k, s)))

        for strat_name, strat_fn in strategies:
            t0 = time.time()
            selected = strat_fn()
            select_t = time.time() - t0

            H_sel = np.concatenate([feats[d] for d in selected])
            r_sel = reff(H_sel)
            retention = r_sel / reff_full

            # Evaluate on each target: LP + kNN
            lp_scores, knn5_scores, knn20_scores = [], [], []
            for t in targets:
                lp = eval_target_aug(feats, labels, t, feats_test[t], labels_test[t],
                                     selected, method="lp")
                if lp is not None:
                    lp_scores.append(lp)

                for kk in KNN_K_VALUES:
                    knn = eval_target_aug(feats, labels, t, feats_test[t], labels_test[t],
                                          selected, method="knn", knn_k=kk)
                    if knn is not None:
                        if kk == 5:
                            knn5_scores.append(knn)
                        else:
                            knn20_scores.append(knn)

            row = {
                "probe": probe, "k": k, "M": M,
                "compression_ratio": ratio,
                "strategy": strat_name,
                "selected": str(selected),
                "reff_sel": r_sel, "reff_full": reff_full,
                "retention": retention,
                "lp_mean": np.mean(lp_scores) if lp_scores else np.nan,
                "knn5_mean": np.mean(knn5_scores) if knn5_scores else np.nan,
                "knn20_mean": np.mean(knn20_scores) if knn20_scores else np.nan,
                "n_targets": len(targets),
                "select_time": select_t,
            }
            all_rows.append(row)

        # Summary for this k
        c_row = [r for r in all_rows if r["k"] == k and r["strategy"] == "Collapse"]
        rs_row = [r for r in all_rows if r["k"] == k and r["strategy"] == "r_sum"]
        rand_rows = [r for r in all_rows if r["k"] == k and r["strategy"].startswith("Random")]

        if c_row and rs_row:
            c_row = c_row[0]
            rs_row = rs_row[0]
            rand_ret = [r["retention"] for r in rand_rows]
            rand_lp = [r["lp_mean"] for r in rand_rows if not np.isnan(r["lp_mean"])]
            rand_k5 = [r["knn5_mean"] for r in rand_rows if not np.isnan(r["knn5_mean"])]
            rand_k20 = [r["knn20_mean"] for r in rand_rows if not np.isnan(r["knn20_mean"])]

            print(f"    Collapse : ret={c_row['retention']*100:.1f}%  "
                  f"LP={c_row['lp_mean']:.4f}  kNN-5={c_row['knn5_mean']:.4f}  kNN-20={c_row['knn20_mean']:.4f}")
            print(f"    r_sum    : ret={rs_row['retention']*100:.1f}%  "
                  f"LP={rs_row['lp_mean']:.4f}  kNN-5={rs_row['knn5_mean']:.4f}  kNN-20={rs_row['knn20_mean']:.4f}")
            if rand_lp:
                print(f"    Random   : ret={np.mean(rand_ret)*100:.1f}%  "
                      f"LP={np.mean(rand_lp):.4f}  kNN-5={np.mean(rand_k5):.4f}  kNN-20={np.mean(rand_k20):.4f}")

    # Save
    if all_rows:
        out_csv = os.path.join(RESULTS, "e7_partA_compression_knn.csv")
        fieldnames = list(all_rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Saved {len(all_rows)} rows to {out_csv}")

    return all_rows


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="E7: k-NN Downstream Validation")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all",
                        help="Which part to run: a (compression), b (correlation), all")
    args = parser.parse_args()

    if args.part in ("b", "all"):
        run_part_b()

    if args.part in ("a", "all"):
        run_part_a()

    print("\n  E7 complete.")
