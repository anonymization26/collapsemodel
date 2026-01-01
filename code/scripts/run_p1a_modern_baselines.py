"""
P1-A Modern Baseline Comparison — Vendi-Greedy / DomainDiverse / Centroid-FF

Adds three modern baselines beyond the FacLoc / k-Medoids / r_sum / Max-r_eff
already in E9, on the same M=43 ResNet-50 pool used in the paper's Table 4
(scaling stress test). Evaluates compression on:
  - Diversity retention (r_eff^sel / r_eff^full)
  - Fair LP via spectral subspace projection (Option A, same as E9)

Strategies added:
  (a) Vendi-Greedy           — greedy max of Vendi Score (squared-σ entropy
                                kernel from Friedman & Dieng 2023). Distinct
                                from r_eff (linear-σ entropy) due to spectral
                                weighting.
  (b) DomainDiverse-Greedy   — metadata baseline: pick from least-represented
                                domain (config.DATASET_DOMAIN), tiebreak on
                                individual r_eff.
  (c) Centroid-FF (DataComp-adapter) — farthest-first traversal in
                                feature-centroid space (mean L2-normed
                                feature per dataset). Adapts DataComp's
                                similarity-based filtering to dataset level.

Reuses E9's loading and fair-LP-projection logic verbatim so results are
directly comparable to the paper's existing numbers.

Run:
    cd collapse-model-repro
    nohup python scripts/run_p1a_modern_baselines.py > p1a.log 2>&1 &
    tail -f p1a.log

Output:
    results/p1a_modern_baselines.csv  (one row per (k, strategy))
    Console summary table
"""
import os, sys, math, time, warnings, glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FEAT_DIR = os.path.join(RESULTS, "p0_features")
SEED = 42
K_SA = 20
N_SAMPLES = 2000

# Domain metadata (mirrors config.DATASET_DOMAIN; copied here so the script
# is self-contained and runnable on a fresh checkout).
DATASET_DOMAIN = {
    # classic
    "mnist":         "handwritten",
    "kmnist":        "handwritten",
    "fashion_mnist": "handwritten",
    "fashionmnist":  "handwritten",
    "cifar10":       "natural",
    "cifar100":      "natural",
    "stl10":         "natural",
    "svhn":          "digit_scene",
    "eurosat":       "satellite",
    "oxford_pets":   "finegrained",
    "oxfordpets":    "finegrained",
    "flowers102":    "finegrained",
    "food101":       "finegrained",
    "dtd":           "texture",
    # medical (MedMNIST family)
    "bloodmnist":    "medical",
    "dermamnist":    "medical",
    "pathmnist":     "medical",
    "octmnist":      "medical",
    "pneumoniamnist": "medical",
    "breastmnist":   "medical",
    "retinamnist":   "medical",
    "tissuemnist":   "medical",
    "organamnist":   "medical",
    "organcmnist":   "medical",
    "organsmnist":   "medical",
    # ImageNet super-categories (semantic groups)
    "imagenet_animal":   "inet_animal",
    "imagenet_artifact": "inet_object",
    "imagenet_food":     "inet_food",
    "imagenet_scene":    "inet_scene",
    "imagenet_vehicle":  "inet_object",
    # ImageNet fine-grained subgroups
    "inet_apparel":       "inet_object",
    "inet_electronic":    "inet_object",
    "inet_furniture":     "inet_object",
    "inet_kitchen":       "inet_object",
    "inet_music_weapon":  "inet_object",
    "inet_sport_equip":   "inet_object",
    "inet_structure":     "inet_scene",
    "inet_vehicle_a":     "inet_object",
    "inet_vehicle_b":     "inet_object",
    "inet_arthropod":     "inet_animal",
    "inet_bird":          "inet_animal",
    "inet_fish":          "inet_animal",
    "inet_reptile":       "inet_animal",
    "inet_large_mammal":  "inet_animal",
    "inet_small_mammal":  "inet_animal",
    "inet_sporting_dog":  "inet_animal",
    "inet_terrier":       "inet_animal",
    "inet_food_fruit":    "inet_food",
    "inet_geology":       "inet_scene",
}

# ─── feature loading (verbatim from E9) ─────────────────────────────────
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

# ─── core metrics ───────────────────────────────────────────────────────
def reff(H):
    """Effective rank using LINEAR singular values (paper's r_eff)."""
    N, d = H.shape
    G = (H @ H.T) / N if N <= d else (H.T @ H) / N
    ev = np.linalg.eigvalsh(G)[::-1]; ev = ev[ev > 1e-10]
    s = np.sqrt(ev); p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def vendi(H):
    """Vendi Score on linear kernel K = (1/N) H H^T:
    eigvals(K) = σ_i^2 / N (for L2-normed rows). Vendi uses SQUARED σ."""
    N, d = H.shape
    G = (H @ H.T) / N if N <= d else (H.T @ H) / N
    ev = np.linalg.eigvalsh(G)[::-1]; ev = ev[ev > 1e-10]
    # ev are σ^2/N; normalize to probability for entropy
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def sa_k(HA, HB, k=K_SA):
    _, _, VtA = np.linalg.svd(HA, full_matrices=False)
    _, _, VtB = np.linalg.svd(HB, full_matrices=False)
    ke = min(k, VtA.shape[0], VtB.shape[0])
    cs = np.linalg.svd(VtA[:ke] @ VtB[:ke].T, compute_uv=False)
    return float(np.mean(np.clip(cs, 0, 1) ** 2))

def cpred(rA, rB, gamma, alpha):
    rd = rA if gamma <= 1 else rB; rs = rB if gamma <= 1 else rA
    A = 1 + gamma**2; D = math.sqrt(max((1-gamma**2)**2 + 4*gamma**2*alpha, 0))
    cp = math.sqrt((A+D)/2); cm = math.sqrt(max((A-D)/2, 1e-30))
    r = max(min(cp/(cp+cm), 1-1e-15), 1e-15)
    Hb = -r*math.log(r) - (1-r)*math.log(1-r)
    return math.exp(Hb + r*math.log(max(rd, 1e-10)) + (1-r)*math.log(max(rs, 1e-10)))

# ─── strategies ─────────────────────────────────────────────────────────
def collapse_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds): return list(all_ds)
    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]; H_cur = feats[seed].copy(); r_cur = rc[seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_pred, best_d = -1, None
        nA = np.linalg.svd(H_cur, compute_uv=False).sum()
        for d in remaining:
            alpha = sa_k(H_cur, feats[d])
            nB = np.linalg.svd(feats[d], compute_uv=False).sum()
            pred = cpred(r_cur, rc[d], nB/(nA+1e-8), alpha)
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

# === NEW: P1-A baselines ================================================
def vendi_greedy(feats, all_ds, k):
    """Greedy maximization of Vendi Score on merged features.
    Distinct from r_eff: uses squared singular values, weighting top
    components more heavily.
    """
    if k >= len(all_ds): return list(all_ds)
    vc = {d: vendi(feats[d]) for d in all_ds}
    seed = max(all_ds, key=lambda d: vc[d])
    selected = [seed]
    H_cur = feats[seed].copy()
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_v, best_d = -np.inf, None
        for d in remaining:
            v = vendi(np.concatenate([H_cur, feats[d]]))
            if v > best_v:
                best_v, best_d = v, d
        if best_d is None: break
        selected.append(best_d); remaining.discard(best_d)
        H_cur = np.concatenate([H_cur, feats[best_d]])
    return selected

def domain_diverse_greedy(feats, rc, all_ds, k, domain_map=DATASET_DOMAIN):
    """Metadata baseline: greedily pick from the least-represented domain.
    Tiebreak by highest individual r_eff. When all domains tied, pick
    highest r_eff.
    """
    if k >= len(all_ds): return list(all_ds)
    selected = []
    remaining = list(all_ds)
    while len(selected) < k and remaining:
        dom_counts = {}
        for sd in selected:
            dom = domain_map.get(sd, "unknown")
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
        # tuple key: (current freq of this candidate's domain, -r_eff)
        # min picks: least-represented domain first, then highest r_eff
        def key(d):
            dom = domain_map.get(d, "unknown")
            return (dom_counts.get(dom, 0), -rc[d])
        pick = min(remaining, key=key)
        selected.append(pick)
        remaining.remove(pick)
    return selected

def centroid_ff_greedy(feats, all_ds, k):
    """DataComp-style adapter: farthest-first traversal in dataset-centroid
    space. Each dataset is summarized by the L2-normed mean of its features
    (a "prototype"), and selection maximizes minimum cosine distance to
    already-selected prototypes.
    """
    if k >= len(all_ds): return list(all_ds)
    cents = {}
    for d in all_ds:
        c = feats[d].mean(0)
        n = np.linalg.norm(c)
        cents[d] = c / max(n, 1e-10)
    pool_mean = np.mean(list(cents.values()), axis=0)
    pool_mean = pool_mean / max(np.linalg.norm(pool_mean), 1e-10)
    # seed: farthest from pool centroid
    seed = max(all_ds, key=lambda d: np.linalg.norm(cents[d] - pool_mean))
    selected = [seed]
    remaining = set(all_ds) - {seed}
    while len(selected) < k:
        best_d, best_dist = None, -np.inf
        for d in remaining:
            mind = min(np.linalg.norm(cents[d] - cents[s]) for s in selected)
            if mind > best_dist:
                best_dist, best_d = mind, d
        if best_d is None: break
        selected.append(best_d); remaining.discard(best_d)
    return selected

# ─── fair LP via spectral subspace projection (Option A, from E9) ───────
def fair_lp_subspace(H_train, y_train, H_test, y_test, V_r):
    try:
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

# ─── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 76)
    print("  P1-A: Modern baseline comparison")
    print("       (Vendi-Greedy / DomainDiverse / Centroid-FF)")
    print("=" * 76, flush=True)

    PROBE = "resnet50"
    K_VALUES = [3, 5, 7, 10]
    N_SEEDS = 10

    train_paths, test_paths = discover_datasets(PROBE)
    feats, labels = {}, {}
    for name, path in train_paths.items():
        try:
            H, y = load_from(path); feats[name] = H
            if y is not None: labels[name] = y
        except Exception as e:
            print(f"  ! load {name}: {e}")

    feats_test, labels_test = {}, {}
    for name, path in test_paths.items():
        if name in feats:
            try:
                Ht, yt = load_from(path)
                if yt is not None:
                    feats_test[name] = Ht
                    labels_test[name] = yt
            except Exception:
                pass

    avail = sorted(feats.keys())
    M = len(avail)
    targets_lp = [d for d in avail if d in feats_test and d in labels_test]
    print(f"  M={M} datasets; LP targets ({len(targets_lp)}): {targets_lp}", flush=True)

    rc = {d: reff(feats[d]) for d in avail}
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"  r_eff(full M={M} pool) = {reff_full:.1f}", flush=True)

    # Per-dataset domain coverage report
    dom_present = {}
    for d in avail:
        dom = DATASET_DOMAIN.get(d, "unknown")
        dom_present.setdefault(dom, []).append(d)
    print("  Domain coverage:")
    for dom, dlist in sorted(dom_present.items()):
        print(f"    {dom:>14s}: {len(dlist)}  ({', '.join(dlist[:5])}{'...' if len(dlist)>5 else ''})")
    flush = sys.stdout.flush; flush()

    all_rows = []
    for k in K_VALUES:
        if k >= M: continue
        ratio = M / k
        print(f"\n  ═══ k={k} ({ratio:.1f}x compression) ═══", flush=True)

        strategies = [
            ("Collapse",       lambda: collapse_greedy(feats, rc, avail, k)),
            ("Vendi_greedy",   lambda: vendi_greedy(feats, avail, k)),
            ("DomainDiverse",  lambda: domain_diverse_greedy(feats, rc, avail, k)),
            ("Centroid_FF",    lambda: centroid_ff_greedy(feats, avail, k)),
            ("r_sum",          lambda: rsum_greedy(feats, rc, avail, k)),
            ("Max_reff",       lambda: maxreff_greedy(rc, avail, k)),
        ]
        for seed in range(1, N_SEEDS+1):
            strategies.append((f"Random_s{seed}", lambda s=seed: random_select(avail, k, s)))

        for strat_name, strat_fn in strategies:
            t0 = time.time()
            try:
                selected = strat_fn()
            except Exception as e:
                print(f"    ! {strat_name} failed: {e}"); continue
            H_sel = np.concatenate([feats[d] for d in selected])
            r_sel = reff(H_sel)
            v_sel = vendi(H_sel)
            retention = r_sel / reff_full * 100

            r_rank = max(int(round(r_sel)), 1)
            _, _, Vt_sel = np.linalg.svd(H_sel, full_matrices=False)
            V_r = Vt_sel[:r_rank].T

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
                "reff_sel": r_sel, "vendi_sel": v_sel,
                "retention_pct": retention,
                "r_rank": r_rank, "lp_mean": mean_lp,
                "n_lp_targets": len(lp_scores),
                "elapsed_s": elapsed,
                "selected": "|".join(selected),
            })
            print(f"    {strat_name:18s}: ret={retention:6.1f}%  "
                  f"vendi={v_sel:6.1f}  LP={mean_lp:.4f}  ({elapsed:.1f}s)", flush=True)

        # per-k summary
        col   = next((r for r in all_rows if r["k"]==k and r["strategy"]=="Collapse"), None)
        vendi_r = next((r for r in all_rows if r["k"]==k and r["strategy"]=="Vendi_greedy"), None)
        domd  = next((r for r in all_rows if r["k"]==k and r["strategy"]=="DomainDiverse"), None)
        cff   = next((r for r in all_rows if r["k"]==k and r["strategy"]=="Centroid_FF"), None)
        rands = [r for r in all_rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_lp = np.array([r["lp_mean"] for r in rands if not np.isnan(r["lp_mean"])])
        rand_ret = np.array([r["retention_pct"] for r in rands])

        print(f"\n  ── k={k} summary ──")
        for nm, r in [("Collapse", col), ("Vendi_greedy", vendi_r),
                      ("DomainDiverse", domd), ("Centroid_FF", cff)]:
            if r is None: continue
            d_lp = r["lp_mean"] - rand_lp.mean()
            d_ret = r["retention_pct"] - rand_ret.mean()
            print(f"    {nm:14s}: ret {r['retention_pct']:.1f}%  "
                  f"(Δ vs Rand {d_ret:+.1f})   "
                  f"LP {r['lp_mean']:.4f}  (ΔLP {d_lp:+.4f})")
        print(f"    {'Random (mean±sd)':14s}: ret {rand_ret.mean():.1f}±{rand_ret.std():.1f}%   "
              f"LP {rand_lp.mean():.4f}±{rand_lp.std():.4f}", flush=True)

    # save full results
    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS, "p1a_modern_baselines.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved {len(df)} rows to {out}")

    # final summary table for paper
    print(f"\n{'='*76}")
    print("  FINAL TABLE (paper-ready)")
    print(f"{'='*76}")
    print(f"  {'k':>3s}  {'Strategy':18s}  {'Retention %':>11s}  {'LP':>6s}  {'ΔLP vs Rand':>11s}")
    for k in K_VALUES:
        if k >= M: continue
        rands = [r for r in all_rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_lp = np.array([r["lp_mean"] for r in rands if not np.isnan(r["lp_mean"])])
        rand_ret = np.array([r["retention_pct"] for r in rands])
        for sname in ["Collapse", "Vendi_greedy", "DomainDiverse", "Centroid_FF",
                      "r_sum", "Max_reff"]:
            r = next((x for x in all_rows if x["k"]==k and x["strategy"]==sname), None)
            if r is None: continue
            d_lp = r["lp_mean"] - rand_lp.mean()
            print(f"  {k:>3d}  {sname:18s}  {r['retention_pct']:>10.1f}   "
                  f"{r['lp_mean']:>.4f}  {d_lp:>+9.4f}")
        print(f"  {k:>3d}  {'Random (mean±sd)':18s}  {rand_ret.mean():>5.1f}±{rand_ret.std():>3.1f}   "
              f"{rand_lp.mean():.4f}±{rand_lp.std():.4f}")
        print()

    # save JSON summary for paper integration
    summary = {}
    for k in K_VALUES:
        if k >= M: continue
        rands = [r for r in all_rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_lp = np.array([r["lp_mean"] for r in rands if not np.isnan(r["lp_mean"])])
        rand_ret = np.array([r["retention_pct"] for r in rands])
        d = {"random_lp_mean": float(rand_lp.mean()),
             "random_lp_std":  float(rand_lp.std()),
             "random_ret_mean": float(rand_ret.mean()),
             "random_ret_std":  float(rand_ret.std())}
        for sname in ["Collapse", "Vendi_greedy", "DomainDiverse", "Centroid_FF",
                      "r_sum", "Max_reff"]:
            r = next((x for x in all_rows if x["k"]==k and x["strategy"]==sname), None)
            if r is None: continue
            d[sname] = {"retention_pct": r["retention_pct"],
                        "lp_mean": r["lp_mean"],
                        "delta_lp_vs_rand": r["lp_mean"] - rand_lp.mean(),
                        "selected": r["selected"]}
        summary[f"k={k}"] = d
    with open(os.path.join(RESULTS, "p1a_modern_baselines_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved JSON summary to {RESULTS}/p1a_modern_baselines_summary.json")
    print("\n  Done.")
