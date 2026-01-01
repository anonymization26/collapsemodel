"""
E6 v2: Extended Compression Experiments
- A: ViT-B/16 on 11 vision datasets (cross-architecture validation)
- B: BERT/RoBERTa on 7 text datasets (cross-modality)
- D: Fixed LP logic (always computable)

LP design (fix D): For each target T ∈ {CIFAR-10, STL-10} (has test set):
- Use subset features as augmenter pool
- Merged train: T_train features + subset features (labels offset by dataset)
- Test: T_test features, predict T's classes only
- LP = proportion of target test samples correctly classified
- Always computable regardless of what strategy selected
"""
import os, sys, math, time, warnings, glob, csv
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FEAT_DIR = os.path.join(RESULTS, "p0_features")
TEXT_DIR = os.path.join(RESULTS, "text_features")
SEED = 42; K_SA = 20; N_SAMPLES = 2000

def load(probe, ds, split="train", base_dir=None):
    if base_dir is None:
        paths = [f"{FEAT_DIR}/{probe}_{ds}_{split}.npz",
                 f"{RESULTS}/feat_{probe}_{ds}.npz",
                 f"{RESULTS}/feat_{probe}_{ds}_{split}.npz"]
    else:
        paths = [f"{base_dir}/{probe}_{ds}.npz"]
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
                H, y = H[idx], y[idx] if y is not None else None
            return H, y
    return None, None

def reff(H):
    N, d = H.shape
    G = (H @ H.T) / N if N <= d else (H.T @ H) / N
    ev = np.linalg.eigvalsh(G)[::-1]; ev = ev[ev > 1e-10]
    s = np.sqrt(ev); p = s / s.sum()
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

def collapse_greedy(feats, rc, all_ds, k):
    if k >= len(all_ds): return list(all_ds)
    seed = max(all_ds, key=lambda d: rc[d])
    selected = [seed]
    H_cur = feats[seed].copy(); r_cur = rc[seed]
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

def rsum_greedy(feats, rc, all_ds, k):
    return sorted(all_ds, key=lambda d: rc[d], reverse=True)[:k]

def random_select(all_ds, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_ds), k, replace=False)
    return [all_ds[i] for i in idx]

def lp_target_aug(feats, labels, target, target_feats_test, target_labels_test, subset):
    """Fixed D: LP always computable.
    Train LR on (target_train features) + (subset features as augmentation),
    labels = target_labels + (target.num_classes offset for each subset dataset).
    Test on target_test features; evaluate accuracy on target's class range only.
    """
    try:
        # Target features and labels
        H_t = feats[target]; y_t = labels[target]
        nc_t = len(np.unique(y_t))
        # Augmenters
        H_parts = [H_t]; y_parts = [y_t.astype(np.int64)]
        offset = nc_t
        for d in subset:
            if d == target: continue
            if d in labels:
                H_parts.append(feats[d])
                y_parts.append(labels[d].astype(np.int64) + offset)
                offset += len(np.unique(labels[d]))
        H_train = np.concatenate(H_parts)
        y_train = np.concatenate(y_parts)
        # Train
        sc = StandardScaler()
        Htr = sc.fit_transform(H_train)
        Hte = sc.transform(target_feats_test)
        clf = LogisticRegression(max_iter=500, C=1.0, n_jobs=-1)
        clf.fit(Htr, y_train)
        preds = clf.predict(Hte)
        # Only count accuracy on target's own class range (0 to nc_t-1)
        acc = (preds == target_labels_test).mean()
        return float(acc)
    except Exception as e:
        return None

def run_compression_experiment(name, probe_id, feats, labels, feats_test, labels_test,
                                K_VALUES, N_SEEDS=10):
    """Run A/B compression experiment and return rows."""
    avail = sorted(feats.keys())
    M = len(avail)
    if M < 3:
        print(f"  {probe_id}: too few datasets ({M})")
        return []
    
    rc = {d: reff(feats[d]) for d in avail}
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"\n  {probe_id}: M={M}, r_eff(full)={reff_full:.1f}")
    for d in avail: print(f"    {d}: r_eff={rc[d]:.1f}")
    
    targets_lp = [d for d in feats_test if d in feats]  # has test set
    print(f"    Targets for LP: {targets_lp}")
    
    rows = []
    for k in K_VALUES:
        if k >= M: continue
        ratio = M / k
        print(f"\n    ═══ k={k} (compression {ratio:.1f}x) ═══")
        
        strategies = [
            ("Collapse", lambda: collapse_greedy(feats, rc, avail, k)),
            ("r_sum",    lambda: rsum_greedy(feats, rc, avail, k)),
        ]
        # Random with 10 seeds
        for seed in range(1, N_SEEDS+1):
            strategies.append((f"Random_s{seed}", lambda s=seed: random_select(avail, k, s)))
        
        for strat_name, strat_fn in strategies:
            t0 = time.time()
            selected = strat_fn()
            select_t = time.time() - t0
            H_sel = np.concatenate([feats[d] for d in selected])
            r_sel = reff(H_sel)
            retention = r_sel / reff_full
            
            # LP on each available target (use fixed logic)
            lp_scores = []
            for t in targets_lp:
                lp = lp_target_aug(feats, labels, t, feats_test[t], labels_test[t], selected)
                if lp is not None: lp_scores.append(lp)
            mean_lp = np.mean(lp_scores) if lp_scores else np.nan
            
            row = {"experiment": name, "probe": probe_id, "k": k, "M": M,
                   "compression_ratio": ratio,
                   "strategy": strat_name, "selected": str(selected),
                   "reff_sel": r_sel, "reff_full": reff_full,
                   "retention": retention,
                   "lp_mean": mean_lp, "n_lp_targets": len(lp_scores),
                   "select_time": select_t}
            rows.append(row)
        
        # Summary for this k
        collapse_row = [r for r in rows if r["k"]==k and r["strategy"]=="Collapse"][0]
        rsum_row = [r for r in rows if r["k"]==k and r["strategy"]=="r_sum"][0]
        rand_rows = [r for r in rows if r["k"]==k and r["strategy"].startswith("Random")]
        rand_ret = [r["retention"] for r in rand_rows]
        rand_lp = [r["lp_mean"] for r in rand_rows if not np.isnan(r["lp_mean"])]
        print(f"      Collapse : retention={collapse_row['retention']*100:.1f}% LP={collapse_row['lp_mean']:.4f}")
        print(f"      r_sum    : retention={rsum_row['retention']*100:.1f}% LP={rsum_row['lp_mean']:.4f}")
        print(f"      Random   : retention={np.mean(rand_ret)*100:.1f}% (std {np.std(rand_ret)*100:.1f}%)  LP={np.mean(rand_lp):.4f}" if rand_lp else f"      Random   : retention={np.mean(rand_ret)*100:.1f}% (std {np.std(rand_ret)*100:.1f}%)  LP=N/A")
    
    return rows

if __name__ == "__main__":
    print("=" * 72)
    print("  E6 v2: Extended Compression (A: ViT-B/16, B: Text)")
    print("=" * 72)
    
    all_rows = []
    
    # ════════════════════════════════════════
    # Experiment A: ViT-B/16 on 11 vision datasets
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("  Experiment A: ViT-B/16 on vision")
    print("="*60)
    
    VIT_DATASETS = ["cifar10","cifar100","stl10","svhn","mnist","fashion_mnist",
                    "imagenet_animal","imagenet_artifact","imagenet_food",
                    "imagenet_scene","imagenet_vehicle"]
    
    feats_vit, labels_vit = {}, {}
    feats_vit_test, labels_vit_test = {}, {}
    for ds in VIT_DATASETS:
        H, y = load("vit_b16", ds, "train")
        if H is not None:
            feats_vit[ds] = H
            if y is not None: labels_vit[ds] = y
            Ht, yt = load("vit_b16", ds, "test")
            if Ht is not None and yt is not None:
                feats_vit_test[ds] = Ht; labels_vit_test[ds] = yt
            print(f"    {ds}: train={H.shape}", "test=yes" if ds in feats_vit_test else "")
    
    rows_a = run_compression_experiment(
        "A_ViT_B16_Vision", "vit_b16",
        feats_vit, labels_vit, feats_vit_test, labels_vit_test,
        K_VALUES=[3,5,7,10], N_SEEDS=10
    )
    all_rows.extend(rows_a)
    
    # Also add existing ResNet-50 results for comparison
    print("\n" + "="*60)
    print("  Experiment A-ref: ResNet-50 (re-run with 10 seeds + fixed LP)")
    print("="*60)
    R50_DATASETS = ["cifar10","cifar100","stl10","svhn","mnist","fashion_mnist",
                    "bloodmnist","dermamnist","pathmnist",
                    "imagenet_animal","imagenet_artifact","imagenet_food",
                    "imagenet_scene","imagenet_vehicle"]
    feats_r50, labels_r50 = {}, {}
    feats_r50_test, labels_r50_test = {}, {}
    for ds in R50_DATASETS:
        H, y = load("resnet50", ds, "train")
        if H is not None:
            feats_r50[ds] = H
            if y is not None: labels_r50[ds] = y
            Ht, yt = load("resnet50", ds, "test")
            if Ht is not None and yt is not None:
                feats_r50_test[ds] = Ht; labels_r50_test[ds] = yt
    
    rows_a_ref = run_compression_experiment(
        "A_ResNet50_Vision", "resnet50",
        feats_r50, labels_r50, feats_r50_test, labels_r50_test,
        K_VALUES=[3,5,7,10], N_SEEDS=10
    )
    all_rows.extend(rows_a_ref)
    
    # ════════════════════════════════════════
    # Experiment B: Text (BERT-base L6 and RoBERTa L6)
    # ════════════════════════════════════════
    print("\n" + "="*60)
    print("  Experiment B: Text Encoders on 7 text datasets")
    print("="*60)
    
    TEXT_DATASETS = ["sst2","ag_news","dbpedia","20news","mnli","qnli","rte"]
    TEXT_PROBES = ["bert_L6", "bert_L12", "roberta_L6", "gpt2_L6"]
    
    for text_probe in TEXT_PROBES:
        print(f"\n  Probe: {text_probe}")
        feats_t = {}
        for ds in TEXT_DATASETS:
            path = os.path.join(TEXT_DIR, f"{text_probe}_{ds}.npz")
            if os.path.exists(path):
                d = np.load(path)
                H = d["H"].astype(np.float64)
                norms = np.linalg.norm(H, axis=1, keepdims=True)
                H = H / np.maximum(norms, 1e-10)
                feats_t[ds] = H
                print(f"    {ds}: {H.shape}")
        
        if len(feats_t) < 3: continue
        
        # Text: no labels/test sets cached, so retention only
        rows_b = run_compression_experiment(
            f"B_{text_probe}_Text", text_probe,
            feats_t, {}, {}, {},  # no labels/test
            K_VALUES=[3,5], N_SEEDS=10
        )
        all_rows.extend(rows_b)
    
    # Save
    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS, "e6_v2_compression.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved {len(df)} rows to {out}")
    
    # Grand summary
    print(f"\n{'='*72}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*72}")
    for exp in sorted(df["experiment"].unique()):
        sub = df[df["experiment"]==exp]
        print(f"\n  {exp}:")
        for k in sorted(sub["k"].unique()):
            sk = sub[sub["k"]==k]
            cr = sk[sk["strategy"]=="Collapse"].iloc[0]
            rs = sk[sk["strategy"]=="r_sum"].iloc[0]
            rand = sk[sk["strategy"].str.startswith("Random")]
            rand_ret_mean = rand["retention"].mean()
            rand_ret_std = rand["retention"].std()
            rand_lp_mean = rand["lp_mean"].mean() if not np.all(np.isnan(rand["lp_mean"])) else np.nan
            delta = (cr["retention"] - rand_ret_mean) * 100
            print(f"    k={int(k):>2d} ({cr['compression_ratio']:.1f}x): "
                  f"Collapse={cr['retention']*100:.1f}% (LP={cr['lp_mean']:.4f})  "
                  f"r_sum={rs['retention']*100:.1f}% (LP={rs['lp_mean']:.4f})  "
                  f"Random={rand_ret_mean*100:.1f}±{rand_ret_std*100:.1f}% (LP={rand_lp_mean:.4f})  "
                  f"Δ={delta:+.1f}pp")
    
    print("\n  Done.")
