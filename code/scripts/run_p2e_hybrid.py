"""
P2-E: Hybrid Collapse + Centroid-FF strategy.

Two-signal hybrid: pick ceil(k/2) via Collapse Greedy (retention signal),
then fill remaining floor(k/2) via Centroid-FF on the unselected pool
(centroid-coverage signal). Tests whether the two signals identified by
P1-A are complementary.

We also include a "Centroid-then-Collapse" variant for symmetry, plus
a Vendi-then-Collapse variant.

Run:
    cd collapse-model-repro
    nohup python3 -u scripts/run_p2e_hybrid.py > p2e.log 2>&1 &
"""
import os, sys, math, time, warnings, glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
# Reuse all infrastructure from P1-A runner
from run_p1a_modern_baselines import (
    discover_datasets, load_from, reff, vendi, sa_k, cpred,
    collapse_greedy, vendi_greedy, centroid_ff_greedy,
    domain_diverse_greedy, maxreff_greedy, rsum_greedy, random_select,
    fair_lp_subspace, DATASET_DOMAIN, RESULTS, K_SA, N_SAMPLES, SEED,
)

# ─── hybrid strategies ──────────────────────────────────────────────────
def hybrid_collapse_centroid(feats, rc, all_ds, k):
    """First half: Collapse Greedy. Second half: Centroid-FF on remaining."""
    if k >= len(all_ds): return list(all_ds)
    half = (k + 1) // 2  # ceil(k/2)
    selected = collapse_greedy(feats, rc, all_ds, half)
    remaining = [d for d in all_ds if d not in selected]
    # centroid-FF on remaining, anchored to already-selected
    cents = {}
    for d in all_ds:
        c = feats[d].mean(0)
        n = np.linalg.norm(c)
        cents[d] = c / max(n, 1e-10)
    while len(selected) < k:
        best_d, best_dist = None, -np.inf
        for d in remaining:
            mind = min(np.linalg.norm(cents[d] - cents[s]) for s in selected)
            if mind > best_dist:
                best_dist, best_d = mind, d
        if best_d is None: break
        selected.append(best_d); remaining.remove(best_d)
    return selected

def hybrid_centroid_collapse(feats, rc, all_ds, k):
    """Reverse: First half Centroid-FF, second half Collapse on remaining."""
    if k >= len(all_ds): return list(all_ds)
    half = (k + 1) // 2
    selected = centroid_ff_greedy(feats, all_ds, half)
    remaining_set = set(all_ds) - set(selected)
    H_cur = np.concatenate([feats[d] for d in selected])
    r_cur = reff(H_cur)
    while len(selected) < k:
        best_pred, best_d = -1, None
        nA = np.linalg.svd(H_cur, compute_uv=False).sum()
        for d in remaining_set:
            alpha = sa_k(H_cur, feats[d])
            nB = np.linalg.svd(feats[d], compute_uv=False).sum()
            pred = cpred(r_cur, rc[d], nB/(nA+1e-8), alpha)
            if pred > best_pred: best_pred, best_d = pred, d
        if best_d is None: break
        selected.append(best_d); remaining_set.discard(best_d)
        H_cur = np.concatenate([H_cur, feats[best_d]])
        r_cur = reff(H_cur)
    return selected

def hybrid_collapse_domain(feats, rc, all_ds, k):
    """First half: Collapse. Second half: DomainDiverse on remaining."""
    if k >= len(all_ds): return list(all_ds)
    half = (k + 1) // 2
    selected = collapse_greedy(feats, rc, all_ds, half)
    remaining = [d for d in all_ds if d not in selected]
    while len(selected) < k and remaining:
        dom_counts = {}
        for sd in selected:
            dom = DATASET_DOMAIN.get(sd, "unknown")
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
        def key(d):
            dom = DATASET_DOMAIN.get(d, "unknown")
            return (dom_counts.get(dom, 0), -rc[d])
        pick = min(remaining, key=key)
        selected.append(pick); remaining.remove(pick)
    return selected

# ─── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 76)
    print("  P2-E: Hybrid Collapse + Centroid-FF / DomainDiverse")
    print("=" * 76, flush=True)

    PROBE = "resnet50"
    K_VALUES = [5, 7, 10]   # k=3 too small to split; skip
    N_SEEDS = 5             # fewer seeds since this is supplementary

    train_paths, test_paths = discover_datasets(PROBE)
    feats, labels = {}, {}
    for name, path in train_paths.items():
        try:
            H, y = load_from(path); feats[name] = H
            if y is not None: labels[name] = y
        except Exception:
            pass
    feats_test, labels_test = {}, {}
    for name, path in test_paths.items():
        if name in feats:
            try:
                Ht, yt = load_from(path)
                if yt is not None:
                    feats_test[name] = Ht; labels_test[name] = yt
            except Exception:
                pass

    avail = sorted(feats.keys())
    M = len(avail)
    targets_lp = [d for d in avail if d in feats_test and d in labels_test]
    print(f"  M={M}; LP targets ({len(targets_lp)}): {targets_lp}", flush=True)

    rc = {d: reff(feats[d]) for d in avail}
    H_full = np.concatenate([feats[d] for d in avail])
    reff_full = reff(H_full)
    print(f"  r_eff(full M={M}) = {reff_full:.1f}", flush=True)

    all_rows = []
    for k in K_VALUES:
        if k >= M: continue
        ratio = M / k
        print(f"\n  ═══ k={k} ({ratio:.1f}x) ═══", flush=True)

        strategies = [
            ("Hybrid_Col_Cen", lambda: hybrid_collapse_centroid(feats, rc, avail, k)),
            ("Hybrid_Cen_Col", lambda: hybrid_centroid_collapse(feats, rc, avail, k)),
            ("Hybrid_Col_Dom", lambda: hybrid_collapse_domain(feats, rc, avail, k)),
        ]
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
                  f"vendi={v_sel:6.1f}  LP={mean_lp:.4f}  ({elapsed:.1f}s)",
                  flush=True)
            print(f"      selected: {selected}", flush=True)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = os.path.join(RESULTS, "p2e_hybrid_baselines.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved {len(df)} rows to {out}")

    # Summary
    summary = {}
    for k in K_VALUES:
        if k >= M: continue
        d = {}
        for sname in ["Hybrid_Col_Cen", "Hybrid_Cen_Col", "Hybrid_Col_Dom"]:
            r = next((x for x in all_rows if x["k"]==k and x["strategy"]==sname), None)
            if r is None: continue
            d[sname] = {
                "retention_pct": r["retention_pct"],
                "lp_mean": r["lp_mean"],
                "selected": r["selected"],
            }
        summary[f"k={k}"] = d
    with open(os.path.join(RESULTS, "p2e_hybrid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  FINAL SUMMARY (paper-ready):")
    for k in K_VALUES:
        if k >= M: continue
        print(f"\n    k={k}:")
        for sname in ["Hybrid_Col_Cen", "Hybrid_Cen_Col", "Hybrid_Col_Dom"]:
            r = next((x for x in all_rows if x["k"]==k and x["strategy"]==sname), None)
            if r is None: continue
            print(f"      {sname:18s}  ret={r['retention_pct']:6.1f}%  LP={r['lp_mean']:.4f}")
    print("\n  Done.")
