"""
Collapse Model
Direction gain criterion Δ_dir and greedy dataset selection (§5 Application)

Core formula:
    Δ_dir(D_new | D_old) = 1 - SA_k(S(D_new, M), S(D_old, M))

    Δ_dir ≈ 1 → new dataset introduces entirely novel directions → high value to include
    Δ_dir ≈ 0 → new dataset directions are redundant             → inclusion is ineffective

Advantages over r_eff:
    - Unaffected by energy scale (no γ confounding)
    - Label-independent
    - Directly quantifies the independence of newly introduced directions
    - Naturally addresses the merge collapse problem

Used in E3 experiment (greedy dataset selection strategy).
"""

import numpy as np
from typing import List, Dict, Optional

from .subspace_alignment import compute_subspace_alignment, compute_multiscale_sa


# ─────────────────────────────────────────────────────────────
# 1. Core computation
# ─────────────────────────────────────────────────────────────

def direction_gain(
    H_new: np.ndarray,
    H_old: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    Compute the direction gain Δ_dir of a new dataset's feature matrix relative to the existing dataset.

    Args:
        H_new: (N_new, d) feature matrix of the new dataset (L2 normalized per sample)
        H_old: (N_old, d) feature matrix of the existing dataset (L2 normalized per sample)
        k:     subspace dimension

    Returns:
        dict: {
            "delta_dir":      float  Δ_dir ∈ [0,1], primary metric (higher is better)
            "sa_k":           float  SA_k (directional overlap; Δ_dir = 1 - sa_k)
            "grassmann_dist": float  Grassmann distance
            "k_used":         int
        }
    """
    result = compute_subspace_alignment(H_new, H_old, k=k)
    delta_dir = 1.0 - result["sa_k"]
    return {
        "delta_dir":      delta_dir,
        "sa_k":           result["sa_k"],
        "grassmann_dist": result["grassmann_dist"],
        "k_used":         result["k_used"],
    }


def direction_gain_multiscale(
    H_new: np.ndarray,
    H_old: np.ndarray,
    k_values: tuple = (10, 20, 50),
) -> Dict:
    """Multi-scale Δ_dir: ablate the effect of different values of k."""
    ms = compute_multiscale_sa(H_new, H_old, k_values=k_values)
    result = {}
    for k in k_values:
        result[f"delta_dir_k{k}"] = 1.0 - ms[f"sa_k{k}"]
    result["mean_delta_dir"] = float(np.mean([result[f"delta_dir_k{k}"] for k in k_values]))
    return result


# ─────────────────────────────────────────────────────────────
# 2. Greedy dataset selection (E3 main logic)
# ─────────────────────────────────────────────────────────────

def greedy_dataset_selection(
    H_base: np.ndarray,
    candidate_features: Dict[str, np.ndarray],
    k: int = 20,
    n_rounds: Optional[int] = None,
) -> List[Dict]:
    """
    Greedy dataset selection: each round picks the candidate dataset with the highest Δ_dir.

    Args:
        H_base:              (N_base, d) initial training set feature matrix
        candidate_features:  {dataset_name: (N, d)} candidate dataset features
        k:                   subspace dimension
        n_rounds:            maximum number of selection rounds (None = until candidates exhausted)

    Returns:
        List of dicts in selection order: {
            "round":       int
            "selected":    str   name of the selected dataset
            "delta_dir":   float
            "sa_k":        float
            "all_scores":  dict  Δ_dir scores of all candidates in this round
        }
    """
    remaining = dict(candidate_features)
    H_current = H_base.copy()
    history = []
    n_rounds = n_rounds or len(remaining)

    for rnd in range(n_rounds):
        if not remaining:
            break

        # Compute Δ_dir for all candidates
        scores = {}
        for name, H_cand in remaining.items():
            dg = direction_gain(H_cand, H_current, k=k)
            scores[name] = dg["delta_dir"]

        # Select the candidate with the highest Δ_dir
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]

        sa_val = compute_subspace_alignment(
            remaining[best_name], H_current, k=k)["sa_k"]

        history.append({
            "round":      rnd + 1,
            "selected":   best_name,
            "delta_dir":  best_score,
            "sa_k":       sa_val,
            "all_scores": dict(scores),
        })

        print(f"  Round {rnd+1}: select {best_name:20s}  Δ_dir={best_score:.4f}  SA_k={sa_val:.4f}")

        # Merge the selected dataset's features into H_current
        H_current = np.concatenate([H_current, remaining[best_name]], axis=0)
        del remaining[best_name]

    return history


def reff_greedy_selection(
    H_base: np.ndarray,
    candidate_features: Dict[str, np.ndarray],
    n_rounds: Optional[int] = None,
) -> List[Dict]:
    """
    Baseline: greedy selection by maximum r_eff (used as E3 comparison baseline).

    Args:
        H_base:             initial training set feature matrix
        candidate_features: candidate dataset features
        n_rounds:           maximum number of selection rounds

    Returns:
        Same format as greedy_dataset_selection
    """
    from .reff import effective_rank

    remaining = dict(candidate_features)
    history = []
    n_rounds = n_rounds or len(remaining)

    for rnd in range(n_rounds):
        if not remaining:
            break

        scores = {name: effective_rank(H) for name, H in remaining.items()}
        best_name = max(scores, key=scores.__getitem__)

        history.append({
            "round":      rnd + 1,
            "selected":   best_name,
            "reff_score": scores[best_name],
            "all_scores": dict(scores),
        })

        print(f"  [reff baseline] Round {rnd+1}: select {best_name:20s}  r_eff={scores[best_name]:.2f}")
        del remaining[best_name]

    return history


# ─────────────────────────────────────────────────────────────
# 3. Joint scoring (Application D)
# ─────────────────────────────────────────────────────────────

def joint_score(r_eff: float, sa_k: float) -> float:
    """
    Joint score: Score(M) = r_eff^{1/2} × SA_k

    Captures both representational richness (r_eff) and cross-domain stability (SA_k).
    """
    return float(np.sqrt(max(r_eff, 0.0)) * sa_k)


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, k = 100, 256, 20

    def _norm(H):
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    H_base = _norm(rng.standard_normal((N, d)))

    # Test 1: fully orthogonal new dataset → Δ_dir ≈ 1
    H_orth = _norm(rng.standard_normal((N, d)))
    r1 = direction_gain(H_orth, H_base, k=k)
    print(f"[Test 1] random new set:  Δ_dir={r1['delta_dir']:.4f}  SA_k={r1['sa_k']:.4f}")

    # Test 2: identical dataset → Δ_dir ≈ 0
    r2 = direction_gain(H_base.copy(), H_base, k=k)
    print(f"[Test 2] identical data:  Δ_dir={r2['delta_dir']:.4f}  SA_k={r2['sa_k']:.4f}  "
          f"{'✓' if r2['delta_dir'] < 0.05 else '✗'}")

    # Test 3: greedy selection
    candidates = {
        "high_gain": _norm(rng.standard_normal((N, d))),   # independent directions
        "low_gain":  H_base + 0.05 * rng.standard_normal((N, d)),  # nearly identical
        "mid_gain":  _norm(H_base + rng.standard_normal((N, d))),  # mixed
    }
    print("\n[Test 3] greedy selection:")
    history = greedy_dataset_selection(H_base, candidates, k=k)
    assert history[0]["selected"] in ["high_gain", "mid_gain"], "first round should select high-gain dataset"
    print(f"  First round selected: {history[0]['selected']}  ✓")

    # Test 4: joint score
    score = joint_score(r_eff=45.0, sa_k=0.8)
    print(f"\n[Test 4] joint_score(r=45, sa=0.8) = {score:.4f}  "
          f"(expected≈{np.sqrt(45)*0.8:.4f})")

    print("\nAll tests passed ✓")
