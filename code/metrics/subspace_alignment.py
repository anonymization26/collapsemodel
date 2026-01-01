"""
Collapse Model
Subspace alignment SA_k (V channel alpha: directional alignment)

Core formula:
    SA_k(S_A, S_B) = (1/k) Σᵢ cos²(θᵢ)

    θ₁,...,θₖ are the principal angles between S_A and S_B
    SA_k = 1 → fully aligned (directional redundancy)
    SA_k = 0 → fully orthogonal (ideal complementarity)

This module is a self-contained pure-numpy computation kernel with no DataLoader
dependency, suitable for callers that pass feature matrices directly.
The SAMetric class (with feature extraction) is available in sa_metric_full.py.
"""

import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────

def principal_angles_from_matrices(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal angles between the top-k subspaces of two feature matrices.

    Steps:
        1. SVD(H_A) → V_A[:k];  SVD(H_B) → V_B[:k]
        2. Cross-product M = V_A.T @ V_B
        3. Singular values of SVD(M) = cos(θᵢ)

    Args:
        H_A: (N_A, d) feature matrix
        H_B: (N_B, d) feature matrix
        k:   subspace dimension

    Returns:
        cos_angles: (k_eff,) cos(θᵢ) ∈ [0,1], descending order
        angles_rad: (k_eff,) θᵢ ∈ [0, π/2], ascending order (corresponding to descending cos)
    """
    N_A, d_A = H_A.shape
    N_B, d_B = H_B.shape
    if d_A != d_B:
        raise ValueError(f"Feature dimension mismatch: {d_A} vs {d_B}")

    k_eff = max(1, min(k, N_A, N_B, d_A))

    _, _, Vt_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B, full_matrices=False)

    V_A = Vt_A[:k_eff, :].T   # (d, k_eff)
    V_B = Vt_B[:k_eff, :].T   # (d, k_eff)

    M_cross = V_A.T @ V_B     # (k_eff, k_eff)
    cos_angles = np.linalg.svd(M_cross, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)

    angles_rad = np.arccos(cos_angles)

    return cos_angles, angles_rad


def compute_subspace_alignment(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    Compute SA_k: subspace alignment of the top-k subspaces of two feature matrices.

    Args:
        H_A: (N_A, d) feature matrix (per-sample L2 normalization recommended)
        H_B: (N_B, d) feature matrix
        k:   principal subspace dimension

    Returns:
        dict: {
            "sa_k":           float  ∈ [0,1], primary metric
            "grassmann_dist": float  Grassmann geodesic distance √(Σθᵢ²)
            "cos_angles":     list   k values of cos(θᵢ) (descending order)
            "k_used":         int    effective k actually used
        }
    """
    cos_angles, angles_rad = principal_angles_from_matrices(H_A, H_B, k)

    sa_k = float(np.mean(cos_angles ** 2))
    grassmann_dist = float(np.sqrt(np.sum(angles_rad ** 2)))
    k_eff = len(cos_angles)

    return {
        "sa_k":           sa_k,
        "grassmann_dist": grassmann_dist,
        "cos_angles":     cos_angles.tolist(),
        "k_used":         k_eff,
    }


def compute_multiscale_sa(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k_values: Tuple[int, ...] = (10, 20, 50),
) -> Dict:
    """
    Multi-scale SA_k: compute and aggregate results for multiple values of k.

    Returns:
        dict: {
            "sa_k10": float, "sa_k20": float, "sa_k50": float,
            "mean_sa_multiscale": float
        }
    """
    results = {}
    sa_list = []
    for k in k_values:
        r = compute_subspace_alignment(H_A, H_B, k=k)
        results[f"sa_k{k}"] = r["sa_k"]
        sa_list.append(r["sa_k"])
    results["mean_sa_multiscale"] = float(np.mean(sa_list))
    return results


def top_k_basis(H: np.ndarray, k: int) -> np.ndarray:
    """
    Extract the top-k right singular vectors of a feature matrix (a point on the Grassmann manifold).

    Returns:
        V_k: (d, k) orthonormal matrix whose columns are the top-k principal directions
    """
    k_eff = max(1, min(k, min(H.shape)))
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k_eff, :].T   # (d, k_eff)


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d = 200, 512

    def _norm(H):
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    # Test 1: identical matrix → SA_k ≈ 1.0
    H = _norm(rng.standard_normal((N, d)))
    r = compute_subspace_alignment(H, H.copy(), k=20)
    assert r["sa_k"] > 0.99, f"identical matrix SA_k={r['sa_k']:.4f} should be ≈1"
    print(f"[Test 1] identical matrix  SA_20={r['sa_k']:.4f}  ✓")

    # Test 2: independent random matrices → SA_k ≈ k/d (close to 0)
    H2 = _norm(rng.standard_normal((N, d)))
    r2 = compute_subspace_alignment(H, H2, k=20)
    print(f"[Test 2] random matrix SA_20={r2['sa_k']:.4f}  (expected close to 0)")

    # Test 3: mild perturbation → SA_k between the two extremes
    H3 = _norm(H + 0.1 * rng.standard_normal((N, d)))
    r3 = compute_subspace_alignment(H, H3, k=20)
    assert r3["sa_k"] > r2["sa_k"], "perturbed SA_k should be higher than random"
    print(f"[Test 3] perturbed matrix SA_20={r3['sa_k']:.4f}  ✓ > random")

    # Test 4: multi-scale
    ms = compute_multiscale_sa(H, H3)
    print(f"[Test 4] multi-scale: " + "  ".join(f"k{k}={ms[f'sa_k{k}']:.4f}" for k in (10, 20, 50)))

    print("\nAll tests passed ✓")
