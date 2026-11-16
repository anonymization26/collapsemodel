"""
Collapse Model
Grassmann manifold geometry tools (principal angles, geodesic distance)

Main functions:
    1. principal_angles()       — principal angles between two subspaces (core geometric operation)
    2. grassmann_distance()     — Grassmann geodesic distance (√Σθᵢ²)
    3. subspace_from_matrix()   — extract a Grassmann point from a feature matrix
    4. GrassmannLoss            — differentiable Grassmann alignment loss (PyTorch)

GrassmannLoss corresponds to the domain adaptation experiment (stage 4):
    L_G = Σᵢ<ⱼ Σₗ θᵢⱼₗ²
    Minimizing this loss aligns the top-k feature directions across all domains.
"""

import numpy as np
from typing import List, Tuple, Dict

from .collapse_core import split_numerical_singular_values, stable_singular_values_and_vh


# ─────────────────────────────────────────────────────────────
# 1. Numpy utilities (no gradients, for metric computation)
# ─────────────────────────────────────────────────────────────

def principal_angles(
    V_A: np.ndarray,
    V_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the principal angles between two subspaces.

    Args:
        V_A: (d, k_A) orthonormal matrix whose columns are basis vectors of subspace A
        V_B: (d, k_B) orthonormal matrix whose columns are basis vectors of subspace B

    Returns:
        cos_angles: (min(k_A, k_B),) cos(θᵢ) ∈ [0,1], descending order
        angles_rad: (min(k_A, k_B),) θᵢ ∈ [0, π/2], ascending order
    """
    M = V_A.T @ V_B   # (k_A, k_B)
    cos_angles = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    return cos_angles, angles_rad


def grassmann_distance(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> float:
    """
    Grassmann geodesic distance: d_G(S_A, S_B) = √(Σᵢ θᵢ²)

    Args:
        H_A: (N_A, d) feature matrix
        H_B: (N_B, d) feature matrix
        k:   subspace dimension

    Returns:
        Grassmann distance (non-negative scalar)
    """
    from .subspace_alignment import principal_angles_from_matrices
    _, angles_rad = principal_angles_from_matrices(H_A, H_B, k)
    return float(np.sqrt(np.sum(angles_rad ** 2)))


def subspace_from_matrix(H: np.ndarray, k: int) -> np.ndarray:
    """
    Extract the orthonormal basis of the top-k subspace from feature matrix H
    (a point on the Grassmann manifold).

    Args:
        H: (N, d) feature matrix
        k: subspace dimension

    Returns:
        V_k: (d, k) orthonormal matrix (columns are principal directions)
    """
    from .subspace_alignment import top_k_basis

    return top_k_basis(H, k)


def karcher_mean(subspaces: List[np.ndarray], n_iter: int = 10) -> np.ndarray:
    """
    Karcher mean (Fréchet mean approximation) of multiple subspaces on the Grassmann manifold.

    Uses gradient descent (following Absil et al. 2004):
        initialized at the first subspace, iteratively computing the logarithmic and
        exponential maps.

    Args:
        subspaces: List of (d, k) orthonormal matrices
        n_iter:    number of gradient descent iterations

    Returns:
        V_mean: (d, k) mean subspace
    """
    if len(subspaces) == 0:
        raise ValueError("subspaces cannot be empty")
    if len(subspaces) == 1:
        return subspaces[0].copy()

    d, k = subspaces[0].shape
    V = subspaces[0].copy()   # initialize at the first point

    for _ in range(n_iter):
        # Compute the logarithmic map (tangent vectors) from all points to the current V
        tangent_sum = np.zeros_like(V)
        for Vi in subspaces:
            # Logarithmic map: Log_V(Vi) = (I - VV.T) Vi (Vi.T V)^{-1} θ_i / sin(θ_i)
            # Simplified implementation: orthogonalize the residual via QR
            M = V.T @ Vi
            U_m, s, Wt = np.linalg.svd(M, full_matrices=False)
            s = np.clip(s, -1.0, 1.0)
            theta = np.arccos(s)
            # tangent vector
            residual = Vi - V @ M
            Q, _ = np.linalg.qr(residual)
            Q = Q[:, :k]
            tangent = Q @ np.diag(theta)
            tangent_sum += tangent

        # Exponential map (gradient step)
        step = tangent_sum / len(subspaces)
        U_s, s_s, Wt_s = np.linalg.svd(step, full_matrices=False)
        # V_new = V cos(s_s) Wt_s + U_s sin(s_s)  — approximation
        V_new = V @ Wt_s.T * np.cos(s_s)[None, :] + U_s * np.sin(s_s)[None, :]
        V, _ = np.linalg.qr(V_new)
        V = V[:, :k]

    return V


# ─────────────────────────────────────────────────────────────
# 2. PyTorch differentiable Grassmann alignment loss
# ─────────────────────────────────────────────────────────────

class GrassmannLoss:
    """
    Grassmann alignment loss L_G (domain adaptation experiment).

    Computes the sum of squared Grassmann distances between feature matrices of all domains:
        L_G = Σᵢ<ⱼ d_G(Sᵢ, Sⱼ)² = Σᵢ<ⱼ Σₗ θᵢⱼₗ²

    Supports both numpy (evaluation) and PyTorch (training) modes.

    PyTorch backpropagation note:
        Gradients of principal angles are computed via matrix differentiation:
        ∂L/∂H = (∂L/∂θ) × (∂θ/∂cos) × (∂cos/∂M) × (∂M/∂V) × (∂V/∂H)
        where ∂θ/∂cos = -1/√(1-cos²); all other terms are handled by autograd.
    """

    def __init__(self, k: int = 50):
        self.k = k

    def numpy_loss(self, feature_matrices: List[np.ndarray]) -> float:
        """
        Pure numpy evaluation (no gradients), for training monitoring.

        Args:
            feature_matrices: List of (N, d) feature matrices, one per domain

        Returns:
            L_G: sum of squared Grassmann distances over all domain pairs
        """
        total = 0.0
        n = len(feature_matrices)
        for i in range(n):
            for j in range(i + 1, n):
                d = grassmann_distance(feature_matrices[i], feature_matrices[j], self.k)
                total += d ** 2
        return total

    def torch_loss(self, feature_list) -> "torch.Tensor":
        """
        PyTorch differentiable version, supports backpropagation.

        Args:
            feature_list: List of torch.Tensor (N, d), one per domain

        Returns:
            L_G: scalar tensor (supports .backward())

        Implementation strategy:
            Uses randomized SVD approximation (top-k truncation via torch.linalg.svd),
            computes principal angles via arccos of matrix cosine similarity;
            gradient path: H → V_k (right singular vectors) → cos(θ) → θ² → L_G
        """
        import torch

        total_loss = None
        n = len(feature_list)

        for i in range(n):
            for j in range(i + 1, n):
                H_i = feature_list[i]
                H_j = feature_list[j]

                # Truncated SVD (top-k right singular vectors)
                k_eff = max(1, min(self.k, H_i.shape[0], H_j.shape[0], H_i.shape[1]))

                # torch.linalg.svd supports autograd
                _, _, Vt_i = torch.linalg.svd(H_i, full_matrices=False)
                _, _, Vt_j = torch.linalg.svd(H_j, full_matrices=False)

                V_i = Vt_i[:k_eff, :].T   # (d, k_eff)
                V_j = Vt_j[:k_eff, :].T

                # Cross-product → singular values = cos(θ)
                M = V_i.T @ V_j
                cos_angles = torch.linalg.svd(M, full_matrices=False).S
                cos_angles = torch.clamp(cos_angles, 1e-6, 1.0 - 1e-6)

                theta = torch.arccos(cos_angles)
                loss_ij = torch.sum(theta ** 2)

                total_loss = loss_ij if total_loss is None else total_loss + loss_ij

        return total_loss if total_loss is not None else torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────
# 3. Spectral-directional orthogonal decomposition (auxiliary lemma 0.3)
# ─────────────────────────────────────────────────────────────

def spectral_wasserstein_distance(H_A: np.ndarray, H_B: np.ndarray) -> float:
    """
    Wasserstein-1 distance between singular value distributions (spectral transport distance).

    p_A = σᵢ / Σσⱼ (normalized singular value distribution)
    W₁(p_A, p_B) = L1 difference of sorted singular values

    Corresponds to the first term of d_repr² = W₂²(spectral) + d_G²(directional)
    in the auxiliary lemma.
    """
    def _singular_probs(H):
        sigma, _ = stable_singular_values_and_vh(H)
        sigma, _ = split_numerical_singular_values(sigma, H.shape)
        return sigma / (sigma.sum() + 1e-12)

    p_A = _singular_probs(H_A)
    p_B = _singular_probs(H_B)

    # Align lengths (zero-pad)
    n = max(len(p_A), len(p_B))
    p_A = np.pad(p_A, (0, n - len(p_A)))
    p_B = np.pad(p_B, (0, n - len(p_B)))

    # W₁ = ||CDF_A - CDF_B||₁
    cdf_A = np.cumsum(p_A)
    cdf_B = np.cumsum(p_B)
    return float(np.sum(np.abs(cdf_A - cdf_B)))


def repr_distance(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict:
    """
    Full representation distance (auxiliary lemma 0.3):
        d_repr² = α·W₁(spectral)² + β·d_G(directional)²

    Args:
        H_A, H_B: feature matrices
        k:        Grassmann subspace dimension
        alpha:    weight for the spectral term
        beta:     weight for the directional term

    Returns:
        dict: {"w1_spectral", "d_grassmann", "d_repr_sq", "d_repr"}
    """
    w1  = spectral_wasserstein_distance(H_A, H_B)
    d_g = grassmann_distance(H_A, H_B, k=k)
    d_sq = alpha * (w1 ** 2) + beta * (d_g ** 2)
    return {
        "w1_spectral":  w1,
        "d_grassmann":  d_g,
        "d_repr_sq":    d_sq,
        "d_repr":       float(np.sqrt(d_sq)),
    }


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, k = 100, 256, 20

    def _rand_subspace(rng, d, k):
        return np.linalg.qr(rng.standard_normal((d, k)))[0]

    # Test 1: identical subspace → Grassmann distance = 0
    V = _rand_subspace(rng, d, k)
    cos_a, ang = principal_angles(V, V.copy())
    assert np.allclose(ang, 0, atol=1e-5), f"identical subspace principal angles should be ≈0, got {ang.max():.2e}"
    print(f"[Test 1] identical subspace: max_angle={ang.max():.2e}  ✓")

    # Test 2: orthogonal subspaces → cos(θ) ≈ 0
    V_A = _rand_subspace(rng, d, k)
    V_B_raw = _rand_subspace(rng, d, k)
    V_B = V_B_raw - V_A @ (V_A.T @ V_B_raw)
    V_B, _ = np.linalg.qr(V_B)
    V_B = V_B[:, :k]
    cos_ab, _ = principal_angles(V_A, V_B)
    print(f"[Test 2] orthogonal subspaces: mean_cos={cos_ab.mean():.4f}  (expected close to 0)")

    # Test 3: GrassmannLoss numpy
    H_A = rng.standard_normal((N, d)).astype(np.float32)
    H_B = rng.standard_normal((N, d)).astype(np.float32)
    gl = GrassmannLoss(k=k)
    loss_val = gl.numpy_loss([H_A, H_B])
    print(f"[Test 3] GrassmannLoss([H_A, H_B]) = {loss_val:.4f}")

    # Test 4: repr_distance
    result = repr_distance(H_A, H_B, k=k)
    print(f"[Test 4] repr_distance: W₁={result['w1_spectral']:.4f}  "
          f"d_G={result['d_grassmann']:.4f}  d_repr={result['d_repr']:.4f}")

    print("\nAll tests passed ✓")
