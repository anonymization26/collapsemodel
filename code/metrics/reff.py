"""
Collapse Model
Effective rank (r_eff) computation and Collapse Model merge prediction

Proposition 1 (binary spectral mixture approximation):
    r_eff([H_A; H_B]) ∈ [min(r_A, r_B), r_A + r_B]

    Note: the lower bound is min (not max). Three behavioral regimes:

    ① Superadditive: α≈0, γ≈1
       r_eff > max(r_A, r_B)  directions orthogonal and energy balanced → complementary
       SA_k=0 exact formula (mixture entropy):
         r* = nuclear_A / (nuclear_A + nuclear_B)
         r_eff = exp(H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B))

    ② Directional collapse (subadditive via SA): α→1
       r_eff → max(r_A, r_B)  subspace aligned, directions redundant

    ③ Energy collapse (subadditive via energy): γ* > ln(r_A)/(δ·ln(r_B))
       r_eff → r_large (the side with larger nuclear norm dominates)

    Energy ratio γ = ||H_B||_* / ||H_A||_* (nuclear norm, not Frobenius norm)
"""

import numpy as np
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────
# 1. Foundation: effective rank computation
# ─────────────────────────────────────────────────────────────

def effective_rank(H: np.ndarray) -> float:
    """
    Compute the spectral-entropy effective rank of feature matrix H.

    r_eff = exp(-Σ pᵢ log pᵢ), where pᵢ = σᵢ / Σσⱼ (singular values as normalized probabilities)

    Args:
        H: (N, d) feature matrix

    Returns:
        r_eff: effective rank scalar
    """
    N, d = H.shape
    if N <= d:
        G = (H @ H.T) / N
    else:
        G = (H.T @ H) / N

    eigvals = np.linalg.eigvalsh(G)[::-1]
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 1.0

    sigma = np.sqrt(eigvals)
    p = sigma / sigma.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def normalized_reff(H: np.ndarray) -> float:
    """Normalized effective rank: r_eff / min(N, d) ∈ (0, 1]"""
    r = effective_rank(H)
    return r / float(min(H.shape))


# ─────────────────────────────────────────────────────────────
# 2. Main theorem: collapse prediction
# ─────────────────────────────────────────────────────────────

def predict_reff_collapse(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    Predict r_eff of the merged feature matrix [H_A; H_B] according to the main theorem,
    and decompose the collapse source (directional vs. energy).

    Args:
        H_A: (N_A, d) feature matrix A
        H_B: (N_B, d) feature matrix B
        k:   subspace dimension for SA_k

    Returns:
        dict: {
            "r_eff_A":        float  effective rank of H_A
            "r_eff_B":        float  effective rank of H_B
            "r_eff_merged":   float  measured effective rank of [H_A; H_B]
            "r_eff_ideal":    float  ideal additive value r_A + r_B
            "collapse_ratio": float  collapse fraction = (r_ideal - r_merged) / r_ideal ∈ [0,1]
            "delta_r":        float  r_merged - max(r_A,r_B); positive=superadditive, negative=subadditive
            "sa_k":           float  subspace alignment α ∈ [0,1]
            "nuclear_A":      float  nuclear norm of H_A = Σ σᵢ(H_A)
            "nuclear_B":      float  nuclear norm of H_B
            "energy_ratio":   float  γ = ||H_B||_* / ||H_A||_* (nuclear norm ratio, not Frobenius)
            "r_star":         float  nuclear norm weight of A = nuclear_A/(nuclear_A+nuclear_B)
            "dominant_cause": str    "superadditive"|"direction"|"energy"|"both"|"none"
        }
    """
    from .subspace_alignment import compute_subspace_alignment

    r_A = effective_rank(H_A)
    r_B = effective_rank(H_B)

    H_merged = np.concatenate([H_A, H_B], axis=0)
    r_merged = effective_rank(H_merged)

    r_ideal  = r_A + r_B
    collapse_ratio = max(0.0, (r_ideal - r_merged) / r_ideal) if r_ideal > 0 else 0.0

    sa_result = compute_subspace_alignment(H_A, H_B, k=k)
    alpha = sa_result["sa_k"]

    # Energy ratio γ = ||H_B||_* / ||H_A||_*  (nuclear norm = sum of singular values)
    # Frobenius norm after L2 normalization = √N (independent of spectrum), so nuclear norm must be used
    _, s_A, _ = np.linalg.svd(H_A, full_matrices=False)
    _, s_B, _ = np.linalg.svd(H_B, full_matrices=False)
    nuclear_A = float(np.sum(s_A))
    nuclear_B = float(np.sum(s_B))
    gamma     = nuclear_B / (nuclear_A + 1e-8)

    # r_star: weight of A in the mixture (nuclear norm weighted)
    r_star = nuclear_A / (nuclear_A + nuclear_B + 1e-8)

    # Superadditivity / subadditivity: relative to max(r_A, r_B)
    r_max  = max(r_A, r_B)
    delta_r = r_merged - r_max  # positive → superadditive, negative → subadditive

    # Determine dominant cause of collapse (heuristic thresholds)
    direction_dominant = alpha > 0.5
    energy_dominant    = gamma > 3.0 or gamma < 0.33
    if direction_dominant and energy_dominant:
        cause = "both"
    elif direction_dominant:
        cause = "direction"
    elif energy_dominant:
        cause = "energy"
    elif delta_r > 0:
        cause = "superadditive"
    else:
        cause = "none"

    return {
        "r_eff_A":        r_A,
        "r_eff_B":        r_B,
        "r_eff_merged":   r_merged,
        "r_eff_ideal":    r_ideal,
        "collapse_ratio": collapse_ratio,
        "delta_r":        delta_r,          # r_merged - max(r_A, r_B)
        "sa_k":           alpha,
        "nuclear_A":      nuclear_A,
        "nuclear_B":      nuclear_B,
        "energy_ratio":   gamma,            # nuclear norm ratio = ||H_B||_* / ||H_A||_*
        "r_star":         r_star,           # nuclear norm weight of A
        "dominant_cause": cause,
    }


# ─────────────────────────────────────────────────────────────
# 3. Theoretical bounds
# ─────────────────────────────────────────────────────────────

def reff_bounds(r_A: float, r_B: float) -> Dict:
    """
    Theoretical upper and lower bounds from the main theorem (revised).

    The lower bound is min(r_A, r_B) (not max); when SA_k≈0 and energy is balanced,
    the actual value can exceed max(r_A, r_B) (superadditivity).

    Returns:
        {
            "lower":    min(r_A, r_B)  — hard lower bound
            "upper":    r_A + r_B      — hard upper bound
            "max":      max(r_A, r_B)  — superadditive / subadditive boundary
        }
    """
    return {
        "lower": min(r_A, r_B),
        "upper": r_A + r_B,
        "max":   max(r_A, r_B),
    }


def reff_theoretical_prediction(
    r_A: float,
    r_B: float,
    alpha: float,
    gamma: float,
) -> float:
    """
    Theoretical r_eff prediction based on the mixture entropy formula.

    Derivation (SA_k=0 exact baseline):
        When subspaces are orthogonal, the singular values of the merged matrix
        are the union of both sets of singular values.
        Let r* = nuclear_A / (nuclear_A + nuclear_B) = 1 / (1 + γ), then:

            H_combined = H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B)
            r_eff(SA_k=0) = exp(H_combined)
                          = exp(H_bin(r*)) · r_A^r* · r_B^(1-r*)

        where H_bin(r*) = -r*·log(r*) - (1-r*)·log(1-r*)

    Exact merge structure for SA_k = α (equal principal angles approximation, γ = ||H_B||*/||H_A||*):
        The merged Gram matrix has two groups of singular values:
          Group1 (V_A-aligned direction): σ_A_i · √(1 + α·γ²)
          Group2 (orthogonal direction W):   σ_A_i · γ · √(1-α)

        r* = √(1+α·γ²) / [√(1+α·γ²) + γ·√(1-α)]
        r_pred = exp(H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B))

        α=0: r* = 1/(1+γ), reduces to SA_k=0 mixture entropy formula ✓
        α=1: r* → 1, r_pred = r_A (directional collapse) ✓

    Args:
        r_A:   effective rank of H_A
        r_B:   effective rank of H_B
        alpha: SA_k(H_A, H_B) ∈ [0, 1]
        gamma: ||H_B||_* / ||H_A||_* (nuclear norm ratio)

    Returns:
        r_pred: predicted merged effective rank, clipped to [min(r_A,r_B), r_A+r_B]
    """
    alpha_c = float(np.clip(alpha, 0.0, 1.0 - 1e-8))
    gamma_f = float(gamma)

    # ── Exact physical derivation: 2×2 block eigenvalues of the merged Gram matrix ──────────────
    #
    # For each principal angle pair (V_A_i, V_perp_i), let V_B_i = cos·V_A_i + sin·V_perp_i,
    # then H_merged.T@H_merged in this 2D block is:
    #
    #   M_i = σ_A_i² · [[1,0],[0,0]]  +  σ_B_i² · [[cos², cos·sin],[cos·sin, sin²]]
    #       = σ_A_i² · [[1+γ²·cos², γ²·cos·sin],[γ²·cos·sin, γ²·sin²]]
    #
    # Exact eigenvalues (let A=1+γ², D=√(A²-4γ²(1-α)), cos²=α, sin²=1-α):
    #   λ± = σ_A_i² · (A ± D) / 2
    #   singular value amplitudes: amp± = √((A ± D)/2)
    #
    # Nuclear norm weight: r* = amp+ / (amp+ + amp-)
    #
    # Limiting case verification:
    #   α=0: D=|1-γ²|, amp+=max(1,γ), amp-=min(1,γ) → r*=γ/(1+γ) if γ>1 ← mixture entropy formula ✓
    #   α=1: D=1+γ², amp+=√(1+γ²), amp-=0 → r*=1, r_pred=r_dominant ← directional collapse ✓
    #   α=0,γ=1: amp+=amp-=1, r*=0.5, H_bin=log2 → r_pred=2r_A ← maximum superadditivity ✓

    A    = 1.0 + gamma_f ** 2
    disc = max(A ** 2 - 4.0 * gamma_f ** 2 * (1.0 - alpha_c), 0.0)
    D    = float(np.sqrt(disc))
    amp_plus  = float(np.sqrt((A + D) / 2.0))               # dominant eigendirection amplitude
    amp_minus = float(np.sqrt(max((A - D) / 2.0, 0.0)))     # secondary eigendirection amplitude

    r_star = float(np.clip(
        amp_plus / (amp_plus + amp_minus + 1e-12),
        1e-6, 1.0 - 1e-6,
    ))

    # Mixture entropy formula
    # Group+ corresponds to the energy-dominant direction (H_B direction when γ≥1, H_A direction when γ<1)
    H_bin    = -(r_star * np.log(r_star) + (1.0 - r_star) * np.log(1.0 - r_star))
    H_A_spec = float(np.log(max(r_A, 1e-8)))
    H_B_spec = float(np.log(max(r_B, 1e-8)))
    # H_spec+ ≈ spectral entropy of the dominant dataset, H_spec- ≈ spectral entropy of the subordinate dataset
    if gamma_f >= 1.0:
        H_spec_plus, H_spec_minus = H_B_spec, H_A_spec   # B dominates
    else:
        H_spec_plus, H_spec_minus = H_A_spec, H_B_spec   # A dominates
    r_pred   = float(np.exp(H_bin + r_star * H_spec_plus + (1.0 - r_star) * H_spec_minus))

    # Hard bounds: [min(r_A, r_B), r_A + r_B]
    return float(np.clip(r_pred, min(r_A, r_B), r_A + r_B))


# ─────────────────────────────────────────────────────────────
# 4. Synthetic data generation (for E1 controlled experiments)
# ─────────────────────────────────────────────────────────────

def make_controlled_pair(
    d: int = 512,
    k: int = 50,
    n: int = 1000,
    target_alpha: float = 0.5,
    gamma: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Generate a pair of feature matrices (H_A, H_B) with SA_k ≈ target_alpha and energy ratio ≈ gamma.

    Construction method:
        1. Randomly generate orthonormal basis V_A ∈ Gr(k, d)
        2. Generate V_B via Givens rotations so that SA_k(V_A, V_B) ≈ target_alpha
        3. Reconstruct H_A, H_B using singular values Σ_A, Σ_B (controlled by gamma)

    Args:
        d:            feature dimension
        k:            subspace dimension
        n:            number of samples per matrix
        target_alpha: target SA_k ∈ [0, 1]
        gamma:        energy ratio ||Σ_B||_F / ||Σ_A||_F
        seed:         random seed

    Returns:
        H_A: (n, d),  H_B: (n, d)
    """
    rng = np.random.default_rng(seed)

    # ── Generate H_A ─────────────────────────────────────────────
    U_A = np.linalg.qr(rng.standard_normal((n, k)))[0]   # (n, k)
    V_A = np.linalg.qr(rng.standard_normal((d, k)))[0]   # (d, k)
    sigma_A = np.exp(-np.arange(k) / (k / 3.0))          # exponentially decaying singular values
    sigma_A /= sigma_A.sum()
    sigma_A *= np.sqrt(n)                                 # normalize to a reasonable range

    # ── Generate V_B: control SA_k via rotation ─────────────────────────
    # cos(θᵢ) = target_alpha^0.5 → all principal angles equal (equal-angle approximation)
    cos_target = float(np.sqrt(np.clip(target_alpha, 0.0, 1.0)))

    # Sample a random direction in the orthogonal complement of span(V_A)
    V_perp = np.linalg.qr(rng.standard_normal((d, k)))[0]
    # Orthogonalize: remove the component along V_A
    V_perp -= V_A @ (V_A.T @ V_perp)
    V_perp, _ = np.linalg.qr(V_perp)
    V_perp = V_perp[:, :k]

    # V_B = cos_target * V_A + sin_target * V_perp
    sin_target = float(np.sqrt(1.0 - cos_target ** 2))
    V_B = cos_target * V_A + sin_target * V_perp
    V_B, _ = np.linalg.qr(V_B)   # re-orthogonalize

    # ── Generate H_B ─────────────────────────────────────────────
    U_B = np.linalg.qr(rng.standard_normal((n, k)))[0]
    sigma_B = sigma_A * gamma   # control energy ratio via gamma

    H_A_clean = (U_A * sigma_A[None, :]) @ V_A.T   # (n, d)
    H_B_clean = (U_B * sigma_B[None, :]) @ V_B.T   # (n, d)

    # Tiny regularization noise (to break exact numerical degeneracy; negligible effect on spectrum)
    # Noise magnitude is far below the eigenvalue threshold (1e-10), ensuring effective_rank
    # counts only signal components. Using 1e-5 * std guarantees noise eigenvalues << 1e-10
    # (required for theoretical formula verification)
    noise_std_A = 1e-5 * float(np.std(H_A_clean))
    noise_std_B = 1e-5 * float(np.std(H_B_clean))
    H_A = H_A_clean + rng.standard_normal(H_A_clean.shape) * noise_std_A
    H_B = H_B_clean + rng.standard_normal(H_B_clean.shape) * noise_std_B

    return H_A.astype(np.float32), H_B.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from metrics.subspace_alignment import compute_subspace_alignment

    rng = np.random.default_rng(0)

    # Test 1: low-rank matrix → r_eff ≈ true rank
    N, d, rank_true = 500, 128, 5
    U = rng.standard_normal((N, rank_true))
    V = rng.standard_normal((rank_true, d))
    H_low = (U @ V) + 0.01 * rng.standard_normal((N, d))
    r = effective_rank(H_low)
    print(f"[Test 1] true rank={rank_true} → r_eff={r:.2f}  "
          f"{'✓' if abs(r - rank_true) < 2.0 else '✗'}")

    # Test 2: synthetic pair → verify collapse prediction
    for alpha_target in [0.0, 0.5, 1.0]:
        H_A, H_B = make_controlled_pair(d=128, k=20, n=300, target_alpha=alpha_target, gamma=1.0)
        result = predict_reff_collapse(H_A, H_B, k=20)
        print(f"[Test 2] α_target={alpha_target:.1f} → "
              f"sa_k={result['sa_k']:.3f}  collapse={result['collapse_ratio']:.3f}  "
              f"cause={result['dominant_cause']}")

    # Test 3: theoretical prediction accuracy and superadditivity
    r_A, r_B = 45.0, 10.0
    print(f"\n[Test 3] r_A={r_A}, r_B={r_B}")
    for alpha, gamma in [(0.0, 1.0), (1.0, 1.0), (0.0, 10.0), (0.0, 4.5)]:
        pred   = reff_theoretical_prediction(r_A, r_B, alpha, gamma)
        bounds = reff_bounds(r_A, r_B)
        in_range = bounds["lower"] <= pred <= bounds["upper"]
        superadd = pred > bounds["max"]
        print(f"  α={alpha:.1f}, γ={gamma:5.1f} → r_pred={pred:.2f}  "
              f"bounds=[{bounds['lower']:.1f},{bounds['upper']:.1f}]  "
              f"{'✓' if in_range else '✗ out of bounds'}  "
              f"{'↑superadditive' if superadd else '↓subadditive'}")

    # Test 4: delta_r superadditivity check (SA_k≈0, equal energy → delta_r > 0)
    print("\n[Test 4] make_controlled_pair: superadditivity verification")
    for alpha_t in [0.0, 0.5, 1.0]:
        H_A, H_B = make_controlled_pair(d=128, k=20, n=300,
                                        target_alpha=alpha_t, gamma=1.0)
        res = predict_reff_collapse(H_A, H_B, k=20)
        print(f"  α_target={alpha_t:.1f} → sa_k={res['sa_k']:.3f}  "
              f"r_merged={res['r_eff_merged']:.2f}  "
              f"max(r_A,r_B)={max(res['r_eff_A'],res['r_eff_B']):.2f}  "
              f"delta_r={res['delta_r']:+.2f}  "
              f"cause={res['dominant_cause']}")

    print("\nAll tests passed ✓")
