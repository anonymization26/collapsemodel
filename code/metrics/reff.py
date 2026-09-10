"""Effective-rank measurements and the Collapse-4S compatibility API.

Collapse-4S is a theory-motivated summary predictor, not a universal theorem.
Its numerical formula and rank convention live in ``collapse_core`` so every
experiment uses the same implementation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .collapse_core import (
    collapse_4s,
    collapse_4s_predict,
    effective_rank,
    effective_rank_from_singular_values,
    energy_dominant_ranks,
    is_superadditive,
    split_numerical_singular_values,
    stable_singular_values_and_vh,
    superadditivity_delta,
)


def normalized_reff(matrix: np.ndarray) -> float:
    """Return effective rank divided by the maximum algebraic rank."""

    return effective_rank(matrix) / float(min(matrix.shape))


def predict_reff_collapse(
    h_a: np.ndarray,
    h_b: np.ndarray,
    k: int = 20,
) -> Dict[str, float | str | bool]:
    """Measure a matrix pair and assemble canonical Collapse-4S diagnostics."""

    singular_a_raw, vh_a = stable_singular_values_and_vh(h_a)
    singular_b_raw, vh_b = stable_singular_values_and_vh(h_b)
    singular_a, _ = split_numerical_singular_values(singular_a_raw, h_a.shape)
    singular_b, _ = split_numerical_singular_values(singular_b_raw, h_b.shape)
    r_a = effective_rank_from_singular_values(singular_a, h_a.shape)
    r_b = effective_rank_from_singular_values(singular_b, h_b.shape)
    merged = np.concatenate([h_a, h_b], axis=0)
    merged_singular, _ = stable_singular_values_and_vh(merged)
    merged_rank = effective_rank_from_singular_values(merged_singular, merged.shape)
    ideal_additive_rank = r_a + r_b
    collapse_ratio = (
        max(0.0, (ideal_additive_rank - merged_rank) / ideal_additive_rank)
        if ideal_additive_rank > 0.0
        else 0.0
    )

    nuclear_a = float(singular_a.sum())
    nuclear_b = float(singular_b.sum())
    if nuclear_a <= 0.0 or nuclear_b <= 0.0:
        raise ValueError("Collapse-4S requires two matrices with positive nuclear mass")
    k_used = min(max(int(k), 1), len(singular_a), len(singular_b))
    cosines = np.linalg.svd(
        vh_a[:k_used] @ vh_b[:k_used].T,
        compute_uv=False,
    )
    alpha = float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))
    gamma = nuclear_b / nuclear_a
    prediction = collapse_4s(r_a, r_b, gamma, alpha)
    delta_r = superadditivity_delta(merged_rank, r_a, r_b, gamma)
    superadditive = is_superadditive(merged_rank, r_a, r_b, gamma)

    direction_dominant = alpha > 0.5
    energy_dominant = gamma > 3.0 or gamma < 0.33
    if direction_dominant and energy_dominant:
        cause = "both"
    elif direction_dominant:
        cause = "direction"
    elif energy_dominant:
        cause = "energy"
    elif superadditive:
        cause = "superadditive"
    else:
        cause = "none"

    return {
        "r_eff_A": r_a,
        "r_eff_B": r_b,
        "r_eff_dominant": prediction.dominant_rank,
        "r_eff_subordinate": prediction.subordinate_rank,
        "r_eff_merged": merged_rank,
        "r_eff_ideal": ideal_additive_rank,
        "collapse_ratio": collapse_ratio,
        "delta_r": delta_r,
        "is_superadditive": superadditive,
        "sa_k": alpha,
        "nuclear_A": nuclear_a,
        "nuclear_B": nuclear_b,
        "energy_ratio": gamma,
        "q": prediction.q,
        "r_star": prediction.q,
        "r_pred_theory": prediction.prediction,
        "rho_c": prediction.rho_c,
        "dominant_cause": cause,
    }


def reff_bounds(r_a: float, r_b: float, gamma: float = 1.0) -> Dict[str, float]:
    """Return diagnostic reference levels, not universal hard bounds."""

    dominant, subordinate = energy_dominant_ranks(r_a, r_b, gamma)
    return {
        "dominant": dominant,
        "subordinate": subordinate,
        "ideal_additive": float(r_a + r_b),
    }


def reff_theoretical_prediction(
    r_a: float,
    r_b: float,
    alpha: float,
    gamma: float,
) -> float:
    """Compatibility wrapper for the former ``(alpha, gamma)`` argument order."""

    return collapse_4s_predict(r_a, r_b, gamma, alpha)


def make_controlled_pair(
    d: int = 512,
    k: int = 50,
    n: int = 1000,
    target_alpha: float = 0.5,
    gamma: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an exact proportional-spectrum paired-direction E1a sample."""

    if not 1 <= k <= min(n, d // 2):
        raise ValueError("k must satisfy 1 <= k <= min(n, d // 2)")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    rng = np.random.default_rng(seed)
    u_a = np.linalg.qr(rng.standard_normal((n, k)))[0]
    v_a = np.linalg.qr(rng.standard_normal((d, k)))[0]
    singular_a = np.exp(-np.arange(k) / (k / 3.0))
    singular_a = singular_a / singular_a.sum() * np.sqrt(n)

    v_perp = rng.standard_normal((d, k))
    v_perp -= v_a @ (v_a.T @ v_perp)
    v_perp = np.linalg.qr(v_perp)[0][:, :k]
    cosine = float(np.sqrt(np.clip(target_alpha, 0.0, 1.0)))
    sine = float(np.sqrt(1.0 - cosine * cosine))
    v_b = cosine * v_a + sine * v_perp

    u_b = np.linalg.qr(rng.standard_normal((n, k)))[0]
    h_a = (u_a * singular_a[None, :]) @ v_a.T
    h_b = (u_b * (gamma * singular_a)[None, :]) @ v_b.T
    return h_a, h_b


if __name__ == "__main__":
    for alpha in (0.0, 0.5, 1.0):
        h_a, h_b = make_controlled_pair(
            d=128,
            k=20,
            n=300,
            target_alpha=alpha,
            gamma=1.0,
        )
        result = predict_reff_collapse(h_a, h_b, k=20)
        print(
            f"alpha={alpha:.1f} measured={result['r_eff_merged']:.8f} "
            f"predicted={result['r_pred_theory']:.8f} delta={result['delta_r']:+.8f}"
        )
