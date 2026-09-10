"""Canonical numerical and Collapse-4S primitives.

This module is intentionally NumPy-only so the paper experiments and the
two-stage screening scripts use exactly the same definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


NUMERICAL_RANK_EPS_MULTIPLIER = 8.0
SUPERADDITIVITY_RELATIVE_TOLERANCE = 1e-10


def numerical_singular_tolerance(
    maximum_singular_value: float,
    matrix_shape: tuple[int, int],
) -> float:
    """Return the shared Gram-stable singular-value cutoff.

    The squared cutoff equals the usual backward-error cutoff for a symmetric
    eigensolve of ``H.T @ H``. Applying this singular-value cutoff to both SVD
    and Gram paths gives them one declared numerical-rank convention.
    """

    if len(matrix_shape) != 2 or min(matrix_shape) < 0:
        raise ValueError(f"invalid matrix shape: {matrix_shape}")
    scale = max(float(maximum_singular_value), 0.0)
    relative = math.sqrt(
        np.finfo(np.float64).eps
        * max(max(matrix_shape), 1)
        * NUMERICAL_RANK_EPS_MULTIPLIER
    )
    return max(scale * relative, np.finfo(np.float64).tiny)


def split_numerical_singular_values(
    singular: np.ndarray,
    matrix_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Split retained modes from modes below the shared numerical cutoff."""

    values = np.maximum(np.asarray(singular, dtype=np.float64), 0.0)
    if values.size == 0:
        return values, values
    tolerance = numerical_singular_tolerance(float(values.max()), matrix_shape)
    retained = values > tolerance
    return values[retained], values[~retained]


def effective_rank_from_singular_values(
    singular: np.ndarray,
    matrix_shape: tuple[int, int] | None = None,
) -> float:
    """Compute entropy effective rank under the shared numerical cutoff."""

    values = np.maximum(np.asarray(singular, dtype=np.float64), 0.0)
    if matrix_shape is None:
        matrix_shape = (values.size, values.size)
    values, _ = split_numerical_singular_values(values, matrix_shape)
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def stable_singular_values_and_vh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a thin float64 SVD, with a deterministic eigen fallback."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("SVD input contains non-finite values")
    try:
        _, singular, vh = np.linalg.svd(values, full_matrices=False)
        return singular, vh
    except np.linalg.LinAlgError:
        rows, columns = values.shape
        size = min(rows, columns)
        if rows <= columns:
            gram = values @ values.T
            eigenvalues, left = np.linalg.eigh((gram + gram.T) / 2.0)
            order = np.argsort(eigenvalues)[::-1]
            singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
            left = left[:, order]
            vh = np.zeros((size, columns), dtype=np.float64)
            tolerance = numerical_singular_tolerance(
                float(singular[0]) if singular.size else 0.0,
                values.shape,
            )
            retained = singular > tolerance
            vh[retained] = (left[:, retained].T @ values) / singular[retained, None]
            return singular, vh

        gram = values.T @ values
        eigenvalues, right = np.linalg.eigh((gram + gram.T) / 2.0)
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        return singular, right[:, order].T


def effective_rank(matrix: np.ndarray) -> float:
    """Compute effective rank directly from singular values."""

    values = np.asarray(matrix)
    singular, _ = stable_singular_values_and_vh(values)
    return effective_rank_from_singular_values(singular, values.shape)


def nuclear_mass(matrix: np.ndarray) -> float:
    """Return nuclear mass after applying the shared numerical cutoff."""

    values = np.asarray(matrix)
    singular, _ = stable_singular_values_and_vh(values)
    singular, _ = split_numerical_singular_values(singular, values.shape)
    return float(singular.sum())


def effective_rank_from_scatter(
    scatter: np.ndarray,
    source_shape: tuple[int, int] | None = None,
) -> float:
    """Compute effective rank from a PSD scatter using the same cutoff as SVD."""

    matrix = np.asarray(scatter, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square scatter matrix, got {matrix.shape}")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))
    shape = source_shape if source_shape is not None else matrix.shape
    return effective_rank_from_singular_values(singular, shape)


def energy_dominant_ranks(
    r_a: float,
    r_b: float,
    gamma: float,
) -> tuple[float, float]:
    """Return ranks of the higher- and lower-nuclear-mass sources."""

    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and positive, got {gamma}")
    if min(r_a, r_b) <= 0.0 or not all(map(math.isfinite, (r_a, r_b))):
        raise ValueError(f"effective ranks must be finite and positive, got {r_a}, {r_b}")
    gamma_value = 1.0 if math.isclose(gamma, 1.0, rel_tol=1e-12) else gamma
    return (
        (float(r_a), float(r_b))
        if gamma_value <= 1.0
        else (float(r_b), float(r_a))
    )


@dataclass(frozen=True)
class Collapse4SResult:
    prediction: float
    dominant_rank: float
    subordinate_rank: float
    q: float
    binary_entropy: float
    rho_c: float


def collapse_4s(
    r_a: float,
    r_b: float,
    gamma: float,
    alpha: float,
) -> Collapse4SResult:
    """Evaluate the paper's unique Collapse-4S summary predictor."""

    dominant, subordinate = energy_dominant_ranks(r_a, r_b, gamma)
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    alignment = min(max(float(alpha), 0.0), 1.0)
    if alignment <= 1e-15:
        alignment = 0.0
    elif alignment >= 1.0 - 1e-15:
        alignment = 1.0
    if alignment == 1.0:
        return Collapse4SResult(
            prediction=dominant,
            dominant_rank=dominant,
            subordinate_rank=subordinate,
            q=1.0,
            binary_entropy=0.0,
            rho_c=math.inf,
        )
    total = 1.0 + gamma * gamma
    discriminant = math.sqrt(
        max((1.0 - gamma * gamma) ** 2 + 4.0 * gamma * gamma * alignment, 0.0)
    )
    plus = math.sqrt(max((total + discriminant) / 2.0, 0.0))
    minus = math.sqrt(max((total - discriminant) / 2.0, 0.0))
    q = plus / (plus + minus)
    if q <= 0.0 or q >= 1.0:
        binary_entropy = 0.0
    else:
        binary_entropy = -q * math.log(q) - (1.0 - q) * math.log1p(-q)
    prediction = (
        dominant
        if q >= 1.0
        else math.exp(
            binary_entropy
            + q * math.log(dominant)
            + (1.0 - q) * math.log(subordinate)
        )
    )
    rho_c = math.inf if q >= 1.0 else math.exp(binary_entropy / (1.0 - q))
    return Collapse4SResult(
        prediction=prediction,
        dominant_rank=dominant,
        subordinate_rank=subordinate,
        q=q,
        binary_entropy=binary_entropy,
        rho_c=rho_c,
    )


def collapse_4s_predict(r_a: float, r_b: float, gamma: float, alpha: float) -> float:
    """Return only the Collapse-4S predicted effective rank."""

    return collapse_4s(r_a, r_b, gamma, alpha).prediction


def superadditivity_delta(
    merged_rank: float,
    r_a: float,
    r_b: float,
    gamma: float,
) -> float:
    """Measure gain over the higher-nuclear-mass source, as defined in the paper."""

    dominant, _ = energy_dominant_ranks(r_a, r_b, gamma)
    return float(merged_rank) - dominant


def is_superadditive(
    merged_rank: float,
    r_a: float,
    r_b: float,
    gamma: float,
) -> bool:
    """Classify gain over ``r_dom`` while suppressing roundoff-only gains."""

    dominant, _ = energy_dominant_ranks(r_a, r_b, gamma)
    tolerance = SUPERADDITIVITY_RELATIVE_TOLERANCE * max(
        abs(float(merged_rank)),
        abs(dominant),
        1.0,
    )
    return float(merged_rank) - dominant > tolerance
