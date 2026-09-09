#!/usr/bin/env python3
"""Classical pool-selection baselines for the two-stage compression study.

The script intentionally depends only on NumPy. It evaluates selectors on the
same controlled-overlap collections as ``two_stage_large_scale.py`` while
recording method runtime and communication requirements separately from metric
evaluation time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


SOURCES = [
    "bloodmnist", "breastmnist", "cifar10", "cifar100", "cifar100_coarse",
    "dermamnist", "fashion_mnist", "mnist", "octmnist", "organamnist",
    "organcmnist", "organsmnist", "pathmnist", "pneumoniamnist",
    "rendered_sst2", "retinamnist", "semeion", "stl10", "svhn",
    "tiny_imagenet", "tissuemnist", "usps",
]
OVERLAPS = [0.25, 0.50, 0.75, 1.00]
CONSTRUCTION_SEEDS = [20260831, 20260832, 20260833]
BUDGETS = [3, 5]
REPRESENTATIONS = ("centroid", "subspace", "sample_nn")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_rows(h: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    return h / np.maximum(norms, 1e-8)


def load_feature(root: Path, name: str, encoder: str) -> np.ndarray:
    path = root / f"{name}_{encoder}_N1000.npz"
    with np.load(path) as archive:
        h = archive["H"].astype(np.float32)
    return normalize_rows(h)


def effective_rank_from_singular_values(singular: np.ndarray) -> float:
    """Return entropy effective rank after a scale-aware numerical-rank cutoff."""

    values = np.asarray(singular, dtype=np.float64)
    if values.size == 0:
        return 0.0
    values = np.maximum(values, 0.0)
    tolerance = max(
        float(values.max()) * np.finfo(np.float64).eps * max(values.size, 1) * 8.0,
        np.finfo(np.float64).tiny,
    )
    values = values[values > tolerance]
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def split_numerical_singular_values(
    singular: np.ndarray, matrix_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Split numerical signal from roundoff-scale singular values."""

    values = np.maximum(np.asarray(singular, dtype=np.float64), 0.0)
    if values.size == 0:
        return values, values
    tolerance = max(
        float(values.max()) * np.finfo(np.float64).eps * max(matrix_shape) * 8.0,
        np.finfo(np.float64).tiny,
    )
    positive = values > tolerance
    return values[positive], values[~positive]


def effective_rank_from_scatter(scatter: np.ndarray) -> float:
    """Compute effective rank of a PSD scatter matrix without counting roundoff modes."""

    matrix = np.asarray(scatter, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = max(float(eigenvalues[-1]), 0.0) if eigenvalues.size else 0.0
    tolerance = max(
        maximum * np.finfo(np.float64).eps * max(matrix.shape) * 8.0,
        np.finfo(np.float64).tiny,
    )
    singular = np.sqrt(np.maximum(eigenvalues[eigenvalues > tolerance], 0.0))
    return effective_rank_from_singular_values(singular)


def effective_rank(h: np.ndarray) -> float:
    matrix = np.asarray(h, dtype=np.float64)
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    return effective_rank_from_scatter(gram)


def stable_singular_values_and_vh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a thin SVD, with a deterministic symmetric-eigen fallback."""

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
            tolerance = (
                singular[0] * np.finfo(np.float64).eps * max(values.shape) * 8.0
                if singular.size else 0.0
            )
            positive = singular > tolerance
            vh[positive] = (
                left[:, positive].T @ values
            ) / singular[positive, None]
            return singular, vh

        gram = values.T @ values
        eigenvalues, right = np.linalg.eigh((gram + gram.T) / 2.0)
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        return singular, right[:, order].T


def feature_vendi_score(h: np.ndarray) -> float:
    """Vendi Score of samples under the linear kernel (squared singular values)."""
    matrix = np.asarray(h, dtype=np.float64)
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    values = np.linalg.eigvalsh((gram + gram.T) / 2.0)
    maximum = max(float(values[-1]), 0.0) if values.size else 0.0
    tolerance = max(
        maximum * np.finfo(np.float64).eps * max(gram.shape) * 8.0,
        np.finfo(np.float64).tiny,
    )
    values = values[values > tolerance]
    if values.size == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


@dataclass(frozen=True)
class GramSketch:
    """Rank-L PSD factor, tail bounds, and locally computed scalar summaries."""

    factor: np.ndarray
    tail_nuclear_bound: float
    tail_squared_bound: float
    tail_rank_bound: int
    full_rank_bound: int
    marginal_effective_rank: float
    marginal_nuclear_mass: float


def make_gram_sketch(features: np.ndarray, rank: int) -> GramSketch:
    """Build B with B.T @ B equal to the rank-L truncated feature Gram matrix."""

    if rank < 1:
        raise ValueError("Gram sketch rank must be positive")
    matrix = np.asarray(features, dtype=np.float64)
    singular, vh = stable_singular_values_and_vh(matrix)
    count = min(rank, singular.size)
    factor = singular[:count, None] * vh[:count]
    tail = singular[count:]
    marginal, _ = split_numerical_singular_values(singular, matrix.shape)
    return GramSketch(
        factor=factor,
        tail_nuclear_bound=float(tail.sum()),
        tail_squared_bound=float(tail @ tail),
        tail_rank_bound=int(tail.size),
        full_rank_bound=min(matrix.shape),
        marginal_effective_rank=effective_rank_from_singular_values(marginal),
        marginal_nuclear_mass=float(marginal.sum()),
    )


def pool_gram_sketches(
    features: dict[str, np.ndarray], rank: int,
) -> dict[str, GramSketch]:
    return {name: make_gram_sketch(matrix, rank) for name, matrix in features.items()}


def gram_sketch_statistics(sketches: Iterable[GramSketch]) -> dict[str, float | int]:
    """Score a composable sketch and evaluate Theorem 8's log-rank interval."""

    parts = list(sketches)
    if not parts:
        raise ValueError("at least one Gram sketch is required")
    factor = np.concatenate([part.factor for part in parts], axis=0)
    raw_singular, _ = stable_singular_values_and_vh(factor)
    singular, numerical_tail = split_numerical_singular_values(
        raw_singular, factor.shape,
    )
    if singular.size == 0:
        raise ValueError("aggregate Gram sketch is numerically zero")
    approximate_rank = effective_rank_from_singular_values(singular)
    approximate_nuclear = float(singular.sum())
    numerical_tail_nuclear = float(numerical_tail.sum())
    numerical_tail_squared = float(numerical_tail @ numerical_tail)
    tail_nuclear = float(
        sum(part.tail_nuclear_bound for part in parts) + numerical_tail_nuclear
    )
    tail_squared = float(
        sum(part.tail_squared_bound for part in parts) + numerical_tail_squared
    )
    tail_rank = min(
        factor.shape[1],
        sum(part.tail_rank_bound for part in parts) + numerical_tail.size,
    )
    rank_bound = min(factor.shape[1], sum(part.full_rank_bound for part in parts))
    frobenius_tail_bound = math.sqrt(max(tail_rank * tail_squared, 0.0))
    tail_bound = min(tail_nuclear, frobenius_tail_bound)
    denominator = approximate_nuclear + tail_bound
    epsilon = tail_bound / denominator if denominator > 0.0 else 0.0
    dimension_bound = max(rank_bound, 1)
    if dimension_bound == 1 or epsilon <= 0.0:
        log_error = 0.0
    elif epsilon <= 1.0 - 1.0 / dimension_bound:
        binary_entropy = (
            -epsilon * math.log(epsilon)
            -(1.0 - epsilon) * math.log1p(-epsilon)
        )
        log_error = binary_entropy + epsilon * math.log(dimension_bound - 1)
    else:
        log_error = math.log(dimension_bound)
    approximate_log_rank = math.log(max(approximate_rank, 1.0))
    return {
        "effective_rank": approximate_rank,
        "log_effective_rank": approximate_log_rank,
        "nuclear_mass": approximate_nuclear,
        "tail_nuclear_bound": tail_nuclear,
        "tail_squared_bound": tail_squared,
        "tail_rank_bound": tail_rank,
        "numerical_tail_nuclear_bound": numerical_tail_nuclear,
        "numerical_tail_squared_bound": numerical_tail_squared,
        "numerical_tail_rank_bound": int(numerical_tail.size),
        "tail_mass_bound": tail_bound,
        "epsilon": epsilon,
        "dimension_bound": dimension_bound,
        "log_error_bound": log_error,
        "log_rank_lower": approximate_log_rank - log_error,
        "log_rank_upper": approximate_log_rank + log_error,
    }


def rank_l_gram_greedy(
    sketches: dict[str, GramSketch],
    k: int,
    *,
    return_diagnostics: bool = False,
) -> list[str] | tuple[list[str], list[dict[str, float | int | str | bool]]]:
    """Greedily maximize the direct rank-L aggregate score using summaries only."""

    if k < 1 or k > len(sketches):
        raise ValueError(f"selection size must be in [1, {len(sketches)}]")
    selected: list[str] = []
    diagnostics: list[dict[str, float | int | str | bool]] = []
    while len(selected) < k:
        remaining = sorted(set(sketches) - set(selected))
        candidate_stats = {
            name: gram_sketch_statistics(sketches[item] for item in selected + [name])
            for name in remaining
        }
        choice = max(
            remaining,
            key=lambda name: float(candidate_stats[name]["log_effective_rank"]),
        )
        competing_upper = max(
            (
                float(candidate_stats[name]["log_rank_upper"])
                for name in remaining if name != choice
            ),
            default=-math.inf,
        )
        chosen = candidate_stats[choice]
        diagnostics.append({
            "step": len(selected) + 1,
            "choice": choice,
            "approximate_effective_rank": float(chosen["effective_rank"]),
            "log_error_bound": float(chosen["log_error_bound"]),
            "numerical_tail_nuclear_bound": float(
                chosen["numerical_tail_nuclear_bound"]
            ),
            "certified_exact_greedy_choice": (
                float(chosen["log_rank_lower"]) > competing_upper
            ),
        })
        selected.append(choice)
    if return_diagnostics:
        return selected, diagnostics
    return selected


def make_collection(
    source: dict[str, np.ndarray],
    overlap: float,
    pool_size: int,
    seed: int,
    duplicate_sources: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    pools: dict[str, np.ndarray] = {}
    family: dict[str, str] = {}
    base_indices: dict[str, np.ndarray] = {}
    complement_indices: dict[str, np.ndarray] = {}

    for offset, name in enumerate(SOURCES):
        h = source[name]
        if len(h) < 2 * pool_size:
            raise ValueError(f"{name} has {len(h)} rows; need at least {2 * pool_size}")
        permutation = np.random.default_rng(seed + offset).permutation(len(h))
        base_indices[name] = permutation[:pool_size]
        complement_indices[name] = permutation[pool_size:2 * pool_size]
        pools[name] = h[base_indices[name]]
        family[name] = name

    shared_count = int(round(pool_size * overlap))
    for offset, name in enumerate(duplicate_sources):
        rng = np.random.default_rng(seed + 10_000 + offset)
        shared = (
            rng.choice(base_indices[name], shared_count, replace=False)
            if shared_count else np.empty(0, dtype=np.int64)
        )
        fresh_count = pool_size - shared_count
        fresh = (
            rng.choice(complement_indices[name], fresh_count, replace=False)
            if fresh_count else np.empty(0, dtype=np.int64)
        )
        indices = np.concatenate([shared, fresh])
        rng.shuffle(indices)
        alias = f"{name}__overlap_{int(overlap * 100):03d}"
        pools[alias] = source[name][indices]
        family[alias] = name
    return pools, family


def pool_statistics(
    features: dict[str, np.ndarray], top_k: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ranks: dict[str, float] = {}
    nuclear: dict[str, float] = {}
    centroids: dict[str, np.ndarray] = {}
    subspaces: dict[str, np.ndarray] = {}
    for name, h in features.items():
        singular, vh = stable_singular_values_and_vh(h)
        positive, _ = split_numerical_singular_values(singular, h.shape)
        ranks[name] = effective_rank_from_singular_values(positive)
        nuclear[name] = float(positive.sum())
        centroid = h.mean(axis=0)
        centroids[name] = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        subspaces[name] = vh[: min(top_k, len(positive))].astype(np.float32)
    return ranks, nuclear, centroids, subspaces


def similarity_matrix(
    features: dict[str, np.ndarray],
    names: list[str],
    representation: str,
    centroids: dict[str, np.ndarray],
    subspaces: dict[str, np.ndarray],
) -> np.ndarray:
    count = len(names)
    similarity = np.eye(count, dtype=np.float64)
    if representation == "centroid":
        matrix = np.stack([centroids[name] for name in names]).astype(np.float64)
        similarity = (matrix @ matrix.T + 1.0) / 2.0
    elif representation == "subspace":
        for row, left in enumerate(names):
            for column in range(row + 1, count):
                right = names[column]
                rank = min(len(subspaces[left]), len(subspaces[right]))
                value = float(np.linalg.norm(subspaces[left] @ subspaces[right].T, ord="fro") ** 2)
                value /= max(rank, 1)
                similarity[row, column] = similarity[column, row] = value
    elif representation == "sample_nn":
        for row, left in enumerate(names):
            for column in range(row + 1, count):
                right = names[column]
                cross = features[left] @ features[right].T
                nearest = 0.5 * (float(cross.max(axis=1).mean()) + float(cross.max(axis=0).mean()))
                similarity[row, column] = similarity[column, row] = (nearest + 1.0) / 2.0
    else:
        raise ValueError(f"unknown representation: {representation}")
    similarity = np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def nearest_psd_correlation(kernel: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
    projected = (vectors * np.maximum(values, 1e-10)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), 1e-12))
    projected = projected / np.outer(scale, scale)
    projected = np.clip((projected + projected.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(projected, 1.0)
    return projected


def deterministic_argmax(candidates: Iterable[str], score: Callable[[str], float]) -> str:
    ordered = sorted(candidates)
    return max(ordered, key=score)


def rank_only(ranks: dict[str, float], k: int) -> list[str]:
    return sorted(ranks, key=lambda name: (-ranks[name], name))[:k]


def family_rank(ranks: dict[str, float], family: dict[str, str], k: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for name in sorted(ranks, key=lambda item: (-ranks[item], item)):
        if family[name] in seen:
            continue
        selected.append(name)
        seen.add(family[name])
        if len(selected) == k:
            break
    return selected


def collapse_predict(r_a: float, r_b: float, gamma: float, alpha: float) -> float:
    dominant, subordinate = (r_a, r_b) if gamma <= 1 else (r_b, r_a)
    total = 1 + gamma * gamma
    discriminant = math.sqrt(max((1 - gamma * gamma) ** 2 + 4 * gamma * gamma * alpha, 0.0))
    plus = math.sqrt((total + discriminant) / 2)
    minus = math.sqrt(max((total - discriminant) / 2, 1e-30))
    probability = min(max(plus / (plus + minus), 1e-12), 1 - 1e-12)
    entropy = -probability * math.log(probability) - (1 - probability) * math.log(1 - probability)
    return math.exp(
        entropy + probability * math.log(dominant) + (1 - probability) * math.log(subordinate)
    )


def collapse_greedy(features: dict[str, np.ndarray], k: int, top_k: int) -> list[str]:
    ranks, nuclear, _, subspaces = pool_statistics(features, top_k)
    selected = [deterministic_argmax(features, lambda name: ranks[name])]
    current = features[selected[0]]
    while len(selected) < k:
        current_stats = pool_statistics({"current": current}, top_k)
        current_rank = current_stats[0]["current"]
        current_nuclear = current_stats[1]["current"]
        current_subspace = current_stats[3]["current"]

        def score(name: str) -> float:
            cosines = np.linalg.svd(current_subspace @ subspaces[name].T, compute_uv=False)
            alpha = float(np.mean(np.clip(cosines, 0, 1) ** 2))
            gamma = nuclear[name] / max(current_nuclear, 1e-12)
            return collapse_predict(current_rank, ranks[name], gamma, alpha)

        remaining = [name for name in features if name not in selected]
        choice = deterministic_argmax(remaining, score)
        selected.append(choice)
        current = np.concatenate([current, features[choice]], axis=0)
    return selected


def full_merged_rank_greedy(features: dict[str, np.ndarray], k: int) -> list[str]:
    """Greedy exact-score baseline; this is not a combinatorial upper bound."""

    selected: list[str] = []
    current: np.ndarray | None = None
    while len(selected) < k:
        remaining = [name for name in features if name not in selected]

        def score(name: str) -> float:
            merged = features[name] if current is None else np.concatenate([current, features[name]], axis=0)
            return effective_rank(merged)

        choice = deterministic_argmax(remaining, score)
        selected.append(choice)
        current = features[choice] if current is None else np.concatenate([current, features[choice]], axis=0)
    return selected


exact_merged_rank_greedy = full_merged_rank_greedy


def exhaustive_merged_rank_oracle(
    features: dict[str, np.ndarray], k: int, max_combinations: int = 10_000,
) -> list[str]:
    """Return the global best size-k set when an explicitly bounded search is feasible."""

    names = sorted(features)
    combination_count = math.comb(len(names), k)
    if combination_count > max_combinations:
        raise ValueError(
            f"exhaustive search needs {combination_count} combinations, "
            f"above limit {max_combinations}"
        )
    best: tuple[str, ...] | None = None
    best_score = -math.inf
    for combination in itertools.combinations(names, k):
        score = effective_rank(np.concatenate([features[name] for name in combination]))
        if score > best_score:
            best = combination
            best_score = score
    if best is None:
        raise ValueError("empty exhaustive search")
    return list(best)


def facility_location(similarity: np.ndarray, names: list[str], k: int) -> list[str]:
    coverage = np.zeros(len(names), dtype=np.float64)
    selected: list[int] = []
    while len(selected) < k:
        remaining = [index for index in range(len(names)) if index not in selected]
        gains = {
            index: float(np.maximum(coverage, similarity[:, index]).sum() - coverage.sum())
            for index in remaining
        }
        choice = max(remaining, key=lambda index: (gains[index], -index))
        selected.append(choice)
        coverage = np.maximum(coverage, similarity[:, choice])
    return [names[index] for index in selected]


def k_center(similarity: np.ndarray, names: list[str], ranks: dict[str, float], k: int) -> list[str]:
    distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    selected = [names.index(deterministic_argmax(names, lambda name: ranks[name]))]
    while len(selected) < k:
        remaining = [index for index in range(len(names)) if index not in selected]
        choice = max(
            remaining,
            key=lambda index: (float(distance[index, selected].min()), ranks[names[index]], -index),
        )
        selected.append(choice)
    return [names[index] for index in selected]


def medoid_cost(distance: np.ndarray, medoids: list[int]) -> float:
    return float(distance[:, medoids].min(axis=1).sum())


def k_medoids_pam(similarity: np.ndarray, names: list[str], k: int) -> list[str]:
    distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    medoids: list[int] = []
    while len(medoids) < k:
        candidates = [index for index in range(len(names)) if index not in medoids]
        choice = min(candidates, key=lambda index: (medoid_cost(distance, medoids + [index]), index))
        medoids.append(choice)

    improved = True
    while improved:
        improved = False
        current_cost = medoid_cost(distance, medoids)
        best_cost = current_cost
        best_swap: tuple[int, int] | None = None
        non_medoids = [index for index in range(len(names)) if index not in medoids]
        for position in range(len(medoids)):
            for candidate in non_medoids:
                proposal = medoids.copy()
                proposal[position] = candidate
                cost = medoid_cost(distance, proposal)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_swap = position, candidate
        if best_swap is not None:
            medoids[best_swap[0]] = best_swap[1]
            improved = True
    return [names[index] for index in medoids]


def agglomerative_medoids(
    similarity: np.ndarray, names: list[str], k: int,
) -> list[str]:
    """Average-linkage agglomeration followed by one deterministic medoid per cluster."""

    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(names))]
    while len(clusters) > k:
        pairs = [
            (left, right)
            for left in range(len(clusters))
            for right in range(left + 1, len(clusters))
        ]

        def linkage(pair: tuple[int, int]) -> float:
            left, right = pair
            return float(np.mean(similarity[np.ix_(clusters[left], clusters[right])]))

        left, right = max(pairs, key=lambda pair: (linkage(pair), -pair[0], -pair[1]))
        merged = tuple(sorted(clusters[left] + clusters[right]))
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in {left, right}
        ]
        clusters.append(merged)
        clusters.sort()

    selected = []
    for cluster in clusters:
        medoid = max(
            cluster,
            key=lambda index: (
                float(similarity[index, list(cluster)].sum()),
                -index,
            ),
        )
        selected.append(names[medoid])
    return sorted(selected)


def leverage_score_selection(
    representation: np.ndarray, names: list[str], k: int,
) -> list[str]:
    """Select rows with largest rank-k statistical leverage scores."""

    matrix = np.asarray(representation, dtype=np.float64)
    if matrix.shape[0] != len(names):
        raise ValueError("one representation row is required per pool")
    left, _, _ = np.linalg.svd(matrix, full_matrices=False)
    count = min(k, left.shape[1])
    scores = np.sum(left[:, :count] ** 2, axis=1)
    order = sorted(range(len(names)), key=lambda index: (-float(scores[index]), names[index]))
    return [names[index] for index in order[:k]]


def logdet_score(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    return float(value) if sign > 0 else -np.inf


def dpp_greedy(
    similarity: np.ndarray, names: list[str], ranks: dict[str, float], k: int,
) -> list[str]:
    kernel = nearest_psd_correlation(similarity)
    quality = np.asarray([ranks[name] for name in names], dtype=np.float64)
    quality /= max(float(quality.max()), 1e-12)
    ensemble = np.outer(quality, quality) * kernel
    selected: list[int] = []
    while len(selected) < k:
        remaining = [index for index in range(len(names)) if index not in selected]

        def score(index: int) -> float:
            subset = selected + [index]
            matrix = ensemble[np.ix_(subset, subset)] + 1e-8 * np.eye(len(subset))
            return logdet_score(matrix)

        choice = max(remaining, key=lambda index: (score(index), -index))
        selected.append(choice)
    return [names[index] for index in selected]


def vendi_score(kernel: np.ndarray) -> float:
    values = np.maximum(np.linalg.eigvalsh((kernel + kernel.T) / 2.0), 0.0)
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    probabilities = values[values > 1e-12] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def pool_vendi_greedy(
    similarity: np.ndarray, names: list[str], ranks: dict[str, float], k: int,
) -> list[str]:
    kernel = nearest_psd_correlation(similarity)
    selected: list[int] = []
    while len(selected) < k:
        remaining = [index for index in range(len(names)) if index not in selected]

        def key(index: int) -> tuple[float, float, int]:
            subset = selected + [index]
            score = vendi_score(kernel[np.ix_(subset, subset)])
            return score, ranks[names[index]], -index

        selected.append(max(remaining, key=key))
    return [names[index] for index in selected]


def merged_vendi_greedy(features: dict[str, np.ndarray], k: int) -> list[str]:
    """Greedily maximize sample-level Vendi Score of the merged feature matrix."""
    selected: list[str] = []
    current: np.ndarray | None = None
    while len(selected) < k:
        remaining = [name for name in features if name not in selected]

        def score(name: str) -> float:
            merged = features[name] if current is None else np.concatenate([current, features[name]], axis=0)
            return feature_vendi_score(merged)

        choice = deterministic_argmax(remaining, score)
        selected.append(choice)
        current = features[choice] if current is None else np.concatenate([current, features[choice]], axis=0)
    return selected


def selection_metrics(
    selected: list[str], features: dict[str, np.ndarray], family: dict[str, str],
) -> dict[str, float | int]:
    merged = np.concatenate([features[name] for name in selected], axis=0)
    unique_families = len({family[name] for name in selected})
    return {
        "merged_reff": effective_rank(merged),
        "unique_families": unique_families,
        "duplicate_count": len(selected) - unique_families,
    }


def representation_bytes(
    representation: str, dimension: int, pool_size: int, top_k: int,
) -> int:
    if representation == "none":
        return 0
    if representation == "scalar":
        return 4
    if representation == "collapse":
        return 4 * (2 + top_k * dimension)
    if representation == "gram_sketch":
        return 8 * (top_k * dimension + 3)
    if representation == "centroid":
        return 4 * (dimension + 1)
    if representation == "subspace":
        return 4 * (top_k * dimension + 1)
    if representation in {"sample_nn", "full_features"}:
        return 4 * pool_size * dimension
    raise ValueError(representation)


def method_spec(method: str) -> tuple[str, str, bool]:
    if method == "random":
        return "none", "none", False
    if method == "rank_only":
        return "scalar", "summary", False
    if method == "rank_l_gram":
        return "gram_sketch", "summary", False
    if method == "collapse":
        return "collapse", "summary", False
    if method == "family_rank":
        return "scalar", "summary+family", True
    if method in {
        "full_merged_rank", "exact_merged_rank_greedy", "exhaustive_merged_rank_oracle",
    }:
        return "full_features", "full", False
    if method == "merged_vendi":
        return "full_features", "full", False
    algorithm, representation = method.rsplit("_", 1)
    if representation == "nn":
        representation = "sample_nn"
    return representation, "full" if representation == "sample_nn" else "summary", False


def validate_selection(selected: list[str], names: list[str], budget: int) -> None:
    if len(selected) != budget:
        raise AssertionError(f"expected {budget} selected pools, got {len(selected)}")
    if len(set(selected)) != budget:
        raise AssertionError(f"selector returned duplicates: {selected}")
    missing = set(selected) - set(names)
    if missing:
        raise AssertionError(f"selector returned unknown pools: {sorted(missing)}")


def output_paths(out_dir: Path, prefix: str, shard_index: int, shard_count: int) -> dict[str, Path]:
    suffix = f"_shard{shard_index}" if shard_count > 1 else ""
    return {
        "results": out_dir / f"{prefix}_results{suffix}.csv",
        "manifest": out_dir / f"{prefix}_manifest{suffix}.json",
        "inventory": out_dir / f"method_resource_inventory{suffix}.csv",
    }


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()
    source = {name: load_feature(args.feature_dir, name, args.encoder) for name in SOURCES}
    full_ranks = {name: effective_rank(h) for name, h in source.items()}
    if args.duplicate_source_policy == "highest_rank":
        duplicate_sources = sorted(
            full_ranks, key=lambda name: (-full_ranks[name], name),
        )[: args.n_duplicate_sources]
    else:
        duplicate_sources = sorted(np.random.default_rng(
            args.duplicate_source_seed,
        ).choice(sorted(SOURCES), args.n_duplicate_sources, replace=False).tolist())
    all_configs = [
        (seed, overlap, budget)
        for seed in CONSTRUCTION_SEEDS
        for overlap in OVERLAPS
        for budget in BUDGETS
    ]
    configs = [
        config for index, config in enumerate(all_configs)
        if index % args.shard_count == args.shard_index
    ]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    rows: list[dict[str, object]] = []
    manifest_configs: dict[str, object] = {}
    runtime: dict[str, list[float]] = defaultdict(list)
    metric_runtime: dict[str, list[float]] = defaultdict(list)
    representations = tuple(args.representations)

    for config_index, (construction_seed, overlap, budget) in enumerate(configs, start=1):
        config_start = time.perf_counter()
        features, family = make_collection(
            source, overlap, args.pool_size, construction_seed, duplicate_sources,
        )
        names = sorted(features)
        stats_start = time.perf_counter()
        ranks, _, centroids, subspaces = pool_statistics(features, args.top_k)
        gram_sketches = pool_gram_sketches(features, args.top_k)
        similarity = {
            representation: similarity_matrix(
                features, names, representation, centroids, subspaces,
            )
            for representation in representations
        }
        preprocessing_seconds = time.perf_counter() - stats_start

        selections: list[tuple[str, int, list[str]]] = []

        def timed_select(method: str, function: Callable[[], list[str]], replicate: int = -1) -> None:
            method_start = time.perf_counter()
            selected = function()
            runtime[method].append(time.perf_counter() - method_start)
            validate_selection(selected, names, budget)
            selections.append((method, replicate, selected))

        timed_select("rank_only", lambda: rank_only(ranks, budget))
        timed_select(
            "rank_l_gram",
            lambda: rank_l_gram_greedy(gram_sketches, budget),
        )
        timed_select("collapse", lambda: collapse_greedy(features, budget, args.top_k))
        timed_select("family_rank", lambda: family_rank(ranks, family, budget))
        if not args.skip_full_merged:
            timed_select(
                "exact_merged_rank_greedy",
                lambda: exact_merged_rank_greedy(features, budget),
            )
            timed_select("merged_vendi", lambda: merged_vendi_greedy(features, budget))
        centroid_matrix = np.stack([centroids[name] for name in names])
        timed_select(
            "leverage_centroid",
            lambda: leverage_score_selection(centroid_matrix, names, budget),
        )
        if args.include_exhaustive:
            timed_select(
                "exhaustive_merged_rank_oracle",
                lambda: exhaustive_merged_rank_oracle(
                    features, budget, args.max_exhaustive_combinations,
                ),
            )

        for representation in representations:
            matrix = similarity[representation]
            suffix = "sample_nn" if representation == "sample_nn" else representation
            timed_select(
                f"facility_{suffix}",
                lambda matrix=matrix: facility_location(matrix, names, budget),
            )
            timed_select(
                f"kcenter_{suffix}",
                lambda matrix=matrix: k_center(matrix, names, ranks, budget),
            )
            timed_select(
                f"kmedoids_{suffix}",
                lambda matrix=matrix: k_medoids_pam(matrix, names, budget),
            )
            timed_select(
                f"agglomerative_{suffix}",
                lambda matrix=matrix: agglomerative_medoids(matrix, names, budget),
            )
            timed_select(
                f"dpp_{suffix}",
                lambda matrix=matrix: dpp_greedy(matrix, names, ranks, budget),
            )
            timed_select(
                f"pool_vendi_{suffix}",
                lambda matrix=matrix: pool_vendi_greedy(matrix, names, ranks, budget),
            )

        rng = np.random.default_rng(
            construction_seed + budget * 1000 + int(overlap * 100) + args.random_seed_offset
        )
        random_start = time.perf_counter()
        for replicate in range(args.n_random):
            selected = sorted(rng.choice(names, budget, replace=False).tolist())
            selections.append(("random", replicate, selected))
        runtime["random"].append((time.perf_counter() - random_start) / max(args.n_random, 1))

        config_key = f"seed{construction_seed}_overlap{overlap:.2f}_k{budget}"
        config_methods: dict[str, object] = {}
        for method, replicate, selected in selections:
            metric_start = time.perf_counter()
            metrics = selection_metrics(selected, features, family)
            metric_runtime[method].append(time.perf_counter() - metric_start)
            selection_key = f"{method}_{replicate:03d}" if replicate >= 0 else method
            config_methods[selection_key] = {"selected": selected, **metrics}
            rows.append({
                "config": config_key,
                "construction_seed": construction_seed,
                "overlap": overlap,
                "budget": budget,
                "method": method,
                "replicate": replicate,
                "selected": "|".join(selected),
                "selection_seconds": runtime[method][-1] if method != "random" else runtime["random"][-1],
                "metric_seconds": metric_runtime[method][-1],
                "preprocessing_seconds": preprocessing_seconds,
                **metrics,
            })
        manifest_configs[config_key] = {"family": family, "methods": config_methods}
        elapsed = time.perf_counter() - config_start
        print(
            f"[{config_index}/{len(configs)}] {config_key} rows={len(selections)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    paths = output_paths(args.out_dir, args.output_prefix, args.shard_index, args.shard_count)
    if rows:
        with paths["results"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    dimension = next(iter(source.values())).shape[1]
    method_names = sorted(runtime)
    inventory_rows = []
    for method in method_names:
        representation, information_class, uses_family = method_spec(method)
        bytes_per_pool = representation_bytes(
            representation, dimension, args.pool_size, args.top_k,
        )
        selection_times = np.asarray(runtime[method], dtype=np.float64)
        evaluation_times = np.asarray(metric_runtime[method], dtype=np.float64)
        inventory_rows.append({
            "method": method,
            "information_class": information_class,
            "representation": representation,
            "uses_labels": False,
            "uses_family_metadata": uses_family,
            "uses_target_data": False,
            "bytes_per_pool": bytes_per_pool,
            "bytes_for_collection": bytes_per_pool * (len(SOURCES) + args.n_duplicate_sources),
            "selection_runs": len(selection_times),
            "selection_seconds_mean": float(selection_times.mean()),
            "selection_seconds_max": float(selection_times.max()),
            "metric_seconds_mean": float(evaluation_times.mean()),
        })
    if inventory_rows:
        with paths["inventory"].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(inventory_rows[0]))
            writer.writeheader()
            writer.writerows(inventory_rows)

    script_path = Path(__file__)
    manifest = {
        "start_time_utc": start_time,
        "end_time_utc": utc_now(),
        "pid": os.getpid(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "feature_dir": str(args.feature_dir),
        "encoder": args.encoder,
        "sources": SOURCES,
        "construction_seeds": CONSTRUCTION_SEEDS,
        "overlaps": OVERLAPS,
        "budgets": BUDGETS,
        "pool_size": args.pool_size,
        "top_k": args.top_k,
        "representations": list(representations),
        "n_random": args.n_random,
        "duplicate_sources": duplicate_sources,
        "duplicate_source_policy": args.duplicate_source_policy,
        "duplicate_source_seed": args.duplicate_source_seed,
        "include_exhaustive": args.include_exhaustive,
        "max_exhaustive_combinations": args.max_exhaustive_combinations,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "configs": manifest_configs,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument("--pool-size", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-duplicate-sources", type=int, default=6)
    parser.add_argument(
        "--duplicate-source-policy", choices=["random", "highest_rank"], default="random",
    )
    parser.add_argument("--duplicate-source-seed", type=int, default=20260908)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--random-seed-offset", type=int, default=0)
    parser.add_argument(
        "--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS),
    )
    parser.add_argument("--skip-full-merged", action="store_true")
    parser.add_argument("--include-exhaustive", action="store_true")
    parser.add_argument("--max-exhaustive-combinations", type=int, default=10_000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--output-prefix", default="classic_screening")
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    if args.n_random < 1:
        parser.error("--n-random must be at least 1")
    if not 1 <= args.n_duplicate_sources <= len(SOURCES):
        parser.error(f"--n-duplicate-sources must be in [1, {len(SOURCES)}]")
    return args


if __name__ == "__main__":
    run(parse_args())
