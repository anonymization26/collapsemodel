#!/usr/bin/env python3
"""E1 synthetic experiments for target-conditioned data selection.

The main table evaluates exact target Bayes risk under the shared linear model.
A separate conditional-shift table deliberately violates that model and records
how unlabeled geometry can cease to predict empirical target utility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.collapse_core import effective_rank_from_scatter  # noqa: E402
from metrics.target_conditioned import (  # noqa: E402
    PSDEigendecomposition,
    RidgeStatistics,
    adaptive_sketch_target_a,
    aggregate_gram,
    candidate_sketch_intervals,
    certified_minimum,
    decompose_psd,
    gram_from_features,
    greedy_d_optimal,
    greedy_target_a,
    combine_ridge_statistics,
    ridge_squared_risk,
    ridge_statistics,
    second_moment,
    sketch_from_decomposition,
    target_a_objective,
)


@dataclass(frozen=True)
class SyntheticProblem:
    features: dict[str, np.ndarray]
    blocks: dict[str, np.ndarray]
    source_directions: dict[str, np.ndarray]
    target_moment: np.ndarray
    estimated_target_moment: np.ndarray
    prior_precision: np.ndarray
    noise_variance: float


def git_revision() -> str:
    injected = os.environ.get("COLLAPSEMODEL_GIT_REVISION")
    if injected:
        return injected
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_covariance(
    rng: np.random.Generator,
    covariance: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return rng.normal(size=(n_samples, covariance.shape[0])) @ factor.T


def make_synthetic_problem(
    seed: int,
    dimension: int,
    candidate_count: int,
    source_samples: int,
    target_samples: int,
    target_rank: int,
    noise_variance: float = 1.0,
) -> SyntheticProblem:
    """Construct candidates spanning a controlled target-to-orthogonal arc."""

    if dimension < 4:
        raise ValueError("dimension must be at least 4")
    if not 1 <= target_rank <= dimension // 2:
        raise ValueError("target_rank must be in [1, dimension // 2]")
    if min(candidate_count, source_samples, target_samples) <= 0:
        raise ValueError("candidate and sample counts must be positive")
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    target_basis = basis[:, :target_rank]
    orthogonal_basis = basis[:, target_rank:2 * target_rank]
    target_eigenvalues = np.geomspace(2.0, 0.5, target_rank)
    background = 0.02
    target_moment = (
        target_basis @ np.diag(target_eigenvalues) @ target_basis.T
        + background * np.eye(dimension)
    )
    target_features = sample_covariance(rng, target_moment, target_samples)
    estimated_target_moment = second_moment(target_features)

    angles = np.linspace(0.0, math.pi / 2.0, candidate_count)
    features = {}
    blocks = {}
    source_directions = {}
    for index, angle in enumerate(angles):
        rotated_basis = (
            math.cos(float(angle)) * target_basis
            + math.sin(float(angle)) * orthogonal_basis
        )
        decay = (0.25, 0.75, 1.5)[index % 3]
        if target_rank == 1:
            source_eigenvalues = np.ones(1)
        else:
            source_eigenvalues = np.exp(
                -decay * np.arange(target_rank) / (target_rank - 1)
            )
            source_eigenvalues /= source_eigenvalues.mean()
        scale = 0.8 + 1.7 * math.sin(float(angle)) ** 2
        covariance = (
            scale
            * rotated_basis
            @ np.diag(source_eigenvalues)
            @ rotated_basis.T
            + background * np.eye(dimension)
        )
        name = f"pool_{index:02d}"
        source = sample_covariance(rng, covariance, source_samples)
        features[name] = source
        blocks[name] = gram_from_features(source)
        direction = rng.normal(size=dimension)
        direction /= max(float(np.linalg.norm(direction)), np.finfo(float).tiny)
        source_directions[name] = direction
    return SyntheticProblem(
        features=features,
        blocks=blocks,
        source_directions=source_directions,
        target_moment=target_moment,
        estimated_target_moment=estimated_target_moment,
        prior_precision=np.eye(dimension),
        noise_variance=noise_variance,
    )


def synthetic_problem_seed(
    seed: int,
    dimension: int,
    candidate_count: int,
    target_samples: int,
    target_rank: int,
) -> int:
    """Return a stable seed for one budget-independent synthetic problem."""

    return (
        seed
        + 1009 * dimension
        + 9176 * candidate_count
        + 37 * target_samples
        + 65537 * target_rank
    )


def deterministic_score_greedy(
    blocks: Mapping[str, np.ndarray],
    k: int,
    score,
) -> tuple[str, ...]:
    selected = []
    current = np.zeros_like(next(iter(blocks.values())))
    for _ in range(k):
        best_name = None
        best_score = -math.inf
        for name in sorted(blocks):
            if name in selected:
                continue
            value = float(score(current + blocks[name]))
            if value > best_score + 1e-13:
                best_name = name
                best_score = value
        assert best_name is not None
        selected.append(best_name)
        current += blocks[best_name]
    return tuple(selected)


def effective_rank_greedy(
    blocks: Mapping[str, np.ndarray],
    k: int,
) -> tuple[str, ...]:
    return deterministic_score_greedy(
        blocks, k, lambda gram: effective_rank_from_scatter(gram)
    )


def trace_greedy(
    blocks: Mapping[str, np.ndarray],
    k: int,
) -> tuple[str, ...]:
    return deterministic_score_greedy(blocks, k, np.trace)


def sketch_a_greedy(
    problem: SyntheticProblem,
    k: int,
    rank: int,
    decompositions: Mapping[str, PSDEigendecomposition] | None = None,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if decompositions is None:
        decompositions = {
            name: decompose_psd(gram) for name, gram in problem.blocks.items()
        }
    sketches = {
        name: sketch_from_decomposition(
            decompositions[name], min(rank, problem.blocks[name].shape[0])
        )
        for name in problem.blocks
    }
    selected = []
    certified_steps = 0
    certified_comparisons = 0
    total_comparisons = 0
    covered_intervals = 0
    total_intervals = 0
    widths = []
    for _ in range(k):
        candidates = [name for name in sorted(sketches) if name not in selected]
        intervals = candidate_sketch_intervals(
            selected,
            candidates,
            sketches,
            problem.estimated_target_moment,
            problem.prior_precision,
            problem.noise_variance,
        )
        choice = min(candidates, key=lambda name: (intervals[name].upper, name))
        certificate = certified_minimum(intervals)
        certified_steps += int(certificate == choice)
        certified_comparisons += sum(
            intervals[choice].upper < intervals[name].lower
            for name in candidates
            if name != choice
        )
        total_comparisons += max(len(candidates) - 1, 0)
        prefix_gram = aggregate_gram(problem.blocks, selected)
        for name in candidates:
            exact = target_a_objective(
                problem.estimated_target_moment,
                problem.prior_precision,
                prefix_gram + problem.blocks[name],
                problem.noise_variance,
            )
            interval = intervals[name]
            tolerance = 1e-10 * max(abs(exact), 1.0)
            covered_intervals += int(
                interval.lower - tolerance <= exact <= interval.upper + tolerance
            )
            total_intervals += 1
        widths.append(intervals[choice].width)
        selected.append(choice)
    full_bytes = len(problem.blocks) * problem.prior_precision.shape[0] ** 2 * 8
    sketch_bytes = sum(sketch.storage_bytes for sketch in sketches.values())
    return tuple(selected), {
        "certified_steps": certified_steps,
        "certificate_rate": certified_steps / k,
        "certified_comparisons": certified_comparisons,
        "total_comparisons": total_comparisons,
        "pair_certificate_rate": (
            certified_comparisons / total_comparisons
            if total_comparisons
            else 1.0
        ),
        "covered_intervals": covered_intervals,
        "total_intervals": total_intervals,
        "interval_coverage_rate": covered_intervals / total_intervals,
        "mean_selected_interval_width": float(np.mean(widths)),
        "sketch_bytes": sketch_bytes,
        "full_gram_bytes": full_bytes,
        "byte_ratio": sketch_bytes / full_bytes,
    }


def enumerate_combination_risks(
    problem: SyntheticProblem,
    k: int,
    max_combinations: int,
) -> list[tuple[tuple[str, ...], float]]:
    names = sorted(problem.blocks)
    count = math.comb(len(names), k)
    if count > max_combinations:
        raise ValueError(
            f"configuration requires {count} combinations; "
            f"increase --max-combinations above {max_combinations}"
        )
    values = []
    for selected in itertools.combinations(names, k):
        risk = target_a_objective(
            problem.target_moment,
            problem.prior_precision,
            aggregate_gram(problem.blocks, selected),
            problem.noise_variance,
        )
        values.append((selected, risk))
    return sorted(values, key=lambda item: (item[1], item[0]))


def selection_record(
    method: str,
    selected: Iterable[str],
    problem: SyntheticProblem,
    ordered_oracle: list[tuple[tuple[str, ...], float]],
    config: dict[str, int],
    diagnostics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    chosen = tuple(sorted(selected))
    risk = target_a_objective(
        problem.target_moment,
        problem.prior_precision,
        aggregate_gram(problem.blocks, chosen),
        problem.noise_variance,
    )
    oracle_risk = ordered_oracle[0][1]
    ranks = {combination: rank + 1 for rank, (combination, _) in enumerate(ordered_oracle)}
    q = min(10, len(ordered_oracle))
    record = {
        **config,
        "method": method,
        "selected": "|".join(chosen),
        "target_risk": risk,
        "oracle_risk": oracle_risk,
        "normalized_regret": (risk - oracle_risk) / max(abs(oracle_risk), 1e-15),
        "combination_rank": ranks[chosen],
        "in_top_q": int(ranks[chosen] <= q),
        "top_q": q,
    }
    if diagnostics:
        record.update(diagnostics)
    return record


def shared_model_records(
    problem: SyntheticProblem,
    seed: int,
    dimension: int,
    candidate_count: int,
    budget: int,
    target_samples: int,
    target_rank: int,
    sketch_ranks: list[int],
    random_repeats: int,
    max_combinations: int,
    decompositions: Mapping[str, PSDEigendecomposition] | None = None,
) -> list[dict[str, object]]:
    config = {
        "seed": seed,
        "dimension": dimension,
        "candidate_count": candidate_count,
        "budget": budget,
        "target_samples": target_samples,
        "target_rank": target_rank,
    }
    ordered_oracle = enumerate_combination_risks(
        problem, budget, max_combinations
    )
    target_result = greedy_target_a(
        problem.blocks,
        problem.estimated_target_moment,
        problem.prior_precision,
        budget,
        problem.noise_variance,
    )
    exact_target_result = greedy_target_a(
        problem.blocks,
        problem.target_moment,
        problem.prior_precision,
        budget,
        problem.noise_variance,
    )
    isotropic_result = greedy_target_a(
        problem.blocks,
        np.eye(dimension) / dimension,
        problem.prior_precision,
        budget,
        problem.noise_variance,
    )
    d_opt_result = greedy_d_optimal(
        problem.blocks,
        problem.prior_precision,
        budget,
        problem.noise_variance,
    )
    methods: list[tuple[str, Iterable[str], Mapping[str, float] | None]] = [
        ("target_a_estimated", target_result.selected, None),
        ("target_a_true_ct_diagnostic", exact_target_result.selected, None),
        ("isotropic_a", isotropic_result.selected, None),
        ("bayesian_d", d_opt_result.selected, None),
        ("effective_rank", effective_rank_greedy(problem.blocks, budget), None),
        ("trace", trace_greedy(problem.blocks, budget), None),
        ("oracle_exhaustive", ordered_oracle[0][0], None),
    ]
    if decompositions is None:
        decompositions = {
            name: decompose_psd(gram) for name, gram in problem.blocks.items()
        }
    for rank in sorted(set(sketch_ranks)):
        selected, diagnostics = sketch_a_greedy(
            problem, budget, rank, decompositions=decompositions
        )
        methods.append((f"target_a_sketch_l{rank}", selected, diagnostics))
    adaptive = adaptive_sketch_target_a(
        problem.blocks,
        problem.estimated_target_moment,
        problem.prior_precision,
        budget,
        rank_schedule=sketch_ranks,
        noise_variance=problem.noise_variance,
        fallback_to_full=True,
        decompositions=decompositions,
    )
    methods.append((
        "target_a_sketch_adaptive",
        adaptive.selected,
        {
            "certified_steps": sum(step.certified for step in adaptive.steps),
            "certificate_rate": adaptive.certificate_rate,
            "uncertified_fallback_steps": sum(
                step.used_uncertified_fallback for step in adaptive.steps
            ),
            "mean_refinement_rounds": float(np.mean([
                step.refinement_rounds for step in adaptive.steps
            ])),
            "sketch_bytes": adaptive.transmitted_bytes,
            "full_gram_bytes": adaptive.full_gram_bytes,
            "byte_ratio": adaptive.byte_ratio,
        },
    ))
    rng = np.random.default_rng(seed + 700_001 + budget)
    names = np.array(sorted(problem.blocks), dtype=object)
    for repeat in range(random_repeats):
        selected = tuple(sorted(rng.choice(names, budget, replace=False).tolist()))
        methods.append(("random", selected, {"random_repeat": repeat}))
    return [
        selection_record(
            method, selected, problem, ordered_oracle, config, diagnostics
        )
        for method, selected, diagnostics in methods
    ]


def conditional_shift_records(
    problem: SyntheticProblem,
    seed: int,
    dimension: int,
    candidate_count: int,
    budget: int,
    target_samples: int,
    target_rank: int,
    shift_levels: list[float],
    target_test_samples: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + 1_900_003 + budget)
    true_weights = rng.normal(size=dimension) / np.sqrt(dimension)
    target_x = sample_covariance(rng, problem.target_moment, target_test_samples)
    target_y = target_x @ true_weights
    target_statistics = ridge_statistics(target_x, target_y)
    selections = {
        "target_a_estimated": greedy_target_a(
            problem.blocks,
            problem.estimated_target_moment,
            problem.prior_precision,
            budget,
            problem.noise_variance,
        ).selected,
        "isotropic_a": greedy_target_a(
            problem.blocks,
            np.eye(dimension) / dimension,
            problem.prior_precision,
            budget,
            problem.noise_variance,
        ).selected,
        "bayesian_d": greedy_d_optimal(
            problem.blocks,
            problem.prior_precision,
            budget,
            problem.noise_variance,
        ).selected,
        "effective_rank": effective_rank_greedy(problem.blocks, budget),
    }
    records = []
    names = sorted(problem.features)
    for shift in shift_levels:
        source_statistics = {}
        for name in names:
            shifted_weights = true_weights + shift * problem.source_directions[name]
            gram = problem.blocks[name]
            source_statistics[name] = RidgeStatistics(
                gram=gram,
                cross=gram @ shifted_weights[:, None],
                response_norm=float(shifted_weights @ gram @ shifted_weights),
                n_samples=problem.features[name].shape[0],
            )
        combination_risks = []
        for selected in itertools.combinations(names, budget):
            risk = ridge_squared_risk(
                combine_ridge_statistics(
                    source_statistics[name] for name in selected
                ),
                target_statistics,
                regularization=problem.noise_variance,
            )
            combination_risks.append((selected, risk))
        combination_risks.sort(key=lambda item: (item[1], item[0]))
        oracle_risk = combination_risks[0][1]
        ranks = {
            selected: rank + 1
            for rank, (selected, _) in enumerate(combination_risks)
        }
        selections_with_oracle = dict(selections)
        selections_with_oracle["conditional_oracle"] = combination_risks[0][0]
        for method, selected in selections_with_oracle.items():
            chosen = tuple(sorted(selected))
            risk = ridge_squared_risk(
                combine_ridge_statistics(
                    source_statistics[name] for name in chosen
                ),
                target_statistics,
                regularization=problem.noise_variance,
            )
            records.append({
                "seed": seed,
                "dimension": dimension,
                "candidate_count": candidate_count,
                "budget": budget,
                "target_samples": target_samples,
                "target_rank": target_rank,
                "shift": shift,
                "method": method,
                "selected": "|".join(chosen),
                "target_mse": risk,
                "conditional_oracle_mse": oracle_risk,
                "normalized_regret": (
                    (risk - oracle_risk) / max(abs(oracle_risk), 1e-15)
                ),
                "combination_rank": ranks[chosen],
            })
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else math.nan


CONFIGURATION_FIELDS = (
    "seed",
    "dimension",
    "candidate_count",
    "budget",
    "target_samples",
    "target_rank",
)


def configuration_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in CONFIGURATION_FIELDS)


def h2_pilot_gate(shared_rows: list[dict[str, object]]) -> dict[str, object]:
    """Evaluate the preregistered H2 gate on deterministic synthetic methods."""

    target_rows = {
        configuration_key(row): row
        for row in shared_rows
        if row["method"] == "target_a_estimated"
    }
    baseline_methods = ("isotropic_a", "bayesian_d", "effective_rank", "trace")
    baseline_scores = {}
    baseline_maps = {}
    for method in baseline_methods:
        rows = {
            configuration_key(row): row
            for row in shared_rows
            if row["method"] == method
        }
        if set(rows) != set(target_rows):
            raise ValueError(f"H2 baseline {method} has incomplete configurations")
        baseline_maps[method] = rows
        baseline_scores[method] = mean(
            float(row["normalized_regret"]) for row in rows.values()
        )
    strongest = min(baseline_methods, key=lambda name: (baseline_scores[name], name))
    differences = np.array([
        (
            float(target_rows[key]["target_risk"])
            - float(baseline_maps[strongest][key]["target_risk"])
        )
        / max(abs(float(baseline_maps[strongest][key]["target_risk"])), 1e-15)
        for key in sorted(target_rows)
    ])
    rng = np.random.default_rng(20260910)
    bootstrap_means = np.mean(
        rng.choice(differences, size=(10_000, len(differences)), replace=True),
        axis=1,
    )
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    mean_difference = float(np.mean(differences))
    noninferior_fraction = float(np.mean(differences <= 0.005))
    checks = {
        "mean_improvement_above_2_percent": mean_difference < -0.02,
        "paired_bootstrap_upper_below_zero": float(upper) < 0.0,
        "noninferior_on_at_least_70_percent": noninferior_fraction >= 0.70,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "strongest_target_blind_baseline": strongest,
        "baseline_mean_normalized_regret": baseline_scores,
        "n_configurations": len(differences),
        "mean_relative_risk_difference": mean_difference,
        "mean_relative_improvement": -mean_difference,
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "noninferior_fraction_at_0_5_percent_tolerance": noninferior_fraction,
        "checks": checks,
    }


def h5_pilot_gate(shared_rows: list[dict[str, object]]) -> dict[str, object]:
    """Evaluate selection fidelity and communication for adaptive sketches."""

    exact = {
        configuration_key(row): row
        for row in shared_rows
        if row["method"] == "target_a_estimated"
    }
    adaptive = {
        configuration_key(row): row
        for row in shared_rows
        if row["method"] == "target_a_sketch_adaptive"
    }
    if set(exact) != set(adaptive):
        raise ValueError("adaptive sketch rows do not match full A-opt configurations")
    selection_match_rate = mean(
        float(adaptive[key]["selected"] == exact[key]["selected"])
        for key in exact
    )
    certificate_rate = mean(
        float(row["certificate_rate"]) for row in adaptive.values()
    )
    byte_ratio = mean(float(row["byte_ratio"]) for row in adaptive.values())
    checks = {
        "same_final_selection": selection_match_rate == 1.0,
        "direct_certificate_path": certificate_rate >= 0.5 and byte_ratio <= 0.25,
        "adaptive_fallback_path": selection_match_rate == 1.0 and byte_ratio <= 0.5,
    }
    passed = checks["same_final_selection"] and (
        checks["direct_certificate_path"] or checks["adaptive_fallback_path"]
    )
    return {
        "status": "passed" if passed else "failed",
        "n_configurations": len(exact),
        "selection_match_rate": selection_match_rate,
        "mean_certificate_rate": certificate_rate,
        "mean_byte_ratio": byte_ratio,
        "checks": checks,
    }


def summarize(
    shared_rows: list[dict[str, object]],
    shift_rows: list[dict[str, object]],
) -> dict[str, object]:
    shared_methods = sorted({str(row["method"]) for row in shared_rows})
    shared_summary = {}
    for method in shared_methods:
        rows = [row for row in shared_rows if row["method"] == method]
        shared_summary[method] = {
            "n": len(rows),
            "mean_normalized_regret": mean(
                float(row["normalized_regret"]) for row in rows
            ),
            "median_normalized_regret": float(np.median([
                float(row["normalized_regret"]) for row in rows
            ])),
            "top_q_recall": mean(float(row["in_top_q"]) for row in rows),
            "mean_combination_rank": mean(
                float(row["combination_rank"]) for row in rows
            ),
        }
        certificate_rows = [
            row
            for row in rows
            if row.get("certificate_rate") not in (None, "")
        ]
        if certificate_rows:
            shared_summary[method]["mean_certificate_rate"] = mean(
                float(row["certificate_rate"]) for row in certificate_rows
            )
            shared_summary[method]["mean_byte_ratio"] = mean(
                float(row["byte_ratio"]) for row in certificate_rows
            )

    shift_summary = {}
    shift_keys = sorted({float(row["shift"]) for row in shift_rows})
    for shift in shift_keys:
        shift_summary[str(shift)] = {}
        methods = sorted({
            str(row["method"]) for row in shift_rows if float(row["shift"]) == shift
        })
        for method in methods:
            rows = [
                row for row in shift_rows
                if float(row["shift"]) == shift and row["method"] == method
            ]
            shift_summary[str(shift)][method] = {
                "n": len(rows),
                "mean_target_mse": mean(float(row["target_mse"]) for row in rows),
                "mean_normalized_regret": mean(
                    float(row["normalized_regret"]) for row in rows
                ),
                "mean_combination_rank": mean(
                    float(row["combination_rank"]) for row in rows
                ),
            }
    return {
        "shared_model": shared_summary,
        "conditional_shift": shift_summary,
        "gates": {
            "h2_pilot": h2_pilot_gate(shared_rows),
            "h5_pilot": h5_pilot_gate(shared_rows),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "target_conditioned" / "e1_synthetic",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910, 20260911, 20260912])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--target-samples", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--target-rank-fractions", type=float, nargs="+", default=[0.25]
    )
    parser.add_argument("--source-samples", type=int, default=64)
    parser.add_argument("--target-test-samples", type=int, default=512)
    parser.add_argument("--sketch-ranks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--shift-levels", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--max-combinations", type=int, default=100_000)
    parser.add_argument("--mini", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    integer_lists = [
        args.seeds,
        args.dimensions,
        args.candidate_counts,
        args.budgets,
        args.target_samples,
        args.sketch_ranks,
    ]
    if any(not values for values in integer_lists):
        raise ValueError("grid arguments must be nonempty")
    if any(value <= 0 for values in integer_lists[1:] for value in values):
        raise ValueError("dimensions, counts, budgets, samples, and ranks must be positive")
    if any(dimension < 4 for dimension in args.dimensions):
        raise ValueError("dimensions must be at least 4")
    if max(args.sketch_ranks) > min(args.dimensions):
        raise ValueError("E1 sketch ranks cannot exceed the smallest dimension")
    if min(args.source_samples, args.target_test_samples, args.random_repeats) <= 0:
        raise ValueError("sample and repeat counts must be positive")
    if any(shift < 0.0 for shift in args.shift_levels):
        raise ValueError("shift levels must be nonnegative")
    if any(
        not math.isfinite(fraction) or fraction <= 0.0 or fraction > 0.5
        for fraction in args.target_rank_fractions
    ):
        raise ValueError("target rank fractions must lie in (0, 0.5]")
    for count in args.candidate_counts:
        if any(budget > count for budget in args.budgets):
            raise ValueError("selection budget cannot exceed candidate count")
    for dimension in args.dimensions:
        target_ranks = [
            max(1, min(int(round(dimension * fraction)), dimension // 2))
            for fraction in args.target_rank_fractions
        ]
        if len(set(target_ranks)) != len(target_ranks):
            raise ValueError(
                "target rank fractions collapse to duplicate ranks at "
                f"dimension {dimension}: {target_ranks}"
            )


def main() -> int:
    args = parse_args()
    if args.mini:
        args.seeds = args.seeds[:1]
        args.dimensions = [min(args.dimensions)]
        args.candidate_counts = [min(args.candidate_counts)]
        args.budgets = [min(3, min(args.candidate_counts))]
        args.target_samples = [min(args.target_samples)]
        args.target_rank_fractions = [0.25]
        args.sketch_ranks = sorted(set([2, min(4, min(args.dimensions))]))
        args.shift_levels = sorted(set([0.0, max(args.shift_levels)]))
        args.random_repeats = min(args.random_repeats, 5)
        args.target_test_samples = min(args.target_test_samples, 256)
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    shared_rows = []
    shift_rows = []
    total_configurations = (
        len(args.seeds)
        * len(args.dimensions)
        * len(args.candidate_counts)
        * len(args.budgets)
        * len(args.target_samples)
        * len(args.target_rank_fractions)
    )
    completed = 0
    for (
        seed,
        dimension,
        candidate_count,
        target_samples,
        target_rank_fraction,
    ) in itertools.product(
        args.seeds,
        args.dimensions,
        args.candidate_counts,
        args.target_samples,
        args.target_rank_fractions,
    ):
        target_rank = max(
            1, min(int(round(dimension * target_rank_fraction)), dimension // 2)
        )
        problem_seed = synthetic_problem_seed(
            seed,
            dimension,
            candidate_count,
            target_samples,
            target_rank,
        )
        problem = make_synthetic_problem(
            seed=problem_seed,
            dimension=dimension,
            candidate_count=candidate_count,
            source_samples=args.source_samples,
            target_samples=target_samples,
            target_rank=target_rank,
        )
        decompositions = {
            name: decompose_psd(gram) for name, gram in problem.blocks.items()
        }
        for budget in args.budgets:
            shared_rows.extend(shared_model_records(
                problem=problem,
                seed=seed,
                dimension=dimension,
                candidate_count=candidate_count,
                budget=budget,
                target_samples=target_samples,
                target_rank=target_rank,
                sketch_ranks=args.sketch_ranks,
                random_repeats=args.random_repeats,
                max_combinations=args.max_combinations,
                decompositions=decompositions,
            ))
            shift_rows.extend(conditional_shift_records(
                problem=problem,
                seed=seed,
                dimension=dimension,
                candidate_count=candidate_count,
                budget=budget,
                target_samples=target_samples,
                target_rank=target_rank,
                shift_levels=args.shift_levels,
                target_test_samples=args.target_test_samples,
            ))
            completed += 1
            print(
                f"[{completed}/{total_configurations}] seed={seed} d={dimension} "
                f"M={candidate_count} K={budget} nT={target_samples} "
                f"rT={target_rank}",
                flush=True,
            )

    write_csv(args.out_dir / "shared_model_results.csv", shared_rows)
    write_csv(args.out_dir / "conditional_shift_results.csv", shift_rows)
    summary = summarize(shared_rows, shift_rows)
    elapsed = time.monotonic() - started
    report = {
        "experiment": "E1 target-conditioned synthetic selection",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "config": {
            "seeds": args.seeds,
            "dimensions": args.dimensions,
            "candidate_counts": args.candidate_counts,
            "budgets": args.budgets,
            "target_samples": args.target_samples,
            "target_rank_fractions": args.target_rank_fractions,
            "source_samples": args.source_samples,
            "target_test_samples": args.target_test_samples,
            "sketch_ranks": args.sketch_ranks,
            "shift_levels": args.shift_levels,
            "random_repeats": args.random_repeats,
            "max_combinations": args.max_combinations,
            "mini": args.mini,
        },
        "counts": {
            "configurations": total_configurations,
            "shared_model_rows": len(shared_rows),
            "conditional_shift_rows": len(shift_rows),
        },
        "summary": summary,
        "provenance": {
            "git_revision": git_revision(),
            "script_sha256": sha256_file(Path(__file__)),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(report["config"], indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": report["experiment"],
        "status": report["status"],
        "created_utc": report["created_utc"],
        "elapsed_seconds": elapsed,
        "config": report["config"],
        "counts": report["counts"],
        "provenance": report["provenance"],
        "outputs": [
            "conditional_shift_results.csv",
            "config.json",
            "manifest.json",
            "README.md",
            "shared_model_results.csv",
            "summary.json",
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# E1 Target-Conditioned Synthetic Selection

Status: completed experimental run; results are synthetic evidence only.

- Configurations: {total_configurations}
- Shared-model rows: {len(shared_rows)}
- Conditional-shift rows: {len(shift_rows)}
- Elapsed seconds: {elapsed:.3f}
- Git revision: `{report['provenance']['git_revision']}`

`shared_model_results.csv` evaluates the theorem-aligned Bayes target risk.
`conditional_shift_results.csv` violates the shared conditional model and is a
required failure-boundary diagnostic. `summary.json` is generated only from the
two raw tables.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "configurations": total_configurations,
        "shared_model_rows": len(shared_rows),
        "conditional_shift_rows": len(shift_rows),
        "elapsed_seconds": elapsed,
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
