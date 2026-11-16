#!/usr/bin/env python3
"""E0 validation for target-conditioned data selection.

This script validates algebraic identities, optimization structure, sketch
coverage, and the label-blind non-identifiability counterexample before any
large experiment is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned import (  # noqa: E402
    adaptive_sketch_target_a,
    aggregate_gram,
    block_marginal_gain,
    continuous_target_a,
    d_opt_information_gain,
    gram_from_features,
    greedy_target_a,
    make_psd_sketch,
    posterior_covariance,
    ridge_squared_risk,
    ridge_statistics,
    sample_marginal_gain,
    sketch_target_a_interval,
    target_a_objective,
    target_a_risk,
)


FORMULA_RELATIVE_TOLERANCE = 1e-10
MONTE_CARLO_RELATIVE_TOLERANCE = 1e-3
CONVEXITY_TOLERANCE = 1e-9
SUBMODULARITY_TOLERANCE = 1e-10
INTERVAL_TOLERANCE = 1e-10


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), np.finfo(float).tiny)


def git_revision() -> str:
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


def random_psd(rng: np.random.Generator, dimension: int, rank: int) -> np.ndarray:
    factor = rng.normal(size=(rank, dimension))
    return factor.T @ factor / max(rank, 1)


def monte_carlo_bayes_risk(
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float]:
    dimension = 4
    prior_covariance = np.diag([1.4, 0.9, 0.5, 0.2])
    prior_precision = np.diag(1.0 / np.diag(prior_covariance))
    target_moment = np.diag([2.0, 1.0, 0.4, 0.1])
    design = rng.normal(size=(7, dimension))
    noise_variance = 0.6
    posterior = posterior_covariance(
        prior_precision, gram_from_features(design), noise_variance
    )
    analytic_reducible = target_a_risk(target_moment, posterior)
    analytic_total = noise_variance + analytic_reducible

    sum_risk = 0.0
    sum_squared = 0.0
    completed = 0
    batch_size = 100_000
    prior_chol = np.linalg.cholesky(prior_covariance)
    while completed < draws:
        count = min(batch_size, draws - completed)
        theta = rng.normal(size=(count, dimension)) @ prior_chol.T
        train_noise = np.sqrt(noise_variance) * rng.normal(size=(count, len(design)))
        responses = theta @ design.T + train_noise
        posterior_mean = (responses @ design / noise_variance) @ posterior
        error = theta - posterior_mean
        conditional_risk = noise_variance + np.einsum(
            "ni,ij,nj->n", error, target_moment, error
        )
        sum_risk += float(conditional_risk.sum())
        sum_squared += float(np.square(conditional_risk).sum())
        completed += count
    empirical = sum_risk / draws
    variance = max(sum_squared / draws - empirical * empirical, 0.0)
    standard_error = float(np.sqrt(variance / draws))
    return {
        "draws": draws,
        "analytic_total_risk": analytic_total,
        "monte_carlo_total_risk": empirical,
        "relative_error": relative_error(empirical, analytic_total),
        "standard_error": standard_error,
        "z_score": abs(empirical - analytic_total) / max(standard_error, 1e-15),
    }


def marginal_identity_checks(
    rng: np.random.Generator,
    trials: int,
) -> dict[str, float]:
    sample_errors = []
    block_errors = []
    for _ in range(trials):
        dimension = 6
        prior = random_psd(rng, dimension, dimension) + np.eye(dimension)
        target = random_psd(rng, dimension, 4)
        base = rng.normal(size=(8, dimension))
        base_gram = gram_from_features(base)
        noise_variance = float(rng.uniform(0.3, 2.0))
        posterior = posterior_covariance(prior, base_gram, noise_variance)
        before = target_a_risk(target, posterior)

        feature = rng.normal(size=dimension)
        weight = float(rng.uniform(0.1, 3.0))
        after_sample = target_a_objective(
            target,
            prior,
            base_gram + weight * np.outer(feature, feature),
            noise_variance,
        )
        sample_gain = sample_marginal_gain(
            posterior, target, feature, noise_variance, weight
        )
        sample_errors.append(relative_error(sample_gain, before - after_sample))

        block = rng.normal(size=(3, dimension))
        after_block = target_a_objective(
            target,
            prior,
            base_gram + gram_from_features(block),
            noise_variance,
        )
        block_gain = block_marginal_gain(
            posterior, target, block, noise_variance
        )
        block_errors.append(relative_error(block_gain, before - after_block))
    return {
        "trials": trials,
        "sample_max_relative_error": float(max(sample_errors)),
        "block_max_relative_error": float(max(block_errors)),
    }


def convexity_checks(
    rng: np.random.Generator,
    trials: int,
) -> dict[str, float]:
    minimum_hessian_eigenvalue = np.inf
    maximum_jensen_violation = -np.inf
    for _ in range(trials):
        dimension = 5
        n_blocks = 4
        prior = random_psd(rng, dimension, dimension) + np.eye(dimension)
        target = random_psd(rng, dimension, 3)
        blocks = [random_psd(rng, dimension, 2) for _ in range(n_blocks)]
        first = rng.uniform(0.0, 1.0, size=n_blocks)
        second = rng.uniform(0.0, 1.0, size=n_blocks)
        mix = float(rng.uniform())
        combined = mix * first + (1.0 - mix) * second
        combined_value, _, hessian = continuous_target_a(
            combined, blocks, target, prior
        )
        first_value = continuous_target_a(first, blocks, target, prior)[0]
        second_value = continuous_target_a(second, blocks, target, prior)[0]
        jensen_violation = combined_value - (
            mix * first_value + (1.0 - mix) * second_value
        )
        maximum_jensen_violation = max(maximum_jensen_violation, jensen_violation)
        minimum_hessian_eigenvalue = min(
            minimum_hessian_eigenvalue,
            float(np.linalg.eigvalsh(hessian)[0]),
        )
    return {
        "trials": trials,
        "minimum_hessian_eigenvalue": minimum_hessian_eigenvalue,
        "maximum_jensen_violation": maximum_jensen_violation,
    }


def powerset(names: tuple[str, ...]) -> list[frozenset[str]]:
    return [
        frozenset(combination)
        for size in range(len(names) + 1)
        for combination in itertools.combinations(names, size)
    ]


def d_opt_submodularity_check(rng: np.random.Generator) -> dict[str, float]:
    dimension = 4
    prior = random_psd(rng, dimension, dimension) + np.eye(dimension)
    rows = rng.normal(size=(5, dimension))
    blocks = {str(index): np.outer(row, row) for index, row in enumerate(rows)}
    names = tuple(sorted(blocks))
    subsets = powerset(names)

    def value(selected: frozenset[str]) -> float:
        gram = aggregate_gram(blocks, selected)
        return d_opt_information_gain(prior, gram)

    minimum_monotone_gain = np.inf
    minimum_diminishing_margin = np.inf
    comparisons = 0
    for smaller in subsets:
        for larger in subsets:
            if not smaller.issubset(larger):
                continue
            for candidate in set(names) - set(larger):
                gain_small = value(smaller | {candidate}) - value(smaller)
                gain_large = value(larger | {candidate}) - value(larger)
                minimum_monotone_gain = min(
                    minimum_monotone_gain, gain_small, gain_large
                )
                minimum_diminishing_margin = min(
                    minimum_diminishing_margin, gain_small - gain_large
                )
                comparisons += 1
    return {
        "comparisons": comparisons,
        "minimum_monotone_gain": minimum_monotone_gain,
        "minimum_diminishing_returns_margin": minimum_diminishing_margin,
    }


def a_opt_non_submodular_counterexample() -> dict[str, object]:
    features = np.array([
        [-1.379107303258097, 13.610895726498347],
        [-0.5438133406059672, 0.6519343443096093],
        [2.008858610711189, 11.588225405885595],
    ])
    blocks = [np.outer(row, row) for row in features]
    prior = np.eye(2)
    target = np.diag([10.0, 0.1])

    def risk(indices: tuple[int, ...]) -> float:
        gram = sum(
            (blocks[index] for index in indices), start=np.zeros((2, 2))
        )
        return target_a_objective(target, prior, gram)

    gain_empty = risk(()) - risk((0,))
    gain_after_two = risk((2,)) - risk((2, 0))
    return {
        "features": features.tolist(),
        "target_moment": target.tolist(),
        "element": 0,
        "larger_context": [2],
        "gain_from_empty": gain_empty,
        "gain_after_context": gain_after_two,
        "increasing_returns_gap": gain_after_two - gain_empty,
        "violates_submodularity": bool(gain_after_two > gain_empty),
    }


def sketch_coverage_checks(
    rng: np.random.Generator,
    trials: int,
) -> dict[str, float]:
    covered = 0
    maximum_violation = 0.0
    widths = []
    adaptive_matches = 0
    adaptive_certified_steps = 0
    adaptive_total_steps = 0
    for trial in range(trials):
        dimension = 8
        prior = random_psd(rng, dimension, dimension) + np.eye(dimension)
        target = random_psd(rng, dimension, 5)
        blocks = {
            str(index): gram_from_features(rng.normal(size=(10, dimension)))
            for index in range(3)
        }
        grams = list(blocks.values())
        rank = trial % (dimension + 1)
        sketches = [make_psd_sketch(gram, rank) for gram in grams]
        interval = sketch_target_a_interval(sketches, target, prior)
        exact = target_a_objective(target, prior, sum(grams, start=np.zeros_like(prior)))
        lower_violation = max(interval.lower - exact, 0.0)
        upper_violation = max(exact - interval.upper, 0.0)
        violation = max(lower_violation, upper_violation)
        maximum_violation = max(maximum_violation, violation)
        covered += int(violation <= INTERVAL_TOLERANCE)
        widths.append(interval.width)
        full = greedy_target_a(blocks, target, prior, k=2)
        adaptive = adaptive_sketch_target_a(
            blocks,
            target,
            prior,
            k=2,
            rank_schedule=[1, 2, 4],
            fallback_to_full=True,
        )
        adaptive_matches += int(adaptive.selected == full.selected)
        adaptive_certified_steps += sum(step.certified for step in adaptive.steps)
        adaptive_total_steps += len(adaptive.steps)
    return {
        "trials": trials,
        "covered": covered,
        "coverage": covered / trials,
        "maximum_absolute_violation": maximum_violation,
        "mean_interval_width": float(np.mean(widths)),
        "maximum_interval_width": float(np.max(widths)),
        "adaptive_full_greedy_matches": adaptive_matches,
        "adaptive_match_rate": adaptive_matches / trials,
        "adaptive_certificate_rate": (
            adaptive_certified_steps / adaptive_total_steps
        ),
    }


def label_blind_counterexample() -> dict[str, object]:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    labels_positive = features.copy()
    labels_negative = -features
    source_positive = ridge_statistics(features, labels_positive)
    source_negative = ridge_statistics(features, labels_negative)
    target_positive = ridge_statistics(features, labels_positive)
    target_negative = ridge_statistics(features, labels_negative)
    regularization = 0.1
    risks = {
        "positive_source_positive_target": ridge_squared_risk(
            source_positive, target_positive, regularization
        ),
        "negative_source_positive_target": ridge_squared_risk(
            source_negative, target_positive, regularization
        ),
        "positive_source_negative_target": ridge_squared_risk(
            source_positive, target_negative, regularization
        ),
        "negative_source_negative_target": ridge_squared_risk(
            source_negative, target_negative, regularization
        ),
    }
    return {
        "candidate_feature_grams_identical": bool(np.array_equal(
            source_positive.gram, source_negative.gram
        )),
        "risks": risks,
        "best_source_for_positive_target": min(
            ("positive", "negative"),
            key=lambda name: risks[f"{name}_source_positive_target"],
        ),
        "best_source_for_negative_target": min(
            ("positive", "negative"),
            key=lambda name: risks[f"{name}_source_negative_target"],
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "target_conditioned" / "e0_theory_validation",
    )
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--monte-carlo-draws", type=int, default=1_000_000)
    parser.add_argument("--formula-trials", type=int, default=100)
    parser.add_argument("--sketch-trials", type=int, default=100)
    parser.add_argument("--mini", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mini:
        args.monte_carlo_draws = min(args.monte_carlo_draws, 200_000)
        args.formula_trials = min(args.formula_trials, 20)
        args.sketch_trials = min(args.sketch_trials, 20)
    if min(args.monte_carlo_draws, args.formula_trials, args.sketch_trials) <= 0:
        raise ValueError("all trial counts must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    monte_carlo = monte_carlo_bayes_risk(rng, args.monte_carlo_draws)
    marginals = marginal_identity_checks(rng, args.formula_trials)
    convexity = convexity_checks(rng, args.formula_trials)
    d_opt = d_opt_submodularity_check(rng)
    a_opt_counterexample = a_opt_non_submodular_counterexample()
    sketches = sketch_coverage_checks(rng, args.sketch_trials)
    label_counterexample = label_blind_counterexample()

    checks = {
        "bayes_risk_monte_carlo": (
            monte_carlo["relative_error"] <= MONTE_CARLO_RELATIVE_TOLERANCE
        ),
        "sample_marginal_identity": (
            marginals["sample_max_relative_error"] <= FORMULA_RELATIVE_TOLERANCE
        ),
        "block_marginal_identity": (
            marginals["block_max_relative_error"] <= FORMULA_RELATIVE_TOLERANCE
        ),
        "a_opt_hessian_psd": (
            convexity["minimum_hessian_eigenvalue"] >= -CONVEXITY_TOLERANCE
        ),
        "a_opt_jensen_convexity": (
            convexity["maximum_jensen_violation"] <= CONVEXITY_TOLERANCE
        ),
        "d_opt_monotone": (
            d_opt["minimum_monotone_gain"] >= -SUBMODULARITY_TOLERANCE
        ),
        "d_opt_diminishing_returns": (
            d_opt["minimum_diminishing_returns_margin"]
            >= -SUBMODULARITY_TOLERANCE
        ),
        "a_opt_non_submodular_counterexample": a_opt_counterexample[
            "violates_submodularity"
        ],
        "sketch_interval_coverage": sketches["covered"] == sketches["trials"],
        "adaptive_sketch_matches_full_greedy": (
            sketches["adaptive_full_greedy_matches"] == sketches["trials"]
        ),
        "label_blind_non_identifiability": (
            label_counterexample["candidate_feature_grams_identical"]
            and label_counterexample["best_source_for_positive_target"] == "positive"
            and label_counterexample["best_source_for_negative_target"] == "negative"
        ),
    }
    passed = all(checks.values())
    report = {
        "experiment": "E0 target-conditioned theory validation",
        "status": "passed" if passed else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "thresholds": {
            "formula_relative_tolerance": FORMULA_RELATIVE_TOLERANCE,
            "monte_carlo_relative_tolerance": MONTE_CARLO_RELATIVE_TOLERANCE,
            "convexity_tolerance": CONVEXITY_TOLERANCE,
            "submodularity_tolerance": SUBMODULARITY_TOLERANCE,
            "interval_tolerance": INTERVAL_TOLERANCE,
        },
        "checks": checks,
        "metrics": {
            "monte_carlo": monte_carlo,
            "marginal_identities": marginals,
            "convexity": convexity,
            "d_opt": d_opt,
            "sketches": sketches,
        },
        "counterexamples": {
            "a_opt_non_submodular": a_opt_counterexample,
            "label_blind_non_identifiability": label_counterexample,
        },
        "provenance": {
            "git_revision": git_revision(),
            "script_sha256": sha256_file(Path(__file__)),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "mini": args.mini,
        },
    }
    report_path = args.out_dir / "e0_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    counterexample_path = args.out_dir / "counterexamples.json"
    counterexample_path.write_text(
        json.dumps(report["counterexamples"], indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "seed": args.seed,
        "monte_carlo_draws": args.monte_carlo_draws,
        "formula_trials": args.formula_trials,
        "sketch_trials": args.sketch_trials,
        "mini": args.mini,
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "experiment": report["experiment"],
        "status": report["status"],
        "created_utc": report["created_utc"],
        "config": config,
        "provenance": report["provenance"],
        "outputs": [
            "config.json",
            "counterexamples.json",
            "e0_validation.json",
            "manifest.json",
            "README.md",
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# E0 Target-Conditioned Theory Validation

Status: {report['status']}.

- Monte Carlo draws: {args.monte_carlo_draws}
- Formula trials: {args.formula_trials}
- Sketch trials: {args.sketch_trials}
- Git revision: `{report['provenance']['git_revision']}`

`e0_validation.json` is the authoritative machine-readable report. A failed
check returns a nonzero process exit code and blocks E1 in the server runner.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "status": report["status"],
        "checks": checks,
        "monte_carlo_relative_error": monte_carlo["relative_error"],
        "sample_marginal_max_relative_error": marginals[
            "sample_max_relative_error"
        ],
        "block_marginal_max_relative_error": marginals[
            "block_max_relative_error"
        ],
        "sketch_coverage": sketches["coverage"],
        "report": str(report_path),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
