#!/usr/bin/env python3
"""Exhaustively diagnose post-hoc oracle headroom for a completed E2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.target_conditioned_e2 import (  # noqa: E402
    E2ArtifactError,
    canonical_json_sha256,
    sha256_file,
    validate_feature_cache,
    write_json_atomic,
)
import run_target_conditioned_e2 as experiment  # noqa: E402
import summarize_target_conditioned_e2 as parent_summary  # noqa: E402


CONFIG_SCHEMA = "target-conditioned-e2-oracle-config-v1"
RUN_SCHEMA = "target-conditioned-e2-oracle-run-v1"
METRICS = ("brier_score", "squared_loss", "nll", "accuracy", "macro_f1", "ece")
LOWER_IS_BETTER = ("brier_score", "squared_loss", "nll", "ece")
RANK_TIE_RELATIVE_TOLERANCE = 1e-12


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise E2ArtifactError(f"{name} must be an object")
    return value


def _required_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise E2ArtifactError(f"{name} must be a nonempty list")
    return value


def load_oracle_config(
    path: Path, parent_config: Mapping[str, object]
) -> dict[str, object]:
    config = experiment.read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise E2ArtifactError(f"oracle config schema must be {CONFIG_SCHEMA}")
    if config.get("status") != "frozen-before-exploratory-oracle-evaluation":
        raise E2ArtifactError("oracle config is not frozen")
    if config.get("parent_experiment_config_file_sha256") in (None, ""):
        raise E2ArtifactError("oracle config does not bind the parent config")
    if config.get("parent_evaluation_runner_revision") in (None, ""):
        raise E2ArtifactError("oracle config does not bind the parent runner")

    for field in ("datasets", "encoders"):
        values = [str(value) for value in _required_list(config.get(field), field)]
        expected = [
            str(value) for value in _required_list(parent_config.get(field), field)
        ]
        if values != expected or len(values) != len(set(values)):
            raise E2ArtifactError(f"oracle {field} differ from the parent experiment")

    analysis = _required_mapping(config.get("analysis"), "analysis")
    parent_selection = _required_mapping(parent_config.get("selection"), "selection")
    budgets = [
        int(value) for value in _required_list(analysis.get("budgets"), "budgets")
    ]
    parent_budgets = [int(value) for value in parent_selection["budgets"]]
    if budgets != parent_budgets:
        raise E2ArtifactError("oracle budgets differ from the parent experiment")
    if int(analysis.get("primary_budget", -1)) != int(
        parent_selection["primary_budget"]
    ):
        raise E2ArtifactError("oracle primary budget differs from the parent experiment")
    if analysis.get("primary_metric") != "brier_score":
        raise E2ArtifactError("oracle primary metric must be brier_score")
    if int(analysis.get("top_q", 0)) < 1:
        raise E2ArtifactError("oracle top_q must be positive")
    if analysis.get("normalized_regret_denominator") != "absolute-oracle-risk":
        raise E2ArtifactError("unsupported oracle regret normalization")
    if analysis.get("combination_tie_break") != "metric-then-lexicographic":
        raise E2ArtifactError("unsupported oracle tie rule")
    if analysis.get("target_a_proxy") != "common-rademacher-hutchinson":
        raise E2ArtifactError("unsupported target-A proxy estimator")
    probes = int(analysis.get("hutchinson_probes", 0))
    if probes < 2 or probes % 2:
        raise E2ArtifactError("hutchinson_probes must be a positive even number")
    if not isinstance(analysis.get("hutchinson_seed"), int):
        raise E2ArtifactError("hutchinson_seed must be an integer")
    if analysis.get("probe_halves_reported") is not True:
        raise E2ArtifactError("probe-half diagnostics must be enabled")
    if int(analysis.get("expected_candidate_count_per_task", 0)) < max(budgets):
        raise E2ArtifactError("expected candidate count is smaller than a budget")

    access = _required_mapping(config.get("access"), "access")
    if access != {
        "target_test_used_to_define_oracle": True,
        "parent_selection_remains_immutable": True,
        "changes_parent_confirmatory_gates": False,
    }:
        raise E2ArtifactError("oracle target-test access boundary is incomplete")
    if not str(config.get("claim_boundary", "")).strip():
        raise E2ArtifactError("oracle config has no claim boundary")
    return config


def canonical_combination(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        selected = tuple(part for part in value.split("|") if part)
    else:
        selected = tuple(str(part) for part in value)
    canonical = tuple(sorted(selected))
    if not canonical or len(canonical) != len(set(canonical)):
        raise E2ArtifactError("combination is empty or contains duplicates")
    return canonical


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise E2ArtifactError(f"cannot write empty {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise E2ArtifactError("rank input must be a finite nonempty vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise E2ArtifactError("Spearman inputs must have the same nontrivial length")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator == 0.0:
        raise E2ArtifactError("Spearman correlation is undefined for a constant rank")
    return float(np.dot(left_centered, right_centered) / denominator)


def fit_ridge_scores_and_target_a_proxy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    class_count: int,
    regularization: float,
    target_probe_rhs: np.ndarray,
    target_sample_count: int,
    noise_variance: float,
) -> tuple[np.ndarray, str, float, float, float, float]:
    """Solve ridge and estimate target A-opt in one factorization."""

    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    probes = np.asarray(target_probe_rhs, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise E2ArtifactError("ridge train/test feature dimensions differ")
    if labels.shape != (len(train),):
        raise E2ArtifactError("ridge labels do not match training rows")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= class_count:
        raise E2ArtifactError("ridge labels are empty or outside the class vocabulary")
    if probes.ndim != 2 or probes.shape[0] != train.shape[1]:
        raise E2ArtifactError("target probe right-hand sides have the wrong shape")
    if probes.shape[1] < 2 or probes.shape[1] % 2:
        raise E2ArtifactError("target probe count must be positive and even")
    if target_sample_count < 1 or regularization <= 0.0 or noise_variance <= 0.0:
        raise E2ArtifactError("ridge/proxy scalar parameters must be positive")

    one_hot = np.eye(class_count, dtype=np.float64)[labels]
    if train.shape[0] < train.shape[1]:
        kernel = train @ train.T
        kernel.flat[:: kernel.shape[0] + 1] += regularization
        projected_probes = train @ probes
        solved = np.linalg.solve(
            kernel, np.concatenate((one_hot, projected_probes), axis=1)
        )
        coefficients = solved[:, :class_count]
        proxy_solution = (
            probes - train.T @ solved[:, class_count:]
        ) / regularization
        scores = test @ train.T @ coefficients
        solver = "dual"
    else:
        gram = train.T @ train
        gram.flat[:: gram.shape[0] + 1] += regularization
        solved = np.linalg.solve(
            gram,
            np.concatenate((train.T @ one_hot, probes), axis=1),
        )
        scores = test @ solved[:, :class_count]
        proxy_solution = solved[:, class_count:]
        solver = "primal"

    quadratic = np.sum(probes * proxy_solution, axis=0)
    if not np.isfinite(quadratic).all() or np.any(quadratic <= 0.0):
        raise E2ArtifactError("target-A Hutchinson quadratic is nonpositive")
    scale = noise_variance / target_sample_count
    midpoint = len(quadratic) // 2
    first = float(np.mean(quadratic[:midpoint]) * scale)
    second = float(np.mean(quadratic[midpoint:]) * scale)
    estimate = float(np.mean(quadratic) * scale)
    half_relative_difference = abs(first - second) / max(abs(estimate), 1e-15)
    return scores, solver, estimate, first, second, half_relative_difference


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10)


def _rank_combination_rows(
    rows: list[dict[str, object]], top_q: int
) -> dict[str, object]:
    if not rows:
        raise E2ArtifactError("cannot rank an empty combination table")
    q = min(top_q, len(rows))
    ordered_brier = sorted(
        rows, key=lambda row: (float(row["brier_score"]), str(row["combination"]))
    )
    ordered_proxy = sorted(
        rows,
        key=lambda row: (
            float(row["target_a_proxy_estimate"]),
            str(row["combination"]),
        ),
    )
    brier_ranks = {
        str(row["combination"]): rank
        for rank, row in enumerate(ordered_brier, start=1)
    }
    proxy_ranks = {
        str(row["combination"]): rank
        for rank, row in enumerate(ordered_proxy, start=1)
    }
    oracle = float(ordered_brier[0]["brier_score"])
    for row in rows:
        name = str(row["combination"])
        risk = float(row["brier_score"])
        row["brier_rank"] = brier_ranks[name]
        row["target_a_proxy_rank"] = proxy_ranks[name]
        row["in_brier_top_q"] = int(brier_ranks[name] <= q)
        row["normalized_brier_regret"] = (risk - oracle) / max(abs(oracle), 1e-15)

    tolerance = RANK_TIE_RELATIVE_TOLERANCE * max(abs(oracle), 1.0)
    tie_count = sum(
        abs(float(row["brier_score"]) - oracle) <= tolerance for row in rows
    )
    return {
        "top_q": q,
        "oracle": ordered_brier[0],
        "worst": ordered_brier[-1],
        "proxy_best": ordered_proxy[0],
        "oracle_tie_count": tie_count,
        "spearman_proxy_brier": spearman_correlation(
            [float(row["target_a_proxy_estimate"]) for row in rows],
            [float(row["brier_score"]) for row in rows],
        ),
    }


def _selection_diagnostics(
    parent_rows: Sequence[Mapping[str, object]],
    target_domain: str,
    budget: int,
    combination_rows: Mapping[tuple[str, ...], Mapping[str, object]],
    oracle_brier: float,
    top_q: int,
) -> list[dict[str, object]]:
    diagnostics = []
    for parent in parent_rows:
        if parent["target_domain"] != target_domain or int(parent["budget"]) != budget:
            continue
        selected = canonical_combination(str(parent["selected"]))
        if selected not in combination_rows:
            raise E2ArtifactError("parent selection is absent from exhaustive combinations")
        exhaustive = combination_rows[selected]
        for metric in METRICS:
            if not _is_close(float(parent[metric]), float(exhaustive[metric])):
                raise E2ArtifactError(
                    f"exhaustive {metric} does not reproduce the parent evaluation"
                )
        if int(parent["selected_sample_count"]) != int(
            exhaustive["selected_sample_count"]
        ):
            raise E2ArtifactError("exhaustive selection sample count differs from parent")
        risk = float(exhaustive["brier_score"])
        diagnostics.append(
            {
                "dataset": parent["dataset"],
                "encoder": parent["encoder"],
                "target_domain": target_domain,
                "budget": budget,
                "method": parent["method"],
                "repeat": int(parent["repeat"]),
                "combination": str(exhaustive["combination"]),
                "selected_sample_count": int(exhaustive["selected_sample_count"]),
                "brier_score": risk,
                "oracle_brier": oracle_brier,
                "normalized_brier_regret": (risk - oracle_brier)
                / max(abs(oracle_brier), 1e-15),
                "brier_rank": int(exhaustive["brier_rank"]),
                "top_q": top_q,
                "in_brier_top_q": int(exhaustive["in_brier_top_q"]),
                "target_a_proxy_rank": int(exhaustive["target_a_proxy_rank"]),
            }
        )
    if not diagnostics:
        raise E2ArtifactError("parent evaluation has no rows for an oracle task")
    return diagnostics


def run_oracle(
    manifest_dir: Path,
    feature_path: Path,
    metadata_path: Path,
    parent_config_path: Path,
    oracle_config_path: Path,
    parent_run_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_names = (
        "combinations.csv",
        "selection_diagnostics.csv",
        "task_summary.csv",
        "manifest.json",
    )
    if any((output_dir / name).exists() for name in output_names):
        raise E2ArtifactError("refusing to overwrite an existing E2 oracle run")
    started_utc = experiment.utc_now()
    started = time.perf_counter()

    parent_config = experiment.load_experiment_config(parent_config_path)
    oracle_config = load_oracle_config(oracle_config_path, parent_config)
    parent_manifest, parent_selection, parent_rows = parent_summary.validate_run(
        parent_run_dir
    )
    parent_config_hash = sha256_file(parent_config_path)
    if (
        parent_manifest.get("experiment_config_file_sha256") != parent_config_hash
        or oracle_config.get("parent_experiment_config_file_sha256")
        != parent_config_hash
    ):
        raise E2ArtifactError("oracle run does not bind the parent experiment config")
    if (
        parent_manifest.get("runner_revision")
        != oracle_config.get("parent_evaluation_runner_revision")
    ):
        raise E2ArtifactError("oracle run does not bind the parent runner revision")

    cache = experiment.load_selection_cache(manifest_dir, feature_path, metadata_path)
    if parent_manifest.get("dataset") != cache.dataset:
        raise E2ArtifactError("parent evaluation dataset differs from cache")
    if parent_manifest.get("encoder") != cache.encoder:
        raise E2ArtifactError("parent evaluation encoder differs from cache")
    experiment.validate_selection_artifact(
        parent_selection,
        parent_config,
        parent_config_hash,
        cache,
    )
    validate_feature_cache(feature_path, metadata_path, manifest_dir, cache.encoder)

    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            features = np.asarray(payload["H"])
            labels = np.asarray(payload["y"])
            sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    except (OSError, ValueError) as error:
        raise E2ArtifactError(f"cannot load oracle cache: {error}") from error
    if not np.array_equal(sample_ids, cache.sample_ids):
        raise E2ArtifactError("oracle cache sample IDs differ from selection cache")

    transform = _required_mapping(parent_config["feature_transform"], "feature_transform")
    epsilon = float(transform["epsilon"])
    parent_selection_config = _required_mapping(
        parent_config["selection"], "selection"
    )
    evaluation = _required_mapping(parent_config["evaluation"], "evaluation")
    regularization = float(evaluation["ridge_regularization"])
    noise_variance = float(parent_selection_config["noise_variance"])
    prior_precision = float(parent_selection_config["prior_precision"])
    if not _is_close(regularization, prior_precision * noise_variance):
        raise E2ArtifactError(
            "oracle proxy reuse requires ridge lambda = prior precision * noise variance"
        )
    ece_bins = int(evaluation["ece_bins"])
    manifest = experiment.read_json(manifest_dir / "manifest.json")
    class_count = len(_required_mapping(manifest.get("class_to_id"), "class_to_id"))
    id_to_index = {value: index for index, value in enumerate(sample_ids)}

    analysis = _required_mapping(oracle_config["analysis"], "analysis")
    budgets = [int(value) for value in analysis["budgets"]]
    top_q = int(analysis["top_q"])
    probe_count = int(analysis["hutchinson_probes"])
    probe_seed = int(analysis["hutchinson_seed"])
    expected_candidates = int(analysis["expected_candidate_count_per_task"])
    parent_task_by_domain = {
        str(task["target_domain"]): task for task in parent_selection["tasks"]
    }

    all_combinations: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, object]] = []
    task_summaries: list[dict[str, object]] = []
    target_access = []
    probe_records = []

    for target_domain in cache.domains:
        task_started = time.perf_counter()
        view = experiment.build_task_view(cache, target_domain, epsilon)
        names = tuple(sorted(view.candidate_features))
        if len(names) != expected_candidates:
            raise E2ArtifactError(
                f"target {target_domain} has {len(names)} candidates, "
                f"expected {expected_candidates}"
            )
        parent_inventory = parent_task_by_domain[target_domain]["candidate_inventory"]
        if [value["candidate_id"] for value in parent_inventory] != list(names):
            raise E2ArtifactError("oracle candidate order differs from parent selection")

        assignment_rows = [
            row
            for row in cache.assignments
            if str(row["target_domain"]) == target_domain
        ]
        candidate_ids: dict[str, list[str]] = {}
        target_test_ids = []
        for row in assignment_rows:
            if row["role"] == "source_candidate":
                candidate_ids.setdefault(str(row["candidate_id"]), []).append(
                    str(row["sample_id"])
                )
            elif row["role"] == "target_test":
                target_test_ids.append(str(row["sample_id"]))
        if tuple(sorted(candidate_ids)) != names or not target_test_ids:
            raise E2ArtifactError("oracle assignment views are incomplete")
        test_indices = np.asarray(
            [id_to_index[value] for value in target_test_ids], dtype=np.int64
        )
        test_features = experiment.normalize_rows(features[test_indices], epsilon)
        test_labels = labels[test_indices]

        rng = np.random.default_rng(
            experiment.stable_seed(probe_seed, cache.dataset, target_domain)
        )
        signs = rng.integers(
            0, 2, size=(len(view.target_features), probe_count), dtype=np.int8
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        target_probe_rhs = np.asarray(view.target_features, dtype=np.float64).T @ signs
        probe_records.append(
            {
                "target_domain": target_domain,
                "target_selection_count": len(view.target_features),
                "probe_count": probe_count,
                "probe_sha256": hashlib.sha256(signs.tobytes(order="C")).hexdigest(),
            }
        )
        target_access.append(
            {
                "target_domain": target_domain,
                "target_test_count": len(target_test_ids),
                "target_test_ids_sha256": experiment.hash_ids(target_test_ids),
            }
        )

        for budget in budgets:
            budget_started = time.perf_counter()
            group_rows: list[dict[str, object]] = []
            combinations = list(itertools.combinations(names, budget))
            for index, selected in enumerate(combinations, start=1):
                train_ids = sorted(
                    sample_id
                    for candidate in selected
                    for sample_id in candidate_ids[candidate]
                )
                train_indices = np.asarray(
                    [id_to_index[value] for value in train_ids], dtype=np.int64
                )
                train_features = experiment.normalize_rows(
                    features[train_indices], epsilon
                )
                fit_started = time.perf_counter()
                (
                    scores,
                    solver,
                    proxy,
                    proxy_first,
                    proxy_second,
                    proxy_half_difference,
                ) = fit_ridge_scores_and_target_a_proxy(
                    train_features,
                    labels[train_indices],
                    test_features,
                    class_count,
                    regularization,
                    target_probe_rhs,
                    len(view.target_features),
                    noise_variance,
                )
                metrics = experiment.prediction_metrics(
                    scores, test_labels, class_count, ece_bins
                )
                group_rows.append(
                    {
                        "dataset": cache.dataset,
                        "encoder": cache.encoder,
                        "target_domain": target_domain,
                        "budget": budget,
                        "combination": "|".join(selected),
                        "selected_sample_count": len(train_ids),
                        "selected_sample_ids_sha256": experiment.hash_ids(train_ids),
                        "target_test_count": len(target_test_ids),
                        "ridge_regularization": regularization,
                        "ridge_solver": solver,
                        "evaluation_seconds": time.perf_counter() - fit_started,
                        "target_a_proxy_estimate": proxy,
                        "target_a_proxy_first_half": proxy_first,
                        "target_a_proxy_second_half": proxy_second,
                        "target_a_proxy_half_relative_difference": (
                            proxy_half_difference
                        ),
                        **metrics,
                    }
                )
                if index % 100 == 0 or index == len(combinations):
                    print(
                        f"oracle dataset={cache.dataset} encoder={cache.encoder} "
                        f"target={target_domain} K={budget} "
                        f"progress={index}/{len(combinations)}",
                        flush=True,
                    )

            rank_summary = _rank_combination_rows(group_rows, top_q)
            row_lookup = {
                canonical_combination(str(row["combination"])): row
                for row in group_rows
            }
            oracle = rank_summary["oracle"]
            proxy_best = rank_summary["proxy_best"]
            diagnostics = _selection_diagnostics(
                parent_rows,
                target_domain,
                budget,
                row_lookup,
                float(oracle["brier_score"]),
                int(rank_summary["top_q"]),
            )
            target_a_rows = [
                row
                for row in diagnostics
                if row["method"] == "target_a" and int(row["repeat"]) == -1
            ]
            if len(target_a_rows) != 1:
                raise E2ArtifactError("parent task has no unique Target A-opt selection")
            target_a_row = target_a_rows[0]
            envelope = min(
                diagnostics,
                key=lambda row: (
                    float(row["brier_score"]),
                    str(row["method"]),
                    int(row["repeat"]),
                ),
            )
            all_combinations.extend(group_rows)
            all_diagnostics.extend(diagnostics)
            task_summaries.append(
                {
                    "dataset": cache.dataset,
                    "encoder": cache.encoder,
                    "target_domain": target_domain,
                    "budget": budget,
                    "combination_count": len(group_rows),
                    "top_q": int(rank_summary["top_q"]),
                    "oracle_combination": oracle["combination"],
                    "oracle_brier": float(oracle["brier_score"]),
                    "oracle_tie_count": int(rank_summary["oracle_tie_count"]),
                    "worst_brier": float(rank_summary["worst"]["brier_score"]),
                    "target_a_combination": target_a_row["combination"],
                    "target_a_brier": float(target_a_row["brier_score"]),
                    "target_a_brier_rank": int(target_a_row["brier_rank"]),
                    "target_a_in_top_q": int(target_a_row["in_brier_top_q"]),
                    "target_a_normalized_brier_regret": float(
                        target_a_row["normalized_brier_regret"]
                    ),
                    "proxy_best_combination": proxy_best["combination"],
                    "proxy_best_brier": float(proxy_best["brier_score"]),
                    "proxy_best_brier_rank": int(proxy_best["brier_rank"]),
                    "proxy_best_normalized_brier_regret": float(
                        proxy_best["normalized_brier_regret"]
                    ),
                    "parent_envelope_method": envelope["method"],
                    "parent_envelope_repeat": int(envelope["repeat"]),
                    "parent_envelope_combination": envelope["combination"],
                    "parent_envelope_brier": float(envelope["brier_score"]),
                    "oracle_relative_headroom_from_parent_envelope": (
                        float(envelope["brier_score"])
                        - float(oracle["brier_score"])
                    )
                    / max(abs(float(envelope["brier_score"])), 1e-15),
                    "spearman_target_a_proxy_vs_brier": float(
                        rank_summary["spearman_proxy_brier"]
                    ),
                    "mean_probe_half_relative_difference": float(
                        np.mean(
                            [
                                float(row["target_a_proxy_half_relative_difference"])
                                for row in group_rows
                            ]
                        )
                    ),
                    "max_probe_half_relative_difference": float(
                        np.max(
                            [
                                float(row["target_a_proxy_half_relative_difference"])
                                for row in group_rows
                            ]
                        )
                    ),
                    "budget_seconds": time.perf_counter() - budget_started,
                }
            )
        print(
            f"oracle completed dataset={cache.dataset} encoder={cache.encoder} "
            f"target={target_domain} seconds={time.perf_counter() - task_started:.2f}",
            flush=True,
        )

    expected_combinations = len(cache.domains) * sum(
        math.comb(expected_candidates, budget) for budget in budgets
    )
    if len(all_combinations) != expected_combinations:
        raise E2ArtifactError("oracle combination grid is incomplete")
    if len(all_diagnostics) != len(parent_rows):
        raise E2ArtifactError("oracle diagnostics do not cover every parent row")

    output_dir.mkdir(parents=True, exist_ok=True)
    combinations_path = output_dir / "combinations.csv"
    diagnostics_path = output_dir / "selection_diagnostics.csv"
    task_summary_path = output_dir / "task_summary.csv"
    write_csv(combinations_path, all_combinations)
    write_csv(diagnostics_path, all_diagnostics)
    write_csv(task_summary_path, task_summaries)
    result: dict[str, object] = {
        "schema_version": RUN_SCHEMA,
        "status": "completed-posthoc-exploratory",
        "dataset": cache.dataset,
        "encoder": cache.encoder,
        "manifest_id": cache.manifest_id,
        "feature_file_sha256": cache.feature_file_sha256,
        "feature_revision": cache.feature_revision,
        "runner_revision": experiment.git_revision(),
        "oracle_config_file_sha256": sha256_file(oracle_config_path),
        "oracle_config": oracle_config,
        "parent_experiment_config_file_sha256": parent_config_hash,
        "parent_experiment_config": parent_config,
        "parent_runner_revision": parent_manifest["runner_revision"],
        "parent_evaluation_id": parent_manifest["evaluation_id"],
        "parent_selection_id": parent_selection["selection_id"],
        "parent_evaluation_manifest_sha256": sha256_file(
            parent_run_dir / "manifest.json"
        ),
        "parent_raw_file_sha256": sha256_file(parent_run_dir / "raw.csv"),
        "parent_selection_file_sha256": sha256_file(
            parent_run_dir / "selection.json"
        ),
        "combination_row_count": len(all_combinations),
        "selection_diagnostic_row_count": len(all_diagnostics),
        "task_summary_row_count": len(task_summaries),
        "combinations_file_sha256": sha256_file(combinations_path),
        "selection_diagnostics_file_sha256": sha256_file(diagnostics_path),
        "task_summary_file_sha256": sha256_file(task_summary_path),
        "oracle_definition": {
            "metric": "brier_score",
            "scope": "all-candidate-combinations-within-each-task-and-budget",
            "uses_target_test_labels": True,
            "tie_break": "metric-then-lexicographic",
            "normalized_regret_denominator": "absolute-oracle-risk",
        },
        "target_a_proxy": {
            "estimator": "common-rademacher-hutchinson",
            "probe_count": probe_count,
            "probe_seed": probe_seed,
            "probe_records": probe_records,
        },
        "evaluation_access": {
            "storage_payload_loaded_full": True,
            "algorithmic_source_labels": "all-enumerated-source-combinations",
            "algorithmic_target_labels": "target_test-only-for-posthoc-oracle",
            "target_selection_labels_used": False,
            "target_calibration_used": False,
            "target_test_access": target_access,
            "changes_parent_confirmatory_gates": False,
        },
        "claim_boundary": oracle_config["claim_boundary"],
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": experiment.utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "device": "cpu",
        },
    }
    result["oracle_run_id"] = canonical_json_sha256(result)
    write_json_atomic(output_dir / "manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--oracle-config", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_oracle(
        args.manifest_dir,
        args.features,
        args.metadata,
        args.parent_config,
        args.oracle_config,
        args.parent_run_dir,
        args.out_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "oracle_run_id": result["oracle_run_id"],
                "combination_rows": result["combination_row_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
