#!/usr/bin/env python3
"""Validate and summarize the complete E2 exhaustive oracle diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
    write_json_atomic,
)
import run_target_conditioned_e2 as experiment  # noqa: E402
import run_target_conditioned_e2_oracle as oracle  # noqa: E402


COMBINATION_NUMERIC_FIELDS = {
    "budget": int,
    "selected_sample_count": int,
    "target_test_count": int,
    "ridge_regularization": float,
    "evaluation_seconds": float,
    "target_a_proxy_estimate": float,
    "target_a_proxy_first_half": float,
    "target_a_proxy_second_half": float,
    "target_a_proxy_half_relative_difference": float,
    "squared_loss": float,
    "brier_score": float,
    "nll": float,
    "accuracy": float,
    "macro_f1": float,
    "ece": float,
    "brier_rank": int,
    "target_a_proxy_rank": int,
    "in_brier_top_q": int,
    "normalized_brier_regret": float,
}
DIAGNOSTIC_NUMERIC_FIELDS = {
    "budget": int,
    "repeat": int,
    "selected_sample_count": int,
    "brier_score": float,
    "oracle_brier": float,
    "normalized_brier_regret": float,
    "brier_rank": int,
    "top_q": int,
    "in_brier_top_q": int,
    "target_a_proxy_rank": int,
}
TASK_NUMERIC_FIELDS = {
    "budget": int,
    "combination_count": int,
    "top_q": int,
    "oracle_brier": float,
    "oracle_tie_count": int,
    "worst_brier": float,
    "target_a_brier": float,
    "target_a_brier_rank": int,
    "target_a_in_top_q": int,
    "target_a_normalized_brier_regret": float,
    "proxy_best_brier": float,
    "proxy_best_brier_rank": int,
    "proxy_best_normalized_brier_regret": float,
    "parent_envelope_repeat": int,
    "parent_envelope_brier": float,
    "oracle_relative_headroom_from_parent_envelope": float,
    "spearman_target_a_proxy_vs_brier": float,
    "mean_probe_half_relative_difference": float,
    "max_probe_half_relative_difference": float,
    "budget_seconds": float,
}


def read_csv_rows(
    path: Path, numeric_fields: Mapping[str, type]
) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except OSError as error:
        raise E2ArtifactError(f"cannot read {path.name}: {error}") from error
    if not rows:
        raise E2ArtifactError(f"{path.name} is empty")
    missing = sorted(set(numeric_fields) - set(rows[0]))
    if missing:
        raise E2ArtifactError(f"{path.name} is missing fields: {missing}")
    output = []
    for row in rows:
        converted: dict[str, object] = dict(row)
        try:
            for field, converter in numeric_fields.items():
                converted[field] = converter(row[field])
        except ValueError as error:
            raise E2ArtifactError(f"{path.name} has an invalid number") from error
        for field, converter in numeric_fields.items():
            if converter is float and not math.isfinite(float(converted[field])):
                raise E2ArtifactError(f"{path.name} has a non-finite {field}")
        output.append(converted)
    return output


def _validate_identifier(value: Mapping[str, object], field: str) -> None:
    core = {key: child for key, child in value.items() if key != field}
    if value.get(field) != canonical_json_sha256(core):
        raise E2ArtifactError(f"{field} does not match its artifact")


def _assert_close(left: float, right: float, message: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        raise E2ArtifactError(message)


def validate_run(
    run_dir: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    manifest_path = run_dir / "manifest.json"
    combinations_path = run_dir / "combinations.csv"
    diagnostics_path = run_dir / "selection_diagnostics.csv"
    tasks_path = run_dir / "task_summary.csv"
    manifest = experiment.read_json(manifest_path)
    if manifest.get("schema_version") != oracle.RUN_SCHEMA:
        raise E2ArtifactError("oracle run has the wrong schema")
    if manifest.get("status") != "completed-posthoc-exploratory":
        raise E2ArtifactError("oracle run is not complete")
    _validate_identifier(manifest, "oracle_run_id")
    hashes = {
        "combinations_file_sha256": combinations_path,
        "selection_diagnostics_file_sha256": diagnostics_path,
        "task_summary_file_sha256": tasks_path,
    }
    for field, path in hashes.items():
        if manifest.get(field) != sha256_file(path):
            raise E2ArtifactError(f"oracle run {path.name} hash mismatch")
    access = manifest.get("evaluation_access")
    if not isinstance(access, dict):
        raise E2ArtifactError("oracle run has no access record")
    if access.get("algorithmic_target_labels") != "target_test-only-for-posthoc-oracle":
        raise E2ArtifactError("oracle run does not disclose target-test oracle access")
    if access.get("target_selection_labels_used") is not False:
        raise E2ArtifactError("oracle run used target-selection labels")
    if access.get("target_calibration_used") is not False:
        raise E2ArtifactError("oracle run used target-calibration labels")
    if access.get("changes_parent_confirmatory_gates") is not False:
        raise E2ArtifactError("oracle run claims to change the parent gates")

    combinations = read_csv_rows(combinations_path, COMBINATION_NUMERIC_FIELDS)
    diagnostics = read_csv_rows(diagnostics_path, DIAGNOSTIC_NUMERIC_FIELDS)
    tasks = read_csv_rows(tasks_path, TASK_NUMERIC_FIELDS)
    expected_counts = {
        "combination_row_count": len(combinations),
        "selection_diagnostic_row_count": len(diagnostics),
        "task_summary_row_count": len(tasks),
    }
    for field, count in expected_counts.items():
        if int(manifest.get(field, -1)) != count:
            raise E2ArtifactError(f"oracle run {field} mismatch")

    dataset = str(manifest.get("dataset"))
    encoder = str(manifest.get("encoder"))
    for table in (combinations, diagnostics, tasks):
        if any(row["dataset"] != dataset or row["encoder"] != encoder for row in table):
            raise E2ArtifactError("oracle CSV identity differs from manifest")

    combination_keys = [
        (row["target_domain"], row["budget"], row["combination"])
        for row in combinations
    ]
    if len(combination_keys) != len(set(combination_keys)):
        raise E2ArtifactError("oracle combination rows are duplicated")
    diagnostic_keys = [
        (
            row["target_domain"],
            row["budget"],
            row["method"],
            row["repeat"],
        )
        for row in diagnostics
    ]
    if len(diagnostic_keys) != len(set(diagnostic_keys)):
        raise E2ArtifactError("oracle diagnostic rows are duplicated")
    task_keys = [(row["target_domain"], row["budget"]) for row in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise E2ArtifactError("oracle task rows are duplicated")

    config = manifest.get("oracle_config")
    parent_config = manifest.get("parent_experiment_config")
    if not isinstance(config, dict) or not isinstance(parent_config, dict):
        raise E2ArtifactError("oracle run does not embed its configs")
    analysis = config.get("analysis")
    if not isinstance(analysis, dict):
        raise E2ArtifactError("oracle run config has no analysis")
    budgets = [int(value) for value in analysis["budgets"]]
    candidate_count = int(analysis["expected_candidate_count_per_task"])
    expected_methods = {
        str(method)
        for values in parent_config["method_groups"].values()
        for method in values
    }
    random_repeats = int(parent_config["selection"]["random_repeats"])
    domains = sorted({str(row["target_domain"]) for row in tasks})
    if len(tasks) != len(domains) * len(budgets):
        raise E2ArtifactError("oracle task grid is incomplete")

    combo_lookup = {
        (str(row["target_domain"]), int(row["budget"]), str(row["combination"])): row
        for row in combinations
    }
    task_lookup = {
        (str(row["target_domain"]), int(row["budget"])): row for row in tasks
    }
    for target_domain in domains:
        for budget in budgets:
            group = [
                row
                for row in combinations
                if row["target_domain"] == target_domain and row["budget"] == budget
            ]
            expected = math.comb(candidate_count, budget)
            if len(group) != expected:
                raise E2ArtifactError("oracle combination grid has the wrong size")
            for row in group:
                selected = oracle.canonical_combination(str(row["combination"]))
                if len(selected) != budget:
                    raise E2ArtifactError("oracle combination has the wrong budget")
                if int(row["in_brier_top_q"]) not in (0, 1):
                    raise E2ArtifactError("oracle top-Q flag is not binary")
            if sorted(int(row["brier_rank"]) for row in group) != list(
                range(1, expected + 1)
            ):
                raise E2ArtifactError("oracle Brier ranks are incomplete")
            if sorted(int(row["target_a_proxy_rank"]) for row in group) != list(
                range(1, expected + 1)
            ):
                raise E2ArtifactError("oracle proxy ranks are incomplete")
            task = task_lookup[(target_domain, budget)]
            if int(task["combination_count"]) != expected:
                raise E2ArtifactError("oracle task combination count mismatch")
            ordered = sorted(
                group,
                key=lambda row: (float(row["brier_score"]), str(row["combination"])),
            )
            _assert_close(
                float(task["oracle_brier"]),
                float(ordered[0]["brier_score"]),
                "oracle task minimum differs from combination table",
            )
            if task["oracle_combination"] != ordered[0]["combination"]:
                raise E2ArtifactError("oracle task combination differs from minimum")

    diagnostic_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in diagnostics:
        key = (str(row["target_domain"]), int(row["budget"]), str(row["combination"]))
        combination = combo_lookup.get(key)
        if combination is None:
            raise E2ArtifactError("oracle diagnostic references an unknown combination")
        _assert_close(
            float(row["brier_score"]),
            float(combination["brier_score"]),
            "oracle diagnostic Brier differs from combination table",
        )
        if int(row["brier_rank"]) != int(combination["brier_rank"]):
            raise E2ArtifactError("oracle diagnostic rank differs from combination table")
        diagnostic_counts[(str(row["target_domain"]), int(row["budget"]), str(row["method"]))] += 1
    for target_domain in domains:
        for budget in budgets:
            for method in expected_methods:
                expected = random_repeats if method == "random" else 1
                if diagnostic_counts[(target_domain, budget, method)] != expected:
                    raise E2ArtifactError("oracle parent-method coverage is incomplete")
    return manifest, combinations, diagnostics, tasks


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    oracle.write_csv(path, rows)


def aggregate_diagnostic_repeats(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["encoder"],
            row["target_domain"],
            row["budget"],
            row["method"],
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        dataset, encoder, target_domain, budget, method = key
        output.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "target_domain": target_domain,
                "budget": int(budget),
                "method": method,
                "repeat_count": len(values),
                "brier_score": float(np.mean([float(row["brier_score"]) for row in values])),
                "oracle_brier": float(np.mean([float(row["oracle_brier"]) for row in values])),
                "normalized_brier_regret": float(
                    np.mean([float(row["normalized_brier_regret"]) for row in values])
                ),
                "brier_rank": float(np.mean([float(row["brier_rank"]) for row in values])),
                "in_brier_top_q": float(
                    np.mean([float(row["in_brier_top_q"]) for row in values])
                ),
                "exact_oracle": float(
                    np.mean([int(row["brier_rank"]) == 1 for row in values])
                ),
            }
        )
    return output


def aggregate_encoders(
    rows: Sequence[Mapping[str, object]], encoders: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["target_domain"],
            row["budget"],
            row["method"],
        )
        grouped[key].append(row)
    output = []
    numeric = (
        "brier_score",
        "oracle_brier",
        "normalized_brier_regret",
        "brier_rank",
        "in_brier_top_q",
        "exact_oracle",
    )
    for key, values in sorted(grouped.items()):
        if sorted(str(value["encoder"]) for value in values) != sorted(encoders):
            raise E2ArtifactError("oracle method unit does not contain every encoder")
        dataset, target_domain, budget, method = key
        output.append(
            {
                "dataset": dataset,
                "target_domain": target_domain,
                "budget": int(budget),
                "method": method,
                **{
                    field: float(np.mean([float(row[field]) for row in values]))
                    for field in numeric
                },
            }
        )
    return output


def aggregate_tasks(
    rows: Sequence[Mapping[str, object]], encoders: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_domain"], row["budget"])].append(row)
    mean_fields = (
        "oracle_brier",
        "worst_brier",
        "target_a_brier",
        "target_a_brier_rank",
        "target_a_in_top_q",
        "target_a_normalized_brier_regret",
        "proxy_best_brier",
        "proxy_best_brier_rank",
        "proxy_best_normalized_brier_regret",
        "parent_envelope_brier",
        "oracle_relative_headroom_from_parent_envelope",
        "spearman_target_a_proxy_vs_brier",
        "mean_probe_half_relative_difference",
    )
    output = []
    for key, values in sorted(grouped.items()):
        if sorted(str(value["encoder"]) for value in values) != sorted(encoders):
            raise E2ArtifactError("oracle task unit does not contain every encoder")
        dataset, target_domain, budget = key
        output.append(
            {
                "dataset": dataset,
                "target_domain": target_domain,
                "budget": int(budget),
                **{
                    field: float(np.mean([float(row[field]) for row in values]))
                    for field in mean_fields
                },
                "max_probe_half_relative_difference": float(
                    np.max(
                        [
                            float(row["max_probe_half_relative_difference"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def method_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget"]), str(row["method"]))].append(row)
    output = []
    for (budget, method), values in sorted(grouped.items()):
        output.append(
            {
                "budget": budget,
                "method": method,
                "task_units": len(values),
                "brier_score_mean": float(
                    np.mean([float(row["brier_score"]) for row in values])
                ),
                "oracle_brier_mean": float(
                    np.mean([float(row["oracle_brier"]) for row in values])
                ),
                "normalized_brier_regret_mean": float(
                    np.mean([float(row["normalized_brier_regret"]) for row in values])
                ),
                "normalized_brier_regret_median": float(
                    np.median([float(row["normalized_brier_regret"]) for row in values])
                ),
                "combination_rank_mean": float(
                    np.mean([float(row["brier_rank"]) for row in values])
                ),
                "top_q_recall": float(
                    np.mean([float(row["in_brier_top_q"]) for row in values])
                ),
                "exact_oracle_recall": float(
                    np.mean([float(row["exact_oracle"]) for row in values])
                ),
            }
        )
    return output


def cross_encoder_correlations(
    combinations: Sequence[Mapping[str, object]], encoders: Sequence[str]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in combinations:
        grouped[
            (str(row["dataset"]), str(row["target_domain"]), int(row["budget"]))
        ].append(row)
    output = []
    for (dataset, target_domain, budget), rows in sorted(grouped.items()):
        by_encoder = {
            encoder: {
                str(row["combination"]): row
                for row in rows
                if row["encoder"] == encoder
            }
            for encoder in encoders
        }
        combinations_by_encoder = [set(values) for values in by_encoder.values()]
        if not combinations_by_encoder or any(
            values != combinations_by_encoder[0] for values in combinations_by_encoder[1:]
        ):
            raise E2ArtifactError("encoders expose different oracle combinations")
        names = sorted(combinations_by_encoder[0])
        if len(encoders) == 1:
            left = right = encoders[0]
        elif len(encoders) == 2:
            left, right = encoders
        else:
            raise E2ArtifactError(
                "cross-encoder diagnostics currently require one or two encoders"
            )
        output.append(
            {
                "dataset": dataset,
                "target_domain": target_domain,
                "budget": budget,
                "combination_count": len(names),
                "encoder_left": left,
                "encoder_right": right,
                "brier_spearman": oracle.spearman_correlation(
                    [float(by_encoder[left][name]["brier_score"]) for name in names],
                    [float(by_encoder[right][name]["brier_score"]) for name in names],
                ),
                "target_a_proxy_spearman": oracle.spearman_correlation(
                    [
                        float(by_encoder[left][name]["target_a_proxy_estimate"])
                        for name in names
                    ],
                    [
                        float(by_encoder[right][name]["target_a_proxy_estimate"])
                        for name in names
                    ],
                ),
            }
        )
    return output


def bootstrap_interval(
    values: Sequence[float], repeats: int, confidence: float, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def summarize(run_dirs: Sequence[Path], output_dir: Path) -> dict[str, object]:
    if (output_dir / "summary.json").exists():
        raise E2ArtifactError("refusing to overwrite an existing E2 oracle summary")
    manifests = []
    all_combinations = []
    all_diagnostics = []
    all_tasks = []
    for run_dir in run_dirs:
        manifest, combinations, diagnostics, tasks = validate_run(run_dir)
        manifests.append(manifest)
        all_combinations.extend(combinations)
        all_diagnostics.extend(diagnostics)
        all_tasks.extend(tasks)

    oracle_configs = [manifest["oracle_config"] for manifest in manifests]
    parent_configs = [manifest["parent_experiment_config"] for manifest in manifests]
    if any(config != oracle_configs[0] for config in oracle_configs[1:]):
        raise E2ArtifactError("oracle runs embed different oracle configs")
    if any(config != parent_configs[0] for config in parent_configs[1:]):
        raise E2ArtifactError("oracle runs embed different parent configs")
    config = oracle_configs[0]
    parent_config = parent_configs[0]
    expected_pairs = {
        (str(dataset), str(encoder))
        for dataset in config["datasets"]
        for encoder in config["encoders"]
    }
    actual_pairs = {
        (str(manifest["dataset"]), str(manifest["encoder"]))
        for manifest in manifests
    }
    if actual_pairs != expected_pairs or len(manifests) != len(expected_pairs):
        raise E2ArtifactError("oracle run matrix is incomplete")
    for field in (
        "oracle_config_file_sha256",
        "runner_revision",
        "parent_experiment_config_file_sha256",
        "parent_runner_revision",
    ):
        if len({str(manifest[field]) for manifest in manifests}) != 1:
            raise E2ArtifactError(f"oracle runs have mixed {field}")
    if len({str(manifest["parent_evaluation_id"]) for manifest in manifests}) != len(
        manifests
    ):
        raise E2ArtifactError("oracle runs reuse a parent evaluation")

    encoders = [str(value) for value in config["encoders"]]
    repeated = aggregate_diagnostic_repeats(all_diagnostics)
    random_repeats = int(parent_config["selection"]["random_repeats"])
    for row in repeated:
        expected = random_repeats if row["method"] == "random" else 1
        if int(row["repeat_count"]) != expected:
            raise E2ArtifactError("oracle method repeat count differs from parent config")
    method_units = aggregate_encoders(repeated, encoders)
    task_units = aggregate_tasks(all_tasks, encoders)
    method_rows = method_summary(method_units)
    correlations = cross_encoder_correlations(all_combinations, encoders)

    primary_budget = int(config["analysis"]["primary_budget"])
    target_blind = [str(value) for value in parent_config["method_groups"]["target_blind"]]
    primary_method_rows = [
        row for row in method_rows if int(row["budget"]) == primary_budget
    ]
    summary_by_method = {str(row["method"]): row for row in primary_method_rows}
    strongest_blind = min(
        target_blind,
        key=lambda method: (
            float(summary_by_method[method]["brier_score_mean"]), method
        ),
    )
    unit_methods = {
        (
            str(row["dataset"]),
            str(row["target_domain"]),
            int(row["budget"]),
            str(row["method"]),
        ): row
        for row in method_units
    }
    primary_tasks = [
        row for row in task_units if int(row["budget"]) == primary_budget
    ]
    correlation_by_unit = {
        (str(row["dataset"]), str(row["target_domain"]), int(row["budget"])): row
        for row in correlations
    }
    headroom_units = []
    for task in primary_tasks:
        unit = (str(task["dataset"]), str(task["target_domain"]), primary_budget)
        baseline = unit_methods[(*unit, strongest_blind)]
        target_a = unit_methods[(*unit, "target_a")]
        baseline_brier = float(baseline["brier_score"])
        target_a_brier = float(target_a["brier_score"])
        oracle_brier = float(task["oracle_brier"])
        oracle_gain = baseline_brier - oracle_brier
        headroom_units.append(
            {
                "dataset": unit[0],
                "target_domain": unit[1],
                "budget": primary_budget,
                "baseline": strongest_blind,
                "baseline_brier": baseline_brier,
                "target_a_brier": target_a_brier,
                "oracle_brier": oracle_brier,
                "oracle_relative_headroom_from_baseline": oracle_gain
                / max(abs(baseline_brier), 1e-15),
                "target_a_normalized_regret": (target_a_brier - oracle_brier)
                / max(abs(oracle_brier), 1e-15),
                "target_a_fraction_of_oracle_gain_captured": (
                    (baseline_brier - target_a_brier) / oracle_gain
                    if oracle_gain > 1e-15
                    else 0.0
                ),
                "target_a_combination_rank": float(target_a["brier_rank"]),
                "target_a_in_top_q": float(target_a["in_brier_top_q"]),
                "parent_envelope_brier": float(task["parent_envelope_brier"]),
                "oracle_relative_headroom_from_parent_envelope": float(
                    task["oracle_relative_headroom_from_parent_envelope"]
                ),
                "mean_within_encoder_proxy_brier_spearman": float(
                    task["spearman_target_a_proxy_vs_brier"]
                ),
                "cross_encoder_brier_spearman": float(
                    correlation_by_unit[unit]["brier_spearman"]
                ),
            }
        )

    parent_analysis = parent_config["primary_analysis"]
    repeats = int(parent_analysis["bootstrap_repeats"])
    confidence = float(parent_analysis["confidence_level"])
    base_seed = int(parent_config["selection"]["random_seed"])
    oracle_headroom = [
        float(row["oracle_relative_headroom_from_baseline"])
        for row in headroom_units
    ]
    target_regret = [float(row["target_a_normalized_regret"]) for row in headroom_units]
    envelope_headroom = [
        float(row["oracle_relative_headroom_from_parent_envelope"])
        for row in headroom_units
    ]
    primary: dict[str, object] = {
        "budget": primary_budget,
        "metric": "brier_score",
        "task_units": len(headroom_units),
        "strongest_parent_target_blind": strongest_blind,
        "baseline_brier_mean": float(
            np.mean([float(row["baseline_brier"]) for row in headroom_units])
        ),
        "target_a_brier_mean": float(
            np.mean([float(row["target_a_brier"]) for row in headroom_units])
        ),
        "oracle_brier_mean": float(
            np.mean([float(row["oracle_brier"]) for row in headroom_units])
        ),
        "oracle_relative_headroom_from_baseline_mean": float(np.mean(oracle_headroom)),
        "oracle_relative_headroom_from_baseline_confidence_interval": (
            bootstrap_interval(
                oracle_headroom,
                repeats,
                confidence,
                experiment.stable_seed(base_seed, "oracle-headroom"),
            )
        ),
        "oracle_headroom_exceeds_parent_2pct_threshold": bool(
            float(np.mean(oracle_headroom)) > 0.02
        ),
        "target_a_normalized_regret_mean": float(np.mean(target_regret)),
        "target_a_normalized_regret_confidence_interval": bootstrap_interval(
            target_regret,
            repeats,
            confidence,
            experiment.stable_seed(base_seed, "target-a-oracle-regret"),
        ),
        "target_a_top_q_recall": float(
            np.mean([float(row["target_a_in_top_q"]) for row in headroom_units])
        ),
        "target_a_combination_rank_mean": float(
            np.mean([float(row["target_a_combination_rank"]) for row in headroom_units])
        ),
        "target_a_fraction_of_oracle_gain_captured_mean": float(
            np.mean(
                [
                    float(row["target_a_fraction_of_oracle_gain_captured"])
                    for row in headroom_units
                ]
            )
        ),
        "oracle_relative_headroom_from_parent_envelope_mean": float(
            np.mean(envelope_headroom)
        ),
        "within_encoder_proxy_brier_spearman_mean": float(
            np.mean(
                [
                    float(row["mean_within_encoder_proxy_brier_spearman"])
                    for row in headroom_units
                ]
            )
        ),
        "cross_encoder_brier_spearman_mean": float(
            np.mean(
                [float(row["cross_encoder_brier_spearman"]) for row in headroom_units]
            )
        ),
        "probe_half_relative_difference_mean": float(
            np.mean(
                [
                    float(row["mean_probe_half_relative_difference"])
                    for row in primary_tasks
                ]
            )
        ),
        "probe_half_relative_difference_max": float(
            np.max(
                [
                    float(row["max_probe_half_relative_difference"])
                    for row in primary_tasks
                ]
            )
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    method_path = output_dir / "method_oracle_summary.csv"
    task_path = output_dir / "task_units.csv"
    correlation_path = output_dir / "correlations.csv"
    headroom_path = output_dir / "primary_headroom_units.csv"
    write_csv(method_path, method_rows)
    write_csv(task_path, task_units)
    write_csv(correlation_path, correlations)
    write_csv(headroom_path, headroom_units)
    summary: dict[str, object] = {
        "schema_version": "target-conditioned-e2-oracle-summary-v1",
        "status": "completed-posthoc-exploratory",
        "runner_revision": next(iter({str(value["runner_revision"]) for value in manifests})),
        "oracle_config_file_sha256": next(
            iter({str(value["oracle_config_file_sha256"]) for value in manifests})
        ),
        "parent_runner_revision": next(
            iter({str(value["parent_runner_revision"]) for value in manifests})
        ),
        "run_count": len(manifests),
        "combination_row_count": len(all_combinations),
        "selection_diagnostic_row_count": len(all_diagnostics),
        "task_unit_count": len({(row["dataset"], row["target_domain"]) for row in task_units}),
        "primary": primary,
        "method_oracle_summary_file_sha256": sha256_file(method_path),
        "task_units_file_sha256": sha256_file(task_path),
        "correlations_file_sha256": sha256_file(correlation_path),
        "primary_headroom_units_file_sha256": sha256_file(headroom_path),
        "inputs": sorted(
            [
                {
                    "dataset": str(value["dataset"]),
                    "encoder": str(value["encoder"]),
                    "oracle_run_id": str(value["oracle_run_id"]),
                    "parent_evaluation_id": str(value["parent_evaluation_id"]),
                }
                for value in manifests
            ],
            key=lambda value: (value["dataset"], value["encoder"]),
        ),
        "claim_boundary": config["claim_boundary"],
    }
    summary["summary_id"] = canonical_json_sha256(summary)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(args.run_dir, args.out_dir)
    print(json.dumps(result["primary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
