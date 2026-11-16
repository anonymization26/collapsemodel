#!/usr/bin/env python3
"""Validate and summarize the complete E2 dataset-by-encoder gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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


NUMERIC_FIELDS = {
    "budget": int,
    "repeat": int,
    "selected_sample_count": int,
    "target_test_count": int,
    "ridge_regularization": float,
    "evaluation_seconds": float,
    "evaluation_cache_hit": int,
    "squared_loss": float,
    "brier_score": float,
    "nll": float,
    "accuracy": float,
    "macro_f1": float,
    "ece": float,
}
METRICS = ("brier_score", "squared_loss", "nll", "accuracy", "macro_f1", "ece")


def read_rows(path: Path) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except OSError as error:
        raise E2ArtifactError(f"cannot read {path.name}: {error}") from error
    if not rows:
        raise E2ArtifactError(f"{path.name} is empty")
    missing = sorted(set(NUMERIC_FIELDS) - set(rows[0]))
    if missing:
        raise E2ArtifactError(f"raw table is missing fields: {missing}")
    parsed = []
    for row in rows:
        converted: dict[str, object] = dict(row)
        try:
            for field, converter in NUMERIC_FIELDS.items():
                converted[field] = converter(row[field])
        except ValueError as error:
            raise E2ArtifactError(
                "raw table contains an invalid numeric field"
            ) from error
        if any(
            not math.isfinite(float(converted[field]))
            for field in METRICS + ("evaluation_seconds",)
        ):
            raise E2ArtifactError("raw table contains non-finite metrics")
        if not 0.0 <= float(converted["accuracy"]) <= 1.0:
            raise E2ArtifactError("accuracy is outside [0, 1]")
        if not 0.0 <= float(converted["macro_f1"]) <= 1.0:
            raise E2ArtifactError("macro-F1 is outside [0, 1]")
        if not 0.0 <= float(converted["ece"]) <= 1.0:
            raise E2ArtifactError("ECE is outside [0, 1]")
        parsed.append(converted)
    return parsed


def _validate_identifier(value: Mapping[str, object], field: str) -> None:
    core = {key: child for key, child in value.items() if key != field}
    if value.get(field) != canonical_json_sha256(core):
        raise E2ArtifactError(f"{field} does not match its artifact")


def validate_run(
    run_dir: Path,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    manifest_path = run_dir / "manifest.json"
    raw_path = run_dir / "raw.csv"
    selection_path = run_dir / "selection.json"
    manifest = experiment.read_json(manifest_path)
    selection = experiment.read_json(selection_path)
    if manifest.get("schema_version") != experiment.EVALUATION_SCHEMA:
        raise E2ArtifactError("run has the wrong evaluation schema")
    if selection.get("schema_version") != experiment.SELECTION_SCHEMA:
        raise E2ArtifactError("run has the wrong selection schema")
    _validate_identifier(manifest, "evaluation_id")
    _validate_identifier(selection, "selection_id")
    if manifest.get("raw_file_sha256") != sha256_file(raw_path):
        raise E2ArtifactError("run raw.csv hash mismatch")
    if manifest.get("selection_file_sha256") != sha256_file(selection_path):
        raise E2ArtifactError("run selection.json hash mismatch")
    if manifest.get("selection_id") != selection.get("selection_id"):
        raise E2ArtifactError("run manifest and selection IDs differ")
    if manifest.get("selection_frozen_before_target_test") is not True:
        raise E2ArtifactError("run did not freeze selection before evaluation")
    access = manifest.get("evaluation_access")
    if not isinstance(access, dict):
        raise E2ArtifactError("run has no evaluation access record")
    if access.get("target_calibration_used_by_metrics") is not False:
        raise E2ArtifactError("run used target-calibration labels")
    rows = read_rows(raw_path)
    if int(manifest.get("row_count", -1)) != len(rows):
        raise E2ArtifactError("run row count mismatch")
    for row in rows:
        if row["dataset"] != manifest.get("dataset"):
            raise E2ArtifactError("raw row dataset differs from manifest")
        if row["encoder"] != manifest.get("encoder"):
            raise E2ArtifactError("raw row encoder differs from manifest")
    keys = [
        (
            row["dataset"],
            row["encoder"],
            row["target_domain"],
            row["budget"],
            row["method"],
            row["repeat"],
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise E2ArtifactError("run contains duplicate result keys")
    return manifest, selection, rows


def aggregate_repeats(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
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
                "budget": budget,
                "method": method,
                "repeat_count": len(values),
                **{
                    metric: float(np.mean([float(row[metric]) for row in values]))
                    for metric in METRICS
                },
            }
        )
    return output


def aggregate_encoders(
    rows: Sequence[Mapping[str, object]],
    encoders: Sequence[str],
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
    for key, values in sorted(grouped.items()):
        if sorted(str(value["encoder"]) for value in values) != sorted(encoders):
            raise E2ArtifactError("a task-method cell does not contain every encoder")
        dataset, target_domain, budget, method = key
        output.append(
            {
                "dataset": dataset,
                "target_domain": target_domain,
                "budget": budget,
                "method": method,
                **{
                    metric: float(np.mean([float(row[metric]) for row in values]))
                    for metric in METRICS
                },
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
                **{
                    f"{metric}_mean": float(
                        np.mean([float(row[metric]) for row in values])
                    )
                    for metric in METRICS
                },
            }
        )
    return output


def _method_values(
    rows: Sequence[Mapping[str, object]],
    budget: int,
    method: str,
    metric: str,
) -> dict[tuple[str, str], float]:
    return {
        (str(row["dataset"]), str(row["target_domain"])): float(row[metric])
        for row in rows
        if int(row["budget"]) == budget and row["method"] == method
    }


def strongest_method(
    rows: Sequence[Mapping[str, object]],
    budget: int,
    methods: Iterable[str],
    metric: str,
) -> str:
    means = {}
    for method in methods:
        values = list(_method_values(rows, budget, method, metric).values())
        if not values:
            raise E2ArtifactError(f"method {method} has no primary-budget values")
        means[method] = float(np.mean(values))
    return min(sorted(means), key=means.__getitem__)


def percentile_interval(values: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    ]


def gate_comparison(
    rows: Sequence[Mapping[str, object]],
    budget: int,
    baseline: str,
    gate_name: str,
    analysis: Mapping[str, object],
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metric = "brier_score"
    target_values = _method_values(rows, budget, "target_a", metric)
    baseline_values = _method_values(rows, budget, baseline, metric)
    if set(target_values) != set(baseline_values):
        raise E2ArtifactError("paired gate methods have different task units")
    units = sorted(target_values)
    target = np.asarray([target_values[unit] for unit in units], dtype=np.float64)
    control = np.asarray([baseline_values[unit] for unit in units], dtype=np.float64)
    delta = target - control
    relative_improvement = (control - target) / np.maximum(np.abs(control), 1e-15)

    repeats = int(analysis["bootstrap_repeats"])
    confidence = float(analysis["confidence_level"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(units), size=(repeats, len(units)))
    bootstrap_delta = delta[indices].mean(axis=1)
    bootstrap_relative = relative_improvement[indices].mean(axis=1)
    tolerance = float(analysis["noninferiority_relative_tolerance"])
    noninferior = target <= control * (1.0 + tolerance)
    delta_interval = percentile_interval(bootstrap_delta, confidence)
    relative_interval = percentile_interval(bootstrap_relative, confidence)
    mean_relative = float(relative_improvement.mean())
    noninferior_fraction = float(noninferior.mean())
    passed = (
        mean_relative > float(analysis["minimum_mean_relative_improvement"])
        and delta_interval[1] < 0.0
        and noninferior_fraction >= float(analysis["minimum_noninferior_unit_fraction"])
    )
    paired = [
        {
            "gate": gate_name,
            "dataset": unit[0],
            "target_domain": unit[1],
            "budget": budget,
            "baseline": baseline,
            "target_a_brier": float(target[index]),
            "baseline_brier": float(control[index]),
            "delta": float(delta[index]),
            "relative_improvement": float(relative_improvement[index]),
            "noninferior": int(noninferior[index]),
        }
        for index, unit in enumerate(units)
    ]
    return (
        {
            "gate": gate_name,
            "passed": bool(passed),
            "budget": budget,
            "metric": metric,
            "baseline": baseline,
            "task_units": len(units),
            "target_a_mean": float(target.mean()),
            "baseline_mean": float(control.mean()),
            "mean_delta": float(delta.mean()),
            "mean_delta_confidence_interval": delta_interval,
            "mean_relative_improvement": mean_relative,
            "mean_relative_improvement_confidence_interval": relative_interval,
            "noninferior_fraction": noninferior_fraction,
            "thresholds": {
                "minimum_mean_relative_improvement": float(
                    analysis["minimum_mean_relative_improvement"]
                ),
                "delta_confidence_interval_upper_below_zero": True,
                "minimum_noninferior_unit_fraction": float(
                    analysis["minimum_noninferior_unit_fraction"]
                ),
                "noninferiority_relative_tolerance": tolerance,
            },
        },
        paired,
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise E2ArtifactError("cannot write an empty summary CSV")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(run_dirs: Sequence[Path], output_dir: Path) -> dict[str, object]:
    if (output_dir / "summary.json").exists():
        raise E2ArtifactError("refusing to overwrite an existing E2 summary")
    manifests = []
    selections = []
    all_rows = []
    for run_dir in run_dirs:
        manifest, selection, rows = validate_run(run_dir)
        manifests.append(manifest)
        selections.append(selection)
        all_rows.extend(rows)
    config_hashes = {str(value["experiment_config_file_sha256"]) for value in manifests}
    if len(config_hashes) != 1:
        raise E2ArtifactError("E2 runs use different experiment configs")
    runner_revisions = {str(value["runner_revision"]) for value in manifests}
    if len(runner_revisions) != 1:
        raise E2ArtifactError("E2 runs use different runner revisions")
    configs = [selection["experiment_config"] for selection in selections]
    if any(value != configs[0] for value in configs[1:]):
        raise E2ArtifactError("E2 selections embed different experiment configs")
    config = configs[0]
    expected_pairs = {
        (str(dataset), str(encoder))
        for dataset in config["datasets"]
        for encoder in config["encoders"]
    }
    actual_pairs = {
        (str(value["dataset"]), str(value["encoder"])) for value in manifests
    }
    if actual_pairs != expected_pairs or len(manifests) != len(expected_pairs):
        raise E2ArtifactError(
            f"E2 run matrix is incomplete: missing={sorted(expected_pairs - actual_pairs)}, "
            f"extra={sorted(actual_pairs - expected_pairs)}"
        )

    repeat_rows = aggregate_repeats(all_rows)
    groups = config["method_groups"]
    methods = {str(method) for values in groups.values() for method in values}
    random_repeats = int(config["selection"]["random_repeats"])
    for row in repeat_rows:
        expected = random_repeats if row["method"] == "random" else 1
        if int(row["repeat_count"]) != expected:
            raise E2ArtifactError("method repeat coverage differs from config")
    if {str(row["method"]) for row in repeat_rows} != methods:
        raise E2ArtifactError("summary method coverage differs from config")

    domains_by_dataset: dict[str, tuple[str, ...]] = {}
    for selection in selections:
        dataset = str(selection["input"]["dataset"])
        domains = tuple(str(task["target_domain"]) for task in selection["tasks"])
        previous = domains_by_dataset.setdefault(dataset, domains)
        if domains != previous:
            raise E2ArtifactError("encoders expose different target-domain tasks")
    budgets = [int(value) for value in config["selection"]["budgets"]]
    expected_cells = {
        (dataset, encoder, target_domain, budget, method)
        for dataset, domains in domains_by_dataset.items()
        for encoder in config["encoders"]
        for target_domain in domains
        for budget in budgets
        for method in methods
    }
    actual_cells = {
        (
            str(row["dataset"]),
            str(row["encoder"]),
            str(row["target_domain"]),
            int(row["budget"]),
            str(row["method"]),
        )
        for row in repeat_rows
    }
    if actual_cells != expected_cells:
        raise E2ArtifactError(
            f"E2 result grid is incomplete: missing={len(expected_cells - actual_cells)}, "
            f"extra={len(actual_cells - expected_cells)}"
        )

    task_rows = aggregate_encoders(
        repeat_rows, [str(value) for value in config["encoders"]]
    )
    primary_budget = int(config["selection"]["primary_budget"])
    target_blind = [str(value) for value in groups["target_blind"]]
    same_information = [
        str(value) for value in groups["target_conditioned"] if str(value) != "target_a"
    ]
    blind_baseline = strongest_method(
        task_rows, primary_budget, target_blind, "brier_score"
    )
    same_baseline = strongest_method(
        task_rows, primary_budget, same_information, "brier_score"
    )
    analysis = config["primary_analysis"]
    base_seed = int(config["selection"]["random_seed"])
    h2, h2_pairs = gate_comparison(
        task_rows,
        primary_budget,
        blind_baseline,
        "H2-target-blind",
        analysis,
        experiment.stable_seed(base_seed, "H2"),
    )
    h2b, h2b_pairs = gate_comparison(
        task_rows,
        primary_budget,
        same_baseline,
        "H2b-same-information",
        analysis,
        experiment.stable_seed(base_seed, "H2b"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = method_summary(task_rows)
    write_csv(output_dir / "method_summary.csv", method_rows)
    write_csv(output_dir / "paired_gates.csv", h2_pairs + h2b_pairs)
    summary: dict[str, object] = {
        "schema_version": "target-conditioned-e2-summary-v1",
        "status": "completed",
        "runner_revision": next(iter(runner_revisions)),
        "experiment_config_file_sha256": next(iter(config_hashes)),
        "run_count": len(manifests),
        "raw_row_count": len(all_rows),
        "task_unit_count": len(
            {(row["dataset"], row["target_domain"]) for row in task_rows}
        ),
        "gates": {"H2": h2, "H2b": h2b},
        "method_summary_file_sha256": sha256_file(output_dir / "method_summary.csv"),
        "paired_gates_file_sha256": sha256_file(output_dir / "paired_gates.csv"),
        "inputs": sorted(
            (
                {
                    "dataset": str(value["dataset"]),
                    "encoder": str(value["encoder"]),
                    "manifest_id": str(value["manifest_id"]),
                    "evaluation_id": str(value["evaluation_id"]),
                }
                for value in manifests
            ),
            key=lambda value: (value["dataset"], value["encoder"]),
        ),
        "claim_boundary": (
            "H2/H2b are evaluated only for the pre-registered PACS and "
            "Office-Home frozen-representation tasks."
        ),
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
    print(json.dumps(result["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
