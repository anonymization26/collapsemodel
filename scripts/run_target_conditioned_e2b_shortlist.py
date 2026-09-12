#!/usr/bin/env python3
"""Run E2b U0 screening, U2 validation, and frozen-test shortlist audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2b import (  # noqa: E402
    SCREEN_SCHEMA,
    TEST_AUDIT_SCHEMA,
    VALIDATION_SCHEMA,
    E2BArtifactError,
    all_combinations,
    canonical_json_sha256,
    combination_name,
    effective_rank_from_scatter,
    hash_ids,
    jl_project,
    load_config,
    parse_combination,
    read_json,
    read_manifest_samples,
    sha256_file,
    stable_seed,
    validate_candidates,
    validate_feature_cache,
    validate_feature_cache_unlabeled,
    validate_manifest,
    write_csv_atomic,
    write_json_atomic,
)


SCREEN_FIELDS = (
    "encoder",
    "target_domain",
    "method",
    "repeat",
    "rank",
    "combination",
    "score",
    "direction",
)
METRIC_FIELDS = (
    "encoder",
    "target_domain",
    "combination",
    "selected_sample_count",
    "brier_score",
    "squared_loss",
    "nll",
    "accuracy",
    "macro_f1",
    "ece",
    "evaluation_seconds",
)
SELECTION_FIELDS = (
    "encoder",
    "target_domain",
    "method",
    "repeat",
    "shortlist_size",
    "shortlist_sha256",
    "selected_combination",
    "validation_brier_score",
    "validation_rank_within_shortlist",
)
AUDIT_FIELDS = (
    "encoder",
    "target_domain",
    "method",
    "repeat",
    "shortlist_size",
    "combination_count",
    "shortlist_reduction",
    "top_q",
    "true_top_q_recall",
    "selected_combination",
    "selected_test_brier_score",
    "test_oracle_brier_score",
    "selected_normalized_regret",
    "selected_test_rank",
    "best_shortlist_test_brier_score",
    "best_shortlist_normalized_regret",
    "best_shortlist_test_rank",
    "screen_spearman_vs_test_brier",
    "passes_reduction_gate",
    "passes_recall_gate",
    "passes_regret_gate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    injected = os.environ.get("SOURCE_GIT_REVISION", "")
    if len(injected) == 40:
        return injected
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        path = ROOT / "SOURCE_REVISION"
        return path.read_text(encoding="ascii").strip() if path.is_file() else "unknown"


def _read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise E2BArtifactError(f"{path.name} columns differ from the frozen schema")
        return [dict(row) for row in reader]


def _load_features(
    feature_path: Path,
    metadata_path: Path,
    manifest_dir: Path,
    config_path: Path,
    encoder: str,
    *,
    load_labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, object]]:
    validator = validate_feature_cache if load_labels else validate_feature_cache_unlabeled
    report = validator(
        feature_path, metadata_path, manifest_dir, config_path, encoder
    )
    with np.load(feature_path, allow_pickle=False) as payload:
        features = np.asarray(payload["H"])
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
        labels = np.asarray(payload["y"]) if load_labels else None
    return features, labels, sample_ids, report


def _candidate_maps(candidate_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    artifact = read_json(candidate_path)
    ids = {}
    domains = {}
    for value in artifact["candidates"]:
        name = str(value["candidate_id"])
        ids[name] = [str(item) for item in value["sample_ids"]]
        domains[name] = str(value["domain"])
    return ids, domains


def _projected_views(
    features: np.ndarray,
    sample_ids: np.ndarray,
    samples: list[dict[str, object]],
    candidate_ids: Mapping[str, Sequence[str]],
    config: Mapping[str, object],
    encoder: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    projection = config["features"]["projection"]
    projected = jl_project(
        features,
        int(projection["output_dimension"]),
        stable_seed(projection["seed"], encoder),
    )
    id_to_index = {value: index for index, value in enumerate(sample_ids)}
    block_indices = {
        name: np.asarray([id_to_index[value] for value in ids], dtype=np.int64)
        for name, ids in candidate_ids.items()
    }
    role_indices: dict[tuple[str, str], np.ndarray] = {}
    domains = [str(value) for value in config["dataset"]["domains"]]
    for domain in domains:
        for role in ("target_selection", "target_validation", "target_test"):
            role_indices[(domain, role)] = np.asarray(
                [
                    index
                    for index, row in enumerate(samples)
                    if row["domain"] == domain and row["role"] == role
                ],
                dtype=np.int64,
            )
    return projected, block_indices, role_indices


def _nearest_psd_correlation(kernel: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
    projected = (vectors * np.maximum(values, 1e-10)) @ vectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), 1e-12))
    projected /= np.outer(scale, scale)
    np.fill_diagonal(projected, 1.0)
    return projected


def _score_combinations(
    block_features: Mapping[str, np.ndarray],
    block_domains: Mapping[str, str],
    target_features: np.ndarray,
    config: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    screen = config["screen"]
    budget = int(config["combinations"]["budget_blocks"])
    fixed_count = int(config["combinations"]["fixed_training_sample_count"])
    names = sorted(block_features)
    combinations = all_combinations(names, budget)
    blocks = {
        name: np.asarray(value, dtype=np.float64).T
        @ np.asarray(value, dtype=np.float64)
        for name, value in block_features.items()
    }
    target = np.asarray(target_features, dtype=np.float64)
    target_moment = target.T @ target / len(target)
    rank = int(screen["subspace_rank"])
    subspaces = {}
    for name in names:
        _, vectors = np.linalg.eigh((blocks[name] + blocks[name].T) / 2.0)
        subspaces[name] = vectors[:, -rank:]
    similarity = np.eye(len(names), dtype=np.float64)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = float(
                np.linalg.norm(
                    subspaces[names[left]].T @ subspaces[names[right]],
                    ord="fro",
                )
                ** 2
                / rank
            )
            similarity[left, right] = similarity[right, left] = value
    similarity = _nearest_psd_correlation(similarity)
    name_to_index = {name: index for index, name in enumerate(names)}
    prior = float(screen["prior_precision"])
    noise = float(screen["noise_variance"])
    scores = {str(method): {} for method in screen["methods"] if method != "random"}
    for combination in combinations:
        label = combination_name(combination)
        scatter = sum((blocks[name] for name in combination), start=np.zeros_like(blocks[names[0]]))
        scatter = (scatter + scatter.T) / 2.0
        eigenvalues, vectors = np.linalg.eigh(scatter)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        target_diagonal = np.sum(vectors * (target_moment @ vectors), axis=0)
        denominator = prior + eigenvalues / noise
        scores["target_a"][label] = float(np.sum(target_diagonal / denominator))
        source_moment = scatter / fixed_count
        difference = source_moment - target_moment
        scores["second_moment_mmd"][label] = float(np.sum(difference * difference))
        scores["target_energy"][label] = float(np.sum(target_moment * source_moment))
        scores["bayesian_d"][label] = float(np.sum(np.log(denominator)))
        scores["merged_effective_rank"][label] = effective_rank_from_scatter(
            scatter, fixed_count
        )
        indices = [name_to_index[name] for name in combination]
        kernel = similarity[np.ix_(indices, indices)] + 1e-8 * np.eye(budget)
        sign, logdet = np.linalg.slogdet(kernel)
        scores["dpp_subspace"][label] = float(logdet if sign > 0 else -math.inf)
        domain_counts = Counter(block_domains[name] for name in combination)
        scores["domain_balance"][label] = float(
            len(domain_counts) - 1e-3 * sum(value * value for value in domain_counts.values())
        )
    return scores


def run_screen(
    manifest_dir: Path,
    config_path: Path,
    candidate_path: Path,
    anchor_feature_path: Path,
    anchor_metadata_path: Path,
    feature_path: Path,
    metadata_path: Path,
    encoder: str,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2BArtifactError("refusing to overwrite a frozen screen run")
    started = time.perf_counter()
    started_utc = utc_now()
    config = load_config(config_path)
    if encoder not in config["features"]["evaluators"]:
        raise E2BArtifactError("screen encoder is not registered")
    manifest_report = validate_manifest(manifest_dir, config_path)
    candidate_report = validate_candidates(
        candidate_path,
        manifest_dir,
        config_path,
        anchor_feature_path,
        anchor_metadata_path,
    )
    features, labels, sample_ids, feature_report = _load_features(
        feature_path,
        metadata_path,
        manifest_dir,
        config_path,
        encoder,
        load_labels=False,
    )
    if labels is not None:
        raise AssertionError("screen loaded labels")
    samples = read_manifest_samples(manifest_dir)
    candidate_ids, candidate_domains = _candidate_maps(candidate_path)
    projected, block_indices, role_indices = _projected_views(
        features, sample_ids, samples, candidate_ids, config, encoder
    )
    rows = []
    tasks = []
    screen_config = config["screen"]
    expected_count = int(config["combinations"]["expected_combinations_per_target"])
    for target_domain in config["dataset"]["domains"]:
        task_started = time.perf_counter()
        names = sorted(
            name for name, domain in candidate_domains.items() if domain != target_domain
        )
        combinations = all_combinations(names, int(config["combinations"]["budget_blocks"]))
        if len(combinations) != expected_count:
            raise E2BArtifactError("screen combination count mismatch")
        block_features = {name: projected[block_indices[name]] for name in names}
        target_indices = role_indices[(str(target_domain), "target_selection")]
        target_features = projected[target_indices]
        method_scores = _score_combinations(
            block_features,
            candidate_domains,
            target_features,
            config,
        )
        for method, values in method_scores.items():
            direction = str(screen_config["methods"][method])
            ordered = sorted(
                values,
                key=lambda name: (
                    values[name] if direction == "ascending" else -values[name],
                    name,
                ),
            )
            for rank, name in enumerate(ordered, 1):
                rows.append(
                    {
                        "encoder": encoder,
                        "target_domain": target_domain,
                        "method": method,
                        "repeat": -1,
                        "rank": rank,
                        "combination": name,
                        "score": values[name],
                        "direction": direction,
                    }
                )
        labels_all = [combination_name(value) for value in combinations]
        for repeat in range(int(screen_config["random_repeats"])):
            rng = np.random.default_rng(
                stable_seed(
                    screen_config["random_seed"],
                    encoder,
                    target_domain,
                    repeat,
                )
            )
            ordered = [labels_all[index] for index in rng.permutation(len(labels_all))]
            for rank, name in enumerate(ordered, 1):
                rows.append(
                    {
                        "encoder": encoder,
                        "target_domain": target_domain,
                        "method": "random",
                        "repeat": repeat,
                        "rank": rank,
                        "combination": name,
                        "score": -rank,
                        "direction": "random",
                    }
                )
        tasks.append(
            {
                "target_domain": target_domain,
                "target_selection_count": len(target_indices),
                "target_selection_ids_sha256": hash_ids(sample_ids[target_indices]),
                "candidate_count": len(names),
                "combination_count": len(combinations),
                "elapsed_seconds": time.perf_counter() - task_started,
            }
        )
        print(
            f"screen encoder={encoder} target={target_domain} combinations={len(combinations)}",
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "rankings.csv"
    write_csv_atomic(rankings_path, SCREEN_FIELDS, rows)
    core: dict[str, object] = {
        "schema_version": SCREEN_SCHEMA,
        "created_utc": utc_now(),
        "manifest_id": manifest_report["manifest_id"],
        "candidate_set_id": candidate_report["candidate_set_id"],
        "config_file_sha256": sha256_file(config_path),
        "encoder": encoder,
        "feature_file_sha256": feature_report["feature_file_sha256"],
        "rankings_file_sha256": sha256_file(rankings_path),
        "row_count": len(rows),
        "runner_revision": git_revision(),
        "access": {
            "source_candidate_features": True,
            "target_selection_features": True,
            "source_labels": False,
            "target_validation_features_or_labels": False,
            "target_test_features_or_labels": False,
        },
        "tasks": tasks,
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }
    artifact = {**core, "screen_id": canonical_json_sha256(core)}
    write_json_atomic(output_dir / "manifest.json", artifact)
    return artifact


def _validate_screen(
    screen_dir: Path,
    config_path: Path,
    candidate_set_id: str,
    feature_file_sha256: str,
    encoder: str,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    artifact = read_json(screen_dir / "manifest.json")
    if artifact.get("schema_version") != SCREEN_SCHEMA:
        raise E2BArtifactError(f"screen schema must be {SCREEN_SCHEMA}")
    core = {key: value for key, value in artifact.items() if key != "screen_id"}
    if artifact.get("screen_id") != canonical_json_sha256(core):
        raise E2BArtifactError("screen identifier mismatch")
    if artifact.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("screen config mismatch")
    if artifact.get("candidate_set_id") != candidate_set_id:
        raise E2BArtifactError("screen candidate set mismatch")
    if artifact.get("feature_file_sha256") != feature_file_sha256:
        raise E2BArtifactError("screen feature cache mismatch")
    if artifact.get("encoder") != encoder:
        raise E2BArtifactError("screen encoder mismatch")
    access = artifact.get("access")
    if not isinstance(access, dict) or any(
        access.get(key) is not False
        for key in (
            "source_labels",
            "target_validation_features_or_labels",
            "target_test_features_or_labels",
        )
    ):
        raise E2BArtifactError("screen access boundary is invalid")
    rankings_path = screen_dir / "rankings.csv"
    if artifact.get("rankings_file_sha256") != sha256_file(rankings_path):
        raise E2BArtifactError("screen rankings hash mismatch")
    rows = _read_csv(rankings_path, SCREEN_FIELDS)
    if int(artifact.get("row_count", -1)) != len(rows):
        raise E2BArtifactError("screen row count mismatch")
    return artifact, rows


def _prediction_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    ece_bins: int,
) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    one_hot = np.eye(class_count, dtype=np.float64)[truth]
    squared_loss = float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    nll = float(
        -np.mean(
            np.log(
                np.maximum(
                    probabilities[np.arange(len(truth)), truth],
                    1e-15,
                )
            )
        )
    )
    f1 = []
    for label in range(class_count):
        true_positive = int(np.sum((predicted == label) & (truth == label)))
        false_positive = int(np.sum((predicted == label) & (truth != label)))
        false_negative = int(np.sum((predicted != label) & (truth == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(np.float64)
    ece = 0.0
    for index in range(ece_bins):
        lower = index / ece_bins
        upper = (index + 1) / ece_bins
        mask = (confidence >= lower) & (
            confidence <= upper if index + 1 == ece_bins else confidence < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "brier_score": brier,
        "squared_loss": squared_loss,
        "nll": nll,
        "accuracy": float(np.mean(predicted == truth)),
        "macro_f1": float(np.mean(f1)),
        "ece": ece,
    }


def _evaluate_combinations(
    combinations: Sequence[str],
    block_features: Mapping[str, np.ndarray],
    block_labels: Mapping[str, np.ndarray],
    target_features: np.ndarray,
    target_labels: np.ndarray,
    class_count: int,
    regularization: float,
    ece_bins: int,
    *,
    progress_label: str,
) -> list[dict[str, object]]:
    dimension = next(iter(block_features.values())).shape[1]
    identity = np.eye(dimension, dtype=np.float64)
    one_hot = np.eye(class_count, dtype=np.float64)
    grams = {}
    right_sides = {}
    for name, value in block_features.items():
        matrix = np.asarray(value, dtype=np.float64)
        labels = np.asarray(block_labels[name], dtype=np.int64)
        grams[name] = matrix.T @ matrix
        right_sides[name] = matrix.T @ one_hot[labels]
    target = np.asarray(target_features, dtype=np.float64)
    rows = []
    for index, label in enumerate(combinations, 1):
        names = parse_combination(label)
        started = time.perf_counter()
        gram = regularization * identity + sum(
            (grams[name] for name in names), start=np.zeros_like(identity)
        )
        rhs = sum(
            (right_sides[name] for name in names),
            start=np.zeros((dimension, class_count), dtype=np.float64),
        )
        weights = np.linalg.solve((gram + gram.T) / 2.0, rhs)
        metrics = _prediction_metrics(
            target @ weights,
            target_labels,
            class_count,
            ece_bins,
        )
        rows.append(
            {
                "combination": label,
                "selected_sample_count": sum(len(block_features[name]) for name in names),
                **metrics,
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
        if index % 50 == 0 or index == len(combinations):
            print(f"{progress_label} evaluated={index}/{len(combinations)}", flush=True)
    return rows


def _screen_groups(
    rows: Sequence[Mapping[str, str]], target_domain: str
) -> dict[tuple[str, int], list[str]]:
    grouped: dict[tuple[str, int], list[tuple[int, str]]] = {}
    for row in rows:
        if row["target_domain"] != target_domain:
            continue
        key = (str(row["method"]), int(row["repeat"]))
        grouped.setdefault(key, []).append((int(row["rank"]), str(row["combination"])))
    result = {}
    for key, values in grouped.items():
        ordered = sorted(values)
        if [rank for rank, _ in ordered] != list(range(1, len(ordered) + 1)):
            raise E2BArtifactError("screen ranks are incomplete")
        result[key] = [name for _, name in ordered]
    return result


def run_validation(
    manifest_dir: Path,
    config_path: Path,
    candidate_path: Path,
    anchor_feature_path: Path,
    anchor_metadata_path: Path,
    feature_path: Path,
    metadata_path: Path,
    encoder: str,
    screen_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2BArtifactError("refusing to overwrite a frozen validation run")
    started = time.perf_counter()
    started_utc = utc_now()
    config = load_config(config_path)
    manifest_report = validate_manifest(manifest_dir, config_path)
    candidate_report = validate_candidates(
        candidate_path,
        manifest_dir,
        config_path,
        anchor_feature_path,
        anchor_metadata_path,
    )
    features, labels, sample_ids, feature_report = _load_features(
        feature_path,
        metadata_path,
        manifest_dir,
        config_path,
        encoder,
        load_labels=True,
    )
    if labels is None:
        raise AssertionError("validation did not load labels")
    screen_artifact, screen_rows = _validate_screen(
        screen_dir,
        config_path,
        str(candidate_report["candidate_set_id"]),
        str(feature_report["feature_file_sha256"]),
        encoder,
    )
    samples = read_manifest_samples(manifest_dir)
    candidate_ids, candidate_domains = _candidate_maps(candidate_path)
    projected, block_indices, role_indices = _projected_views(
        features, sample_ids, samples, candidate_ids, config, encoder
    )
    validation_config = config["validation"]
    audit_config = config["test_audit"]
    regularization = float(validation_config["ridge_regularization"])
    ece_bins = int(audit_config["ece_bins"])
    class_count = int(manifest_report["class_count"])
    shortlist_sizes = [int(value) for value in config["screen"]["shortlist_sizes"]]
    metric_rows = []
    selection_rows = []
    tasks = []
    for target_domain in config["dataset"]["domains"]:
        groups = _screen_groups(screen_rows, str(target_domain))
        union = sorted(
            {
                combination
                for ranking in groups.values()
                for combination in ranking[: max(shortlist_sizes)]
            }
        )
        names = sorted(
            name for name, domain in candidate_domains.items() if domain != target_domain
        )
        block_features = {name: projected[block_indices[name]] for name in names}
        block_labels = {name: labels[block_indices[name]] for name in names}
        target_indices = role_indices[(str(target_domain), "target_validation")]
        evaluated = _evaluate_combinations(
            union,
            block_features,
            block_labels,
            projected[target_indices],
            labels[target_indices],
            class_count,
            regularization,
            ece_bins,
            progress_label=f"validate encoder={encoder} target={target_domain}",
        )
        by_name = {str(row["combination"]): row for row in evaluated}
        for row in evaluated:
            metric_rows.append(
                {
                    "encoder": encoder,
                    "target_domain": target_domain,
                    **row,
                }
            )
        for (method, repeat), ranking in sorted(groups.items()):
            for size in shortlist_sizes:
                shortlist = ranking[:size]
                selected = min(
                    shortlist,
                    key=lambda name: (float(by_name[name]["brier_score"]), name),
                )
                selection_rows.append(
                    {
                        "encoder": encoder,
                        "target_domain": target_domain,
                        "method": method,
                        "repeat": repeat,
                        "shortlist_size": size,
                        "shortlist_sha256": hash_ids(shortlist),
                        "selected_combination": selected,
                        "validation_brier_score": by_name[selected]["brier_score"],
                        "validation_rank_within_shortlist": 1,
                    }
                )
        tasks.append(
            {
                "target_domain": target_domain,
                "logical_evaluations_per_method_and_size": shortlist_sizes,
                "physical_unique_validation_evaluations": len(union),
                "target_validation_count": len(target_indices),
                "target_validation_ids_sha256": hash_ids(sample_ids[target_indices]),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    selections_path = output_dir / "selections.csv"
    write_csv_atomic(metrics_path, METRIC_FIELDS, metric_rows)
    write_csv_atomic(selections_path, SELECTION_FIELDS, selection_rows)
    core: dict[str, object] = {
        "schema_version": VALIDATION_SCHEMA,
        "created_utc": utc_now(),
        "manifest_id": manifest_report["manifest_id"],
        "candidate_set_id": candidate_report["candidate_set_id"],
        "config_file_sha256": sha256_file(config_path),
        "encoder": encoder,
        "feature_file_sha256": feature_report["feature_file_sha256"],
        "screen_id": screen_artifact["screen_id"],
        "screen_manifest_file_sha256": sha256_file(screen_dir / "manifest.json"),
        "screen_rankings_file_sha256": sha256_file(screen_dir / "rankings.csv"),
        "metrics_file_sha256": sha256_file(metrics_path),
        "selections_file_sha256": sha256_file(selections_path),
        "metric_row_count": len(metric_rows),
        "selection_row_count": len(selection_rows),
        "runner_revision": git_revision(),
        "selection_frozen_before_target_test": True,
        "access": {
            "source_labels": True,
            "target_validation_labels": True,
            "target_test_features": False,
            "target_test_labels": False,
        },
        "tasks": tasks,
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }
    artifact = {**core, "validation_id": canonical_json_sha256(core)}
    write_json_atomic(output_dir / "manifest.json", artifact)
    return artifact


def _validate_validation(
    validation_dir: Path,
    config_path: Path,
    screen_artifact: Mapping[str, object],
    feature_file_sha256: str,
    encoder: str,
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]]]:
    artifact = read_json(validation_dir / "manifest.json")
    if artifact.get("schema_version") != VALIDATION_SCHEMA:
        raise E2BArtifactError(f"validation schema must be {VALIDATION_SCHEMA}")
    core = {key: value for key, value in artifact.items() if key != "validation_id"}
    if artifact.get("validation_id") != canonical_json_sha256(core):
        raise E2BArtifactError("validation identifier mismatch")
    if artifact.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("validation config mismatch")
    if artifact.get("screen_id") != screen_artifact["screen_id"]:
        raise E2BArtifactError("validation screen mismatch")
    if artifact.get("feature_file_sha256") != feature_file_sha256:
        raise E2BArtifactError("validation feature cache mismatch")
    if artifact.get("encoder") != encoder:
        raise E2BArtifactError("validation encoder mismatch")
    if artifact.get("selection_frozen_before_target_test") is not True:
        raise E2BArtifactError("validation selection is not frozen")
    access = artifact.get("access")
    if not isinstance(access, dict) or any(
        access.get(key) is not False
        for key in ("target_test_features", "target_test_labels")
    ):
        raise E2BArtifactError("validation accessed target test")
    metrics_path = validation_dir / "metrics.csv"
    selections_path = validation_dir / "selections.csv"
    if artifact.get("metrics_file_sha256") != sha256_file(metrics_path):
        raise E2BArtifactError("validation metrics hash mismatch")
    if artifact.get("selections_file_sha256") != sha256_file(selections_path):
        raise E2BArtifactError("validation selections hash mismatch")
    metrics = _read_csv(metrics_path, METRIC_FIELDS)
    selections = _read_csv(selections_path, SELECTION_FIELDS)
    if int(artifact.get("metric_row_count", -1)) != len(metrics):
        raise E2BArtifactError("validation metric row count mismatch")
    if int(artifact.get("selection_row_count", -1)) != len(selections):
        raise E2BArtifactError("validation selection row count mismatch")
    return artifact, metrics, selections


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
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


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    return 0.0 if denominator == 0.0 else float(
        np.dot(left_centered, right_centered) / denominator
    )


def run_test_audit(
    manifest_dir: Path,
    config_path: Path,
    candidate_path: Path,
    anchor_feature_path: Path,
    anchor_metadata_path: Path,
    feature_path: Path,
    metadata_path: Path,
    encoder: str,
    screen_dir: Path,
    validation_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2BArtifactError("refusing to overwrite a test audit")
    started = time.perf_counter()
    started_utc = utc_now()
    config = load_config(config_path)
    manifest_report = validate_manifest(manifest_dir, config_path)
    candidate_report = validate_candidates(
        candidate_path,
        manifest_dir,
        config_path,
        anchor_feature_path,
        anchor_metadata_path,
    )
    features, labels, sample_ids, feature_report = _load_features(
        feature_path,
        metadata_path,
        manifest_dir,
        config_path,
        encoder,
        load_labels=True,
    )
    if labels is None:
        raise AssertionError("test audit did not load labels")
    screen_artifact, screen_rows = _validate_screen(
        screen_dir,
        config_path,
        str(candidate_report["candidate_set_id"]),
        str(feature_report["feature_file_sha256"]),
        encoder,
    )
    validation_artifact, _, selection_rows = _validate_validation(
        validation_dir,
        config_path,
        screen_artifact,
        str(feature_report["feature_file_sha256"]),
        encoder,
    )
    samples = read_manifest_samples(manifest_dir)
    candidate_ids, candidate_domains = _candidate_maps(candidate_path)
    projected, block_indices, role_indices = _projected_views(
        features, sample_ids, samples, candidate_ids, config, encoder
    )
    regularization = float(config["validation"]["ridge_regularization"])
    audit_config = config["test_audit"]
    ece_bins = int(audit_config["ece_bins"])
    top_q = int(audit_config["top_q"])
    class_count = int(manifest_report["class_count"])
    expected_count = int(config["combinations"]["expected_combinations_per_target"])
    gates = config["primary_analysis"]["success_gate"]
    combination_rows = []
    audit_rows = []
    tasks = []
    for target_domain in config["dataset"]["domains"]:
        names = sorted(
            name for name, domain in candidate_domains.items() if domain != target_domain
        )
        combinations = [
            combination_name(value)
            for value in all_combinations(
                names, int(config["combinations"]["budget_blocks"])
            )
        ]
        if len(combinations) != expected_count:
            raise E2BArtifactError("test combination count mismatch")
        block_features = {name: projected[block_indices[name]] for name in names}
        block_labels = {name: labels[block_indices[name]] for name in names}
        test_indices = role_indices[(str(target_domain), "target_test")]
        evaluated = _evaluate_combinations(
            combinations,
            block_features,
            block_labels,
            projected[test_indices],
            labels[test_indices],
            class_count,
            regularization,
            ece_bins,
            progress_label=f"test encoder={encoder} target={target_domain}",
        )
        ordered = sorted(
            evaluated,
            key=lambda row: (float(row["brier_score"]), str(row["combination"])),
        )
        rank_by_name = {
            str(row["combination"]): rank for rank, row in enumerate(ordered, 1)
        }
        by_name = {str(row["combination"]): row for row in evaluated}
        oracle = float(ordered[0]["brier_score"])
        top = {str(row["combination"]) for row in ordered[:top_q]}
        for row in evaluated:
            combination_rows.append(
                {
                    "encoder": encoder,
                    "target_domain": target_domain,
                    **row,
                    "test_rank": rank_by_name[str(row["combination"])],
                    "in_test_top_q": int(str(row["combination"]) in top),
                    "normalized_regret": (
                        float(row["brier_score"]) - oracle
                    )
                    / max(abs(oracle), 1e-15),
                }
            )
        groups = _screen_groups(screen_rows, str(target_domain))
        selections = {
            (str(row["method"]), int(row["repeat"]), int(row["shortlist_size"])): row
            for row in selection_rows
            if row["target_domain"] == target_domain
        }
        for (method, repeat), ranking in sorted(groups.items()):
            brier_order_values = [float(by_name[name]["brier_score"]) for name in ranking]
            correlation = _spearman(
                list(range(1, len(ranking) + 1)), brier_order_values
            )
            for size in [int(value) for value in config["screen"]["shortlist_sizes"]]:
                shortlist = ranking[:size]
                selection = selections[(method, repeat, size)]
                if selection["shortlist_sha256"] != hash_ids(shortlist):
                    raise E2BArtifactError("validation shortlist differs from screen")
                selected_name = str(selection["selected_combination"])
                if selected_name not in shortlist:
                    raise E2BArtifactError("validation selected outside its shortlist")
                selected = by_name[selected_name]
                best = min(
                    (by_name[name] for name in shortlist),
                    key=lambda row: (float(row["brier_score"]), str(row["combination"])),
                )
                recall = len(top.intersection(shortlist)) / top_q
                reduction = (expected_count - size) / expected_count
                selected_regret = (
                    float(selected["brier_score"]) - oracle
                ) / max(abs(oracle), 1e-15)
                best_regret = (
                    float(best["brier_score"]) - oracle
                ) / max(abs(oracle), 1e-15)
                audit_rows.append(
                    {
                        "encoder": encoder,
                        "target_domain": target_domain,
                        "method": method,
                        "repeat": repeat,
                        "shortlist_size": size,
                        "combination_count": expected_count,
                        "shortlist_reduction": reduction,
                        "top_q": top_q,
                        "true_top_q_recall": recall,
                        "selected_combination": selected_name,
                        "selected_test_brier_score": selected["brier_score"],
                        "test_oracle_brier_score": oracle,
                        "selected_normalized_regret": selected_regret,
                        "selected_test_rank": rank_by_name[selected_name],
                        "best_shortlist_test_brier_score": best["brier_score"],
                        "best_shortlist_normalized_regret": best_regret,
                        "best_shortlist_test_rank": rank_by_name[str(best["combination"])],
                        "screen_spearman_vs_test_brier": correlation,
                        "passes_reduction_gate": int(
                            reduction
                            >= float(gates["primary_shortlist_reduction_at_least"])
                        ),
                        "passes_recall_gate": int(
                            recall >= float(gates["mean_true_top_q_recall_at_least"])
                        ),
                        "passes_regret_gate": int(
                            selected_regret
                            <= float(
                                gates[
                                    "mean_validation_selected_normalized_regret_at_most"
                                ]
                            )
                        ),
                    }
                )
        tasks.append(
            {
                "target_domain": target_domain,
                "target_test_count": len(test_indices),
                "target_test_ids_sha256": hash_ids(sample_ids[test_indices]),
                "combination_count": len(combinations),
                "test_oracle_combination": ordered[0]["combination"],
                "test_oracle_brier_score": oracle,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations_path = output_dir / "combinations.csv"
    audit_path = output_dir / "shortlist_audit.csv"
    combination_fields = METRIC_FIELDS + ("test_rank", "in_test_top_q", "normalized_regret")
    write_csv_atomic(combinations_path, combination_fields, combination_rows)
    write_csv_atomic(audit_path, AUDIT_FIELDS, audit_rows)
    core: dict[str, object] = {
        "schema_version": TEST_AUDIT_SCHEMA,
        "created_utc": utc_now(),
        "manifest_id": manifest_report["manifest_id"],
        "candidate_set_id": candidate_report["candidate_set_id"],
        "config_file_sha256": sha256_file(config_path),
        "encoder": encoder,
        "feature_file_sha256": feature_report["feature_file_sha256"],
        "screen_id": screen_artifact["screen_id"],
        "validation_id": validation_artifact["validation_id"],
        "validation_manifest_file_sha256": sha256_file(
            validation_dir / "manifest.json"
        ),
        "validation_selections_file_sha256": sha256_file(
            validation_dir / "selections.csv"
        ),
        "combinations_file_sha256": sha256_file(combinations_path),
        "audit_file_sha256": sha256_file(audit_path),
        "combination_row_count": len(combination_rows),
        "audit_row_count": len(audit_rows),
        "runner_revision": git_revision(),
        "access": {
            "screen_and_validation_hashes_verified_first": True,
            "target_test_features": True,
            "target_test_labels": True,
            "test_outcomes_change_screen_or_validation_selection": False,
        },
        "tasks": tasks,
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "device": "cpu",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }
    artifact = {**core, "test_audit_id": canonical_json_sha256(core)}
    write_json_atomic(output_dir / "manifest.json", artifact)
    return artifact


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--anchor-features", type=Path, required=True)
    parser.add_argument("--anchor-metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    _add_common(screen)
    validation = subparsers.add_parser("validate")
    _add_common(validation)
    validation.add_argument("--screen-dir", type=Path, required=True)
    audit = subparsers.add_parser("test-audit")
    _add_common(audit)
    audit.add_argument("--screen-dir", type=Path, required=True)
    audit.add_argument("--validation-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = (
        args.manifest_dir,
        args.config,
        args.candidates,
        args.anchor_features,
        args.anchor_metadata,
        args.features,
        args.metadata,
        args.encoder,
    )
    if args.command == "screen":
        artifact = run_screen(*common, args.output_dir)
        identifier = artifact["screen_id"]
    elif args.command == "validate":
        artifact = run_validation(*common, args.screen_dir, args.output_dir)
        identifier = artifact["validation_id"]
    else:
        artifact = run_test_audit(
            *common,
            args.screen_dir,
            args.validation_dir,
            args.output_dir,
        )
        identifier = artifact["test_audit_id"]
    print(
        json.dumps(
            {"status": args.command, "artifact_id": identifier},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
