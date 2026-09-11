#!/usr/bin/env python3
"""Run leak-audited E2 selection and frozen-ridge evaluation.

The ``select`` command only passes unlabeled source-candidate features and
``target_selection`` features to selectors. It writes an immutable selection
artifact. The separate ``evaluate`` command verifies that artifact before it
loads labels or slices ``target_test`` rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.collapse_core import effective_rank_from_scatter  # noqa: E402
from metrics.target_conditioned_e2 import (  # noqa: E402
    CACHE_SCHEMA,
    ASSIGNMENT_FIELDS,
    E2ArtifactError,
    canonical_json_sha256,
    read_manifest_samples,
    sha256_file,
    validate_feature_cache,
    validate_manifest_bundle,
    write_json_atomic,
)
import two_stage_classic_baselines as classic  # noqa: E402


CONFIG_SCHEMA = "target-conditioned-e2-experiment-config-v1"
SELECTION_SCHEMA = "target-conditioned-e2-selection-v1"
EVALUATION_SCHEMA = "target-conditioned-e2-evaluation-v1"
TIE_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SelectionCache:
    dataset: str
    encoder: str
    manifest_id: str
    feature_file_sha256: str
    feature_revision: str
    domains: tuple[str, ...]
    sample_ids: np.ndarray
    features: np.ndarray
    sample_domains: Mapping[str, str]
    assignments: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class TaskView:
    target_domain: str
    target_features: np.ndarray
    target_sample_ids: tuple[str, ...]
    candidate_features: Mapping[str, np.ndarray]
    candidate_sample_ids: Mapping[str, tuple[str, ...]]
    candidate_domains: Mapping[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2ArtifactError(
            f"cannot read JSON object {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise E2ArtifactError(f"{path.name} must contain a JSON object")
    return value


def read_assignments(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != ASSIGNMENT_FIELDS:
                raise E2ArtifactError("assignments.csv columns are incorrect")
            return tuple(dict(row) for row in reader)
    except OSError as error:
        raise E2ArtifactError(f"cannot read assignments.csv: {error}") from error


def hash_ids(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") % (2**32)


def normalize_rows(features: np.ndarray, epsilon: float) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise E2ArtifactError("feature view must be a nonempty matrix")
    if not np.isfinite(matrix).all():
        raise E2ArtifactError("feature view contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(epsilon))


def _required_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise E2ArtifactError(f"{name} must be an object")
    return value


def _required_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise E2ArtifactError(f"{name} must be a nonempty list")
    return value


def load_experiment_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise E2ArtifactError(f"experiment config schema must be {CONFIG_SCHEMA}")
    if config.get("status") != "frozen-before-method-evaluation":
        raise E2ArtifactError("experiment config is not frozen")
    for field in ("datasets", "encoders"):
        values = [str(value) for value in _required_list(config.get(field), field)]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise E2ArtifactError(f"{field} must contain unique nonempty names")

    transform = _required_mapping(config.get("feature_transform"), "feature_transform")
    if transform != {
        "name": "row-l2-normalize",
        "epsilon": 1e-8,
        "intercept": False,
        "centering": "none",
    }:
        raise E2ArtifactError("unsupported or incomplete feature transform")

    selection = _required_mapping(config.get("selection"), "selection")
    budgets = [
        int(value) for value in _required_list(selection.get("budgets"), "budgets")
    ]
    if budgets != sorted(set(budgets)) or min(budgets) < 1:
        raise E2ArtifactError(
            "selection budgets must be sorted unique positive integers"
        )
    if int(selection.get("primary_budget", -1)) not in budgets:
        raise E2ArtifactError("primary budget must be one of the selection budgets")
    for key in ("prior_precision", "noise_variance"):
        value = float(selection.get(key, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise E2ArtifactError(f"selection.{key} must be positive")
    for key in ("subspace_rank", "rank_l", "random_repeats"):
        if int(selection.get(key, 0)) < 1:
            raise E2ArtifactError(f"selection.{key} must be positive")
    if selection.get("candidate_cost") != "one-per-manifest-block":
        raise E2ArtifactError("E2 v1 requires one unit of cost per manifest block")
    if not isinstance(selection.get("random_seed"), int):
        raise E2ArtifactError("selection.random_seed must be an integer")
    for key in (
        "uses_source_labels",
        "uses_target_labels",
        "uses_target_calibration",
        "uses_target_test",
    ):
        if selection.get(key) is not False:
            raise E2ArtifactError(f"selection.{key} must be false")

    evaluation = _required_mapping(config.get("evaluation"), "evaluation")
    if float(evaluation.get("ridge_regularization", math.nan)) <= 0.0:
        raise E2ArtifactError("ridge regularization must be positive")
    if evaluation.get("primary_metric") != "brier_score":
        raise E2ArtifactError("E2 v1 primary metric must be brier_score")
    if int(evaluation.get("ece_bins", 0)) < 1:
        raise E2ArtifactError("evaluation.ece_bins must be positive")
    expected_metrics = {
        "squared_loss",
        "nll",
        "accuracy",
        "macro_f1",
        "ece",
    }
    if {
        str(value)
        for value in _required_list(
            evaluation.get("supporting_metrics"), "supporting_metrics"
        )
    } != expected_metrics:
        raise E2ArtifactError("E2 v1 supporting metrics are incomplete")
    if evaluation.get("target_test_opened_after_selection_freeze") is not True:
        raise E2ArtifactError("target-test access order must be explicit")

    analysis = _required_mapping(config.get("primary_analysis"), "primary_analysis")
    if analysis.get("unit") != "dataset-target-domain":
        raise E2ArtifactError("primary analysis unit must be dataset-target-domain")
    if analysis.get("encoder_handling") != "average-within-unit":
        raise E2ArtifactError("encoders must be averaged within each task unit")
    if int(analysis.get("bootstrap_repeats", 0)) < 1:
        raise E2ArtifactError("bootstrap_repeats must be positive")
    confidence = float(analysis.get("confidence_level", math.nan))
    if not 0.0 < confidence < 1.0:
        raise E2ArtifactError("confidence_level must be in (0, 1)")
    for key in (
        "noninferiority_relative_tolerance",
        "minimum_mean_relative_improvement",
        "minimum_noninferior_unit_fraction",
    ):
        value = float(analysis.get(key, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise E2ArtifactError(f"primary_analysis.{key} must be in [0, 1]")

    groups = _required_mapping(config.get("method_groups"), "method_groups")
    methods = []
    for group in ("target_conditioned", "target_blind"):
        methods.extend(str(value) for value in _required_list(groups.get(group), group))
    if (
        len(methods) != len(set(methods))
        or "target_a" not in methods
        or "random" not in methods
    ):
        raise E2ArtifactError("method groups are duplicated or incomplete")
    return config


def load_selection_cache(
    manifest_dir: Path,
    feature_path: Path,
    metadata_path: Path,
) -> SelectionCache:
    """Load cache inputs without indexing the label payload."""

    report = validate_manifest_bundle(manifest_dir)
    manifest = read_json(manifest_dir / "manifest.json")
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise E2ArtifactError(f"cache schema must be {CACHE_SCHEMA}")
    if metadata.get("dataset") != report["dataset"]:
        raise E2ArtifactError("cache dataset differs from manifest")
    if metadata.get("manifest_id") != report["manifest_id"]:
        raise E2ArtifactError("cache manifest identifier mismatch")
    if metadata.get("feature_file_sha256") != sha256_file(feature_path):
        raise E2ArtifactError("feature file hash mismatch")
    if metadata.get("samples_file_sha256") != manifest.get("samples_file_sha256"):
        raise E2ArtifactError("cache samples hash mismatch")
    if metadata.get("assignments_file_sha256") != manifest.get(
        "assignments_file_sha256"
    ):
        raise E2ArtifactError("cache assignments hash mismatch")

    encoder = _required_mapping(metadata.get("encoder"), "encoder")
    encoder_name = str(encoder.get("name", ""))
    if not encoder_name:
        raise E2ArtifactError("cache encoder name is empty")
    code = _required_mapping(metadata.get("code"), "code")
    feature_revision = str(code.get("git_revision", ""))
    if not feature_revision:
        raise E2ArtifactError("feature cache has no source revision")

    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            if set(payload.files) != {"H", "y", "sample_ids"}:
                raise E2ArtifactError("feature cache keys are incorrect")
            features = np.asarray(payload["H"])
            sample_ids = np.asarray(payload["sample_ids"])
    except (OSError, ValueError) as error:
        raise E2ArtifactError(f"cannot read feature cache: {error}") from error

    if features.ndim != 2 or features.dtype != np.float32:
        raise E2ArtifactError("selection cache H must be a float32 matrix")
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise E2ArtifactError("selection cache sample_ids must be fixed-width strings")
    if not np.isfinite(features).all():
        raise E2ArtifactError("selection cache H contains non-finite values")
    samples = read_manifest_samples(manifest_dir)
    expected_ids = np.asarray([str(row["sample_id"]) for row in samples])
    if not np.array_equal(sample_ids.astype(str), expected_ids):
        raise E2ArtifactError("selection cache sample IDs differ from manifest")
    if list(features.shape) != metadata.get("feature_shape"):
        raise E2ArtifactError("selection cache feature shape differs from metadata")

    return SelectionCache(
        dataset=str(report["dataset"]),
        encoder=encoder_name,
        manifest_id=str(report["manifest_id"]),
        feature_file_sha256=str(metadata["feature_file_sha256"]),
        feature_revision=feature_revision,
        domains=tuple(str(value) for value in report["domains"]),
        sample_ids=sample_ids.astype(str),
        features=features,
        sample_domains={str(row["sample_id"]): str(row["domain"]) for row in samples},
        assignments=read_assignments(manifest_dir / "assignments.csv"),
    )


def build_task_view(
    cache: SelectionCache, target_domain: str, epsilon: float
) -> TaskView:
    id_to_index = {value: index for index, value in enumerate(cache.sample_ids)}
    rows = [
        row for row in cache.assignments if str(row["target_domain"]) == target_domain
    ]
    if len(rows) != len(cache.sample_ids):
        raise E2ArtifactError("target assignment view is incomplete")

    target_ids = [
        str(row["sample_id"]) for row in rows if row["role"] == "target_selection"
    ]
    if not target_ids:
        raise E2ArtifactError("target_selection view is empty")
    candidate_ids: dict[str, list[str]] = {}
    for row in rows:
        if row["role"] != "source_candidate":
            continue
        candidate = str(row["candidate_id"])
        if not candidate:
            raise E2ArtifactError("source candidate row has no candidate ID")
        candidate_ids.setdefault(candidate, []).append(str(row["sample_id"]))
    if not candidate_ids:
        raise E2ArtifactError("source candidate view is empty")

    def indices(values: Sequence[str]) -> np.ndarray:
        try:
            return np.asarray([id_to_index[value] for value in values], dtype=np.int64)
        except KeyError as error:
            raise E2ArtifactError(
                "assignment references an unknown sample ID"
            ) from error

    candidate_domains: dict[str, str] = {}
    candidate_features: dict[str, np.ndarray] = {}
    candidate_sample_ids: dict[str, tuple[str, ...]] = {}
    for name in sorted(candidate_ids):
        ids = tuple(candidate_ids[name])
        domains = {cache.sample_domains[value] for value in ids}
        if len(domains) != 1 or target_domain in domains:
            raise E2ArtifactError(
                "candidate block crosses domains or contains the target"
            )
        candidate_domains[name] = next(iter(domains))
        candidate_features[name] = normalize_rows(cache.features[indices(ids)], epsilon)
        candidate_sample_ids[name] = ids

    target_tuple = tuple(target_ids)
    return TaskView(
        target_domain=target_domain,
        target_features=normalize_rows(cache.features[indices(target_tuple)], epsilon),
        target_sample_ids=target_tuple,
        candidate_features=candidate_features,
        candidate_sample_ids=candidate_sample_ids,
        candidate_domains=candidate_domains,
    )


def gram_blocks(features: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    blocks = {}
    for name, value in features.items():
        matrix = np.asarray(value, dtype=np.float64)
        gram = matrix.T @ matrix
        blocks[name] = (gram + gram.T) / 2.0
    return blocks


def target_moment(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    moment = matrix.T @ matrix / matrix.shape[0]
    return (moment + moment.T) / 2.0


def target_a_value(
    information: np.ndarray,
    target_features: np.ndarray,
) -> float:
    """Compute tr(C_T A^-1) without materializing either inverse or C_T."""

    array = np.asarray(information, dtype=np.float64)
    matrix = (array + array.T) / 2.0
    chol = np.linalg.cholesky(matrix)
    targets = np.asarray(target_features, dtype=np.float64)
    projected = np.linalg.solve(chol, targets.T)
    return float(np.sum(projected * projected) / targets.shape[0])


def _strictly_better(value: float, best: float, minimize: bool) -> bool:
    tolerance = TIE_RELATIVE_TOLERANCE * max(abs(value), abs(best), 1.0)
    return value < best - tolerance if minimize else value > best + tolerance


def target_a_greedy(
    blocks: Mapping[str, np.ndarray],
    target_features: np.ndarray,
    maximum_budget: int,
    prior_precision: float,
    noise_variance: float,
) -> list[str]:
    names = sorted(blocks)
    dimension = next(iter(blocks.values())).shape[0]
    prior = prior_precision * np.eye(dimension)
    total = np.zeros((dimension, dimension), dtype=np.float64)
    selected: list[str] = []
    for _ in range(maximum_budget):
        best_name: str | None = None
        best_value = math.inf
        for name in names:
            if name in selected:
                continue
            information = prior + (total + blocks[name]) / noise_variance
            value = target_a_value(information, target_features)
            if best_name is None or _strictly_better(value, best_value, True):
                best_name = name
                best_value = value
        if best_name is None:
            raise E2ArtifactError("target A-optimal greedy selection stopped early")
        selected.append(best_name)
        total += blocks[best_name]
    return selected


def d_opt_greedy(
    blocks: Mapping[str, np.ndarray],
    maximum_budget: int,
    prior_precision: float,
    noise_variance: float,
) -> list[str]:
    names = sorted(blocks)
    dimension = next(iter(blocks.values())).shape[0]
    prior = prior_precision * np.eye(dimension)
    total = np.zeros((dimension, dimension), dtype=np.float64)
    selected: list[str] = []
    for _ in range(maximum_budget):
        best_name: str | None = None
        best_value = -math.inf
        for name in names:
            if name in selected:
                continue
            information = prior + (total + blocks[name]) / noise_variance
            chol = np.linalg.cholesky((information + information.T) / 2.0)
            value = 2.0 * float(np.log(np.diag(chol)).sum())
            if best_name is None or _strictly_better(value, best_value, False):
                best_name = name
                best_value = value
        if best_name is None:
            raise E2ArtifactError("D-optimal greedy selection stopped early")
        selected.append(best_name)
        total += blocks[best_name]
    return selected


def target_energy_selection(
    blocks: Mapping[str, np.ndarray],
    moment: np.ndarray,
    maximum_budget: int,
) -> list[str]:
    scores = {name: float(np.sum(moment * block.T)) for name, block in blocks.items()}
    return sorted(scores, key=lambda name: (-scores[name], name))[:maximum_budget]


def second_moment_mmd_greedy(
    blocks: Mapping[str, np.ndarray],
    sample_counts: Mapping[str, int],
    moment: np.ndarray,
    maximum_budget: int,
) -> list[str]:
    names = sorted(blocks)
    current = np.zeros_like(next(iter(blocks.values())))
    current_count = 0
    selected: list[str] = []
    for _ in range(maximum_budget):
        best_name: str | None = None
        best_value = math.inf
        for name in names:
            if name in selected:
                continue
            count = current_count + sample_counts[name]
            difference = (current + blocks[name]) / count - moment
            value = float(np.sum(difference * difference))
            if best_name is None or _strictly_better(value, best_value, True):
                best_name = name
                best_value = value
        if best_name is None:
            raise E2ArtifactError("second-moment MMD greedy selection stopped early")
        selected.append(best_name)
        current += blocks[best_name]
        current_count += sample_counts[best_name]
    return selected


def effective_rank_greedy(
    blocks: Mapping[str, np.ndarray],
    sample_counts: Mapping[str, int],
    maximum_budget: int,
) -> list[str]:
    names = sorted(blocks)
    dimension = next(iter(blocks.values())).shape[0]
    current = np.zeros((dimension, dimension), dtype=np.float64)
    current_count = 0
    selected: list[str] = []
    for _ in range(maximum_budget):
        best_name: str | None = None
        best_value = -math.inf
        for name in names:
            if name in selected:
                continue
            value = effective_rank_from_scatter(
                current + blocks[name],
                source_shape=(current_count + sample_counts[name], dimension),
            )
            if best_name is None or _strictly_better(value, best_value, False):
                best_name = name
                best_value = value
        if best_name is None:
            raise E2ArtifactError("effective-rank greedy selection stopped early")
        selected.append(best_name)
        current += blocks[best_name]
        current_count += sample_counts[best_name]
    return selected


def domain_balance_selection(view: TaskView, maximum_budget: int) -> list[str]:
    selected: list[str] = []
    domain_counts: Counter[str] = Counter()
    names = sorted(view.candidate_features)
    for _ in range(maximum_budget):
        remaining = [name for name in names if name not in selected]
        choice = min(
            remaining,
            key=lambda name: (
                domain_counts[view.candidate_domains[name]],
                -len(view.candidate_sample_ids[name]),
                name,
            ),
        )
        selected.append(choice)
        domain_counts[view.candidate_domains[choice]] += 1
    return selected


def run_method_selections(
    view: TaskView,
    config: Mapping[str, object],
    dataset: str,
) -> tuple[dict[tuple[str, int], list[str]], dict[str, float]]:
    selection = _required_mapping(config["selection"], "selection")
    maximum_budget = max(int(value) for value in selection["budgets"])
    if maximum_budget > len(view.candidate_features):
        raise E2ArtifactError("selection budget exceeds candidate count")
    prior = float(selection["prior_precision"])
    noise = float(selection["noise_variance"])
    subspace_rank = int(selection["subspace_rank"])
    rank_l = int(selection["rank_l"])
    sample_counts = {
        name: len(values) for name, values in view.candidate_sample_ids.items()
    }
    blocks = gram_blocks(view.candidate_features)
    moment = target_moment(view.target_features)
    names = sorted(view.candidate_features)

    sequences: dict[tuple[str, int], list[str]] = {}
    runtimes: dict[str, float] = {}

    def timed(method: str, function: Callable[[], list[str]]) -> None:
        started = time.perf_counter()
        value = function()
        elapsed = time.perf_counter() - started
        if len(value) != maximum_budget or len(set(value)) != maximum_budget:
            raise E2ArtifactError(f"{method} returned an invalid selection")
        if not set(value).issubset(names):
            raise E2ArtifactError(f"{method} returned an unknown candidate")
        sequences[(method, -1)] = value
        runtimes[method] = elapsed

    timed(
        "target_a",
        lambda: target_a_greedy(
            blocks,
            view.target_features,
            maximum_budget,
            prior,
            noise,
        ),
    )
    timed(
        "target_energy",
        lambda: target_energy_selection(blocks, moment, maximum_budget),
    )
    timed(
        "second_moment_mmd",
        lambda: second_moment_mmd_greedy(blocks, sample_counts, moment, maximum_budget),
    )
    timed(
        "bayesian_d",
        lambda: d_opt_greedy(blocks, maximum_budget, prior, noise),
    )
    timed(
        "effective_rank",
        lambda: effective_rank_greedy(blocks, sample_counts, maximum_budget),
    )
    timed(
        "collapse_4s",
        lambda: classic.collapse_greedy(
            dict(view.candidate_features), maximum_budget, top_k=subspace_rank
        ),
    )

    context_started = time.perf_counter()
    ranks, _, centroids, subspaces = classic.pool_statistics(
        dict(view.candidate_features), top_k=subspace_rank
    )
    similarity = classic.similarity_matrix(
        dict(view.candidate_features),
        names,
        "subspace",
        centroids,
        subspaces,
    )
    sketches = classic.pool_gram_sketches(dict(view.candidate_features), rank=rank_l)
    context_seconds = time.perf_counter() - context_started
    timed(
        "spectrum_rank_l",
        lambda: list(classic.rank_l_gram_greedy(sketches, maximum_budget)),
    )
    timed(
        "facility_subspace",
        lambda: classic.facility_location(similarity, names, maximum_budget),
    )
    timed(
        "kcenter_subspace",
        lambda: classic.k_center(similarity, names, ranks, maximum_budget),
    )
    timed(
        "dpp_subspace",
        lambda: classic.dpp_greedy(similarity, names, ranks, maximum_budget),
    )
    runtimes["classic_context"] = context_seconds

    timed(
        "size",
        lambda: sorted(sample_counts, key=lambda name: (-sample_counts[name], name))[
            :maximum_budget
        ],
    )
    timed(
        "domain_balance",
        lambda: domain_balance_selection(view, maximum_budget),
    )

    random_started = time.perf_counter()
    for repeat in range(int(selection["random_repeats"])):
        rng = np.random.default_rng(
            stable_seed(selection["random_seed"], dataset, view.target_domain, repeat)
        )
        sequence = rng.permutation(names).tolist()[:maximum_budget]
        sequences[("random", repeat)] = [str(value) for value in sequence]
    runtimes["random"] = (time.perf_counter() - random_started) / int(
        selection["random_repeats"]
    )

    expected = {
        str(method)
        for values in _required_mapping(
            config["method_groups"], "method_groups"
        ).values()
        for method in _required_list(values, "method group")
    }
    actual = {method for method, _ in sequences}
    if actual != expected:
        raise E2ArtifactError(
            f"implemented methods differ from config: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return sequences, runtimes


def _selected_indices(
    cache: SelectionCache,
    task: TaskView,
    selected: Sequence[str],
) -> np.ndarray:
    id_to_index = {value: index for index, value in enumerate(cache.sample_ids)}
    ids = sorted(
        value
        for candidate in selected
        for value in task.candidate_sample_ids[candidate]
    )
    return np.asarray([id_to_index[value] for value in ids], dtype=np.int64)


def _selection_records(
    cache: SelectionCache,
    task: TaskView,
    sequences: Mapping[tuple[str, int], Sequence[str]],
    runtimes: Mapping[str, float],
    budgets: Sequence[int],
) -> list[dict[str, object]]:
    records = []
    for budget in budgets:
        for (method, repeat), sequence in sorted(sequences.items()):
            selected_order = list(sequence[:budget])
            selected = sorted(selected_order)
            indices = _selected_indices(cache, task, selected)
            domain_counts = Counter(task.candidate_domains[name] for name in selected)
            records.append(
                {
                    "budget": budget,
                    "method": method,
                    "repeat": repeat,
                    "selected": selected,
                    "selected_order": selected_order,
                    "selected_sample_count": int(indices.size),
                    "selected_sample_ids_sha256": hash_ids(cache.sample_ids[indices]),
                    "selected_domain_counts": dict(sorted(domain_counts.items())),
                    "selection_seconds_for_max_budget": float(runtimes[method]),
                }
            )
    return records


def run_selection(
    manifest_dir: Path,
    feature_path: Path,
    metadata_path: Path,
    config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise E2ArtifactError("refusing to overwrite an existing frozen selection")
    started_utc = utc_now()
    started = time.perf_counter()
    config = load_experiment_config(config_path)
    cache = load_selection_cache(manifest_dir, feature_path, metadata_path)
    if cache.dataset not in [str(value) for value in config["datasets"]]:
        raise E2ArtifactError("dataset is not pre-registered")
    if cache.encoder not in [str(value) for value in config["encoders"]]:
        raise E2ArtifactError("encoder is not pre-registered")
    transform = _required_mapping(config["feature_transform"], "feature_transform")
    epsilon = float(transform["epsilon"])
    selection_config = _required_mapping(config["selection"], "selection")
    budgets = [int(value) for value in selection_config["budgets"]]

    tasks = []
    for target_domain in cache.domains:
        task_started = time.perf_counter()
        view = build_task_view(cache, target_domain, epsilon)
        sequences, runtimes = run_method_selections(view, config, cache.dataset)
        inventory = []
        for name in sorted(view.candidate_features):
            ids = view.candidate_sample_ids[name]
            inventory.append(
                {
                    "candidate_id": name,
                    "domain": view.candidate_domains[name],
                    "sample_count": len(ids),
                    "sample_ids_sha256": hash_ids(ids),
                }
            )
        tasks.append(
            {
                "target_domain": target_domain,
                "algorithmic_access": {
                    "roles": ["source_candidate.H", "target_selection.H"],
                    "labels_accessed": False,
                    "target_calibration_accessed": False,
                    "target_test_accessed": False,
                    "target_selection_count": len(view.target_sample_ids),
                    "target_selection_ids_sha256": hash_ids(view.target_sample_ids),
                    "source_candidate_count": sum(
                        len(value) for value in view.candidate_sample_ids.values()
                    ),
                    "source_candidate_ids_sha256": hash_ids(
                        value
                        for name in sorted(view.candidate_sample_ids)
                        for value in view.candidate_sample_ids[name]
                    ),
                },
                "candidate_inventory": inventory,
                "method_runtime_seconds": dict(sorted(runtimes.items())),
                "selections": _selection_records(
                    cache, view, sequences, runtimes, budgets
                ),
                "task_seconds": time.perf_counter() - task_started,
            }
        )
        print(
            f"selected dataset={cache.dataset} encoder={cache.encoder} "
            f"target={target_domain} seconds={tasks[-1]['task_seconds']:.2f}",
            flush=True,
        )

    artifact: dict[str, object] = {
        "schema_version": SELECTION_SCHEMA,
        "input": {
            "dataset": cache.dataset,
            "encoder": cache.encoder,
            "manifest_id": cache.manifest_id,
            "feature_file_sha256": cache.feature_file_sha256,
            "feature_revision": cache.feature_revision,
        },
        "experiment_config_file_sha256": sha256_file(config_path),
        "experiment_config": config,
        "runner_revision": git_revision(),
        "selection_frozen_before_target_test": True,
        "storage_access": {
            "H": "monolithic-cache-loaded-then-role-sliced",
            "y": "payload-key-not-indexed-or-loaded",
            "algorithmic_views_only": True,
        },
        "tasks": tasks,
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "device": "cpu",
        },
    }
    artifact["selection_id"] = canonical_json_sha256(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, artifact)
    return artifact


def validate_selection_artifact(
    artifact: Mapping[str, object],
    config: Mapping[str, object],
    config_sha256: str,
    cache: SelectionCache,
    *,
    expected_runner_revision: str | None = None,
) -> None:
    if artifact.get("schema_version") != SELECTION_SCHEMA:
        raise E2ArtifactError(f"selection schema must be {SELECTION_SCHEMA}")
    core = {key: value for key, value in artifact.items() if key != "selection_id"}
    if artifact.get("selection_id") != canonical_json_sha256(core):
        raise E2ArtifactError("selection identifier mismatch")
    if artifact.get("experiment_config_file_sha256") != config_sha256:
        raise E2ArtifactError("selection uses a different experiment config file")
    if artifact.get("experiment_config") != config:
        raise E2ArtifactError("selection embeds a different experiment config")
    if artifact.get("selection_frozen_before_target_test") is not True:
        raise E2ArtifactError("selection was not frozen before target-test access")
    runner_revision = (
        git_revision()
        if expected_runner_revision is None
        else expected_runner_revision
    )
    if artifact.get("runner_revision") != runner_revision:
        raise E2ArtifactError("selection and evaluator runner revisions differ")
    inputs = _required_mapping(artifact.get("input"), "selection input")
    expected_input = {
        "dataset": cache.dataset,
        "encoder": cache.encoder,
        "manifest_id": cache.manifest_id,
        "feature_file_sha256": cache.feature_file_sha256,
        "feature_revision": cache.feature_revision,
    }
    if inputs != expected_input:
        raise E2ArtifactError("selection input differs from the validated cache")

    selection_config = _required_mapping(config["selection"], "selection")
    budgets = [int(value) for value in selection_config["budgets"]]
    random_repeats = int(selection_config["random_repeats"])
    groups = _required_mapping(config["method_groups"], "method_groups")
    methods = {
        str(method)
        for values in groups.values()
        for method in _required_list(values, "method group")
    }
    tasks = _required_list(artifact.get("tasks"), "selection tasks")
    if [
        str(task.get("target_domain")) for task in tasks if isinstance(task, dict)
    ] != list(cache.domains):
        raise E2ArtifactError("selection task domains are incomplete or reordered")
    epsilon = float(
        _required_mapping(config["feature_transform"], "feature_transform")["epsilon"]
    )
    for task_value in tasks:
        task = _required_mapping(task_value, "selection task")
        target_domain = str(task["target_domain"])
        current_view = build_task_view(cache, target_domain, epsilon)
        access = _required_mapping(task.get("algorithmic_access"), "algorithmic_access")
        if access.get("labels_accessed") is not False:
            raise E2ArtifactError("selection reports label access")
        if access.get("target_calibration_accessed") is not False:
            raise E2ArtifactError("selection reports target-calibration access")
        if access.get("target_test_accessed") is not False:
            raise E2ArtifactError("selection reports target-test access")
        expected_access = {
            "roles": ["source_candidate.H", "target_selection.H"],
            "labels_accessed": False,
            "target_calibration_accessed": False,
            "target_test_accessed": False,
            "target_selection_count": len(current_view.target_sample_ids),
            "target_selection_ids_sha256": hash_ids(current_view.target_sample_ids),
            "source_candidate_count": sum(
                len(value) for value in current_view.candidate_sample_ids.values()
            ),
            "source_candidate_ids_sha256": hash_ids(
                value
                for name in sorted(current_view.candidate_sample_ids)
                for value in current_view.candidate_sample_ids[name]
            ),
        }
        if access != expected_access:
            raise E2ArtifactError("selection access record differs from manifest views")
        inventory = _required_list(
            task.get("candidate_inventory"), "candidate inventory"
        )
        expected_inventory = [
            {
                "candidate_id": name,
                "domain": current_view.candidate_domains[name],
                "sample_count": len(current_view.candidate_sample_ids[name]),
                "sample_ids_sha256": hash_ids(current_view.candidate_sample_ids[name]),
            }
            for name in sorted(current_view.candidate_features)
        ]
        if inventory != expected_inventory:
            raise E2ArtifactError(
                "selection candidate inventory differs from manifest views"
            )
        candidate_names = {value["candidate_id"] for value in expected_inventory}
        records = _required_list(task.get("selections"), "selection records")
        counts: Counter[tuple[int, str]] = Counter()
        prefixes: dict[tuple[str, int], dict[int, list[str]]] = {}
        for value in records:
            record = _required_mapping(value, "selection record")
            budget = int(record.get("budget", -1))
            method = str(record.get("method", ""))
            selected = [str(item) for item in record.get("selected", [])]
            selected_order = [str(item) for item in record.get("selected_order", [])]
            if budget not in budgets or method not in methods:
                raise E2ArtifactError(
                    "selection record has an unknown method or budget"
                )
            if len(selected) != budget or len(set(selected)) != budget:
                raise E2ArtifactError("selection record has the wrong cardinality")
            if selected != sorted(selected) or set(selected) != set(selected_order):
                raise E2ArtifactError("selection record order fields are inconsistent")
            if not set(selected).issubset(candidate_names):
                raise E2ArtifactError("selection record contains an unknown candidate")
            indices = _selected_indices(cache, current_view, selected)
            if int(record.get("selected_sample_count", -1)) != int(indices.size):
                raise E2ArtifactError(
                    "selection sample count differs from manifest views"
                )
            if record.get("selected_sample_ids_sha256") != hash_ids(
                cache.sample_ids[indices]
            ):
                raise E2ArtifactError(
                    "selection sample hash differs from manifest views"
                )
            expected_domains = dict(
                sorted(
                    Counter(
                        current_view.candidate_domains[name] for name in selected
                    ).items()
                )
            )
            if record.get("selected_domain_counts") != expected_domains:
                raise E2ArtifactError(
                    "selection domain counts differ from manifest views"
                )
            repeat = int(record.get("repeat", -2))
            if (method == "random" and repeat not in range(random_repeats)) or (
                method != "random" and repeat != -1
            ):
                raise E2ArtifactError("selection repeat index is invalid")
            prefixes.setdefault((method, repeat), {})[budget] = selected_order
            counts[(budget, method)] += 1
        for budget in budgets:
            for method in methods:
                expected = random_repeats if method == "random" else 1
                if counts[(budget, method)] != expected:
                    raise E2ArtifactError("selection method coverage is incomplete")
        for values in prefixes.values():
            maximum = values[max(budgets)]
            if any(values[budget] != maximum[:budget] for budget in budgets):
                raise E2ArtifactError(
                    "selection records are not nested budget prefixes"
                )


def fit_ridge_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    class_count: int,
    regularization: float,
) -> tuple[np.ndarray, str]:
    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise E2ArtifactError("ridge train/test feature dimensions differ")
    if labels.shape != (train.shape[0],):
        raise E2ArtifactError("ridge labels do not match training rows")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= class_count:
        raise E2ArtifactError("ridge labels are empty or outside the class vocabulary")
    one_hot = np.eye(class_count, dtype=np.float64)[labels]
    if train.shape[0] < train.shape[1]:
        kernel = train @ train.T
        kernel.flat[:: kernel.shape[0] + 1] += regularization
        coefficients = np.linalg.solve(kernel, one_hot)
        return test @ train.T @ coefficients, "dual"
    gram = train.T @ train
    gram.flat[:: gram.shape[0] + 1] += regularization
    weights = np.linalg.solve(gram, train.T @ one_hot)
    return test @ weights, "primal"


def prediction_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    ece_bins: int,
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.shape != (len(truth), class_count):
        raise E2ArtifactError("prediction score shape is incorrect")
    one_hot = np.eye(class_count, dtype=np.float64)[truth]
    squared_loss = float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    nll = float(
        -np.mean(np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-15)))
    )
    accuracy = float(np.mean(predicted == truth))

    f1_values = []
    for label in range(class_count):
        true_positive = int(np.sum((predicted == label) & (truth == label)))
        false_positive = int(np.sum((predicted == label) & (truth != label)))
        false_negative = int(np.sum((predicted != label) & (truth == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(np.float64)
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, ece_bins, endpoint=False):
        upper = lower + 1.0 / ece_bins
        mask = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "squared_loss": squared_loss,
        "brier_score": brier,
        "nll": nll,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "ece": ece,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise E2ArtifactError("cannot write an empty evaluation table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_evaluation(
    manifest_dir: Path,
    feature_path: Path,
    metadata_path: Path,
    config_path: Path,
    selection_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    if (output_dir / "raw.csv").exists() or (output_dir / "manifest.json").exists():
        raise E2ArtifactError("refusing to overwrite an existing E2 evaluation")
    started_utc = utc_now()
    started = time.perf_counter()
    config = load_experiment_config(config_path)
    cache = load_selection_cache(manifest_dir, feature_path, metadata_path)
    selection_artifact = read_json(selection_path)
    validate_selection_artifact(
        selection_artifact,
        config,
        sha256_file(config_path),
        cache,
    )
    validate_feature_cache(feature_path, metadata_path, manifest_dir, cache.encoder)

    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            features = np.asarray(payload["H"])
            labels = np.asarray(payload["y"])
            sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    except (OSError, ValueError) as error:
        raise E2ArtifactError(f"cannot load evaluation cache: {error}") from error
    transform = _required_mapping(config["feature_transform"], "feature_transform")
    epsilon = float(transform["epsilon"])
    evaluation = _required_mapping(config["evaluation"], "evaluation")
    regularization = float(evaluation["ridge_regularization"])
    ece_bins = int(evaluation["ece_bins"])
    manifest = read_json(manifest_dir / "manifest.json")
    class_count = len(_required_mapping(manifest.get("class_to_id"), "class_to_id"))
    id_to_index = {value: index for index, value in enumerate(sample_ids)}
    assignment_by_target = {
        domain: [
            row for row in cache.assignments if str(row["target_domain"]) == domain
        ]
        for domain in cache.domains
    }
    task_artifacts = {
        str(
            _required_mapping(value, "selection task")["target_domain"]
        ): _required_mapping(value, "selection task")
        for value in _required_list(selection_artifact["tasks"], "selection tasks")
    }

    rows: list[dict[str, object]] = []
    metric_cache: dict[
        tuple[str, tuple[str, ...]], tuple[dict[str, float], str, float]
    ] = {}
    target_access = []
    for target_domain in cache.domains:
        assignment_rows = assignment_by_target[target_domain]
        candidate_ids: dict[str, list[str]] = {}
        target_test_ids = []
        for row in assignment_rows:
            if row["role"] == "source_candidate":
                candidate_ids.setdefault(str(row["candidate_id"]), []).append(
                    str(row["sample_id"])
                )
            elif row["role"] == "target_test":
                target_test_ids.append(str(row["sample_id"]))
        if not target_test_ids:
            raise E2ArtifactError("target_test view is empty")
        test_indices = np.asarray(
            [id_to_index[value] for value in target_test_ids], dtype=np.int64
        )
        test_features = normalize_rows(features[test_indices], epsilon)
        test_labels = labels[test_indices]
        target_access.append(
            {
                "target_domain": target_domain,
                "target_test_count": len(target_test_ids),
                "target_test_ids_sha256": hash_ids(target_test_ids),
            }
        )

        task = task_artifacts[target_domain]
        for record_value in _required_list(task["selections"], "selection records"):
            record = _required_mapping(record_value, "selection record")
            selected = tuple(str(value) for value in record["selected"])
            key = (target_domain, selected)
            cache_hit = key in metric_cache
            if not cache_hit:
                train_ids = sorted(
                    value
                    for candidate in selected
                    for value in candidate_ids[candidate]
                )
                train_indices = np.asarray(
                    [id_to_index[value] for value in train_ids], dtype=np.int64
                )
                train_features = normalize_rows(features[train_indices], epsilon)
                metric_started = time.perf_counter()
                scores, solver = fit_ridge_scores(
                    train_features,
                    labels[train_indices],
                    test_features,
                    class_count,
                    regularization,
                )
                metrics = prediction_metrics(scores, test_labels, class_count, ece_bins)
                metric_cache[key] = (
                    metrics,
                    solver,
                    time.perf_counter() - metric_started,
                )
            metrics, solver, metric_seconds = metric_cache[key]
            rows.append(
                {
                    "dataset": cache.dataset,
                    "encoder": cache.encoder,
                    "target_domain": target_domain,
                    "budget": int(record["budget"]),
                    "method": str(record["method"]),
                    "repeat": int(record["repeat"]),
                    "selected": "|".join(selected),
                    "selected_sample_count": int(record["selected_sample_count"]),
                    "selected_sample_ids_sha256": str(
                        record["selected_sample_ids_sha256"]
                    ),
                    "target_test_count": len(target_test_ids),
                    "ridge_regularization": regularization,
                    "ridge_solver": solver,
                    "evaluation_seconds": metric_seconds,
                    "evaluation_cache_hit": int(cache_hit),
                    **metrics,
                }
            )
        print(
            f"evaluated dataset={cache.dataset} encoder={cache.encoder} "
            f"target={target_domain} rows={len(task['selections'])}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.csv"
    _write_csv(raw_path, rows)
    evaluation_manifest: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "dataset": cache.dataset,
        "encoder": cache.encoder,
        "manifest_id": cache.manifest_id,
        "feature_file_sha256": cache.feature_file_sha256,
        "feature_revision": cache.feature_revision,
        "runner_revision": git_revision(),
        "experiment_config_file_sha256": sha256_file(config_path),
        "selection_id": selection_artifact["selection_id"],
        "selection_file_sha256": sha256_file(selection_path),
        "selection_frozen_before_target_test": True,
        "evaluation_access": {
            "storage_integrity_validation_reads_full_cache": True,
            "source_labels": "selected-source-samples-only",
            "target_labels": "target_test-only",
            "target_calibration_used_by_metrics": False,
            "target_test_access": target_access,
        },
        "row_count": len(rows),
        "unique_fits": len(metric_cache),
        "raw_file_sha256": sha256_file(raw_path),
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "device": "cpu",
        },
    }
    evaluation_manifest["evaluation_id"] = canonical_json_sha256(evaluation_manifest)
    write_json_atomic(output_dir / "manifest.json", evaluation_manifest)
    return evaluation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("select", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest-dir", type=Path, required=True)
        child.add_argument("--features", type=Path, required=True)
        child.add_argument("--metadata", type=Path, required=True)
        child.add_argument("--config", type=Path, required=True)
        if command == "select":
            child.add_argument("--output", type=Path, required=True)
        else:
            child.add_argument("--selection", type=Path, required=True)
            child.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        result = run_selection(
            args.manifest_dir,
            args.features,
            args.metadata,
            args.config,
            args.output,
        )
        print(
            json.dumps(
                {
                    "status": "selection-frozen",
                    "selection_id": result["selection_id"],
                    "tasks": len(result["tasks"]),
                },
                sort_keys=True,
            )
        )
    else:
        result = run_evaluation(
            args.manifest_dir,
            args.features,
            args.metadata,
            args.config,
            args.selection,
            args.out_dir,
        )
        print(
            json.dumps(
                {
                    "status": "evaluated",
                    "evaluation_id": result["evaluation_id"],
                    "rows": result["row_count"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
