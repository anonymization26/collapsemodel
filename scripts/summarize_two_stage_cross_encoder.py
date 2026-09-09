#!/usr/bin/env python3
"""Aggregate natural-shortlist results across frozen encoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np


METRICS = [
    "best_candidate_recall",
    "top3_candidate_recall",
    "validation_regret",
    "normalized_validation_regret",
    "removed_fraction",
]
HELDOUT_METRICS = ["test_regret_vs_exhaustive_validation_selection"]
OPTIONAL_METRICS = [
    "best_candidate_family_recall",
    "top3_candidate_family_recall",
    "selected_family_count",
    "removed_family_fraction",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_detail_result(path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = path.with_name(path.name.replace(
        "_shortlist_results.csv", "_shortlist_manifest.json",
    ))
    if not manifest_path.exists():
        raise ValueError(f"missing shortlist provenance manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("detail_sha256") != sha256_file(path):
        raise ValueError(f"shortlist result checksum mismatch: {path}")
    if int(manifest.get("detail_rows", -1)) != len(rows):
        raise ValueError(f"shortlist result row count mismatch: {path}")
    encoder = str(manifest.get("encoder", ""))
    targets = [str(value) for value in manifest.get("targets", [])]
    shortlist_sizes = [int(value) for value in manifest.get("shortlist_sizes", [])]
    methods = {str(value) for value in manifest.get("methods", [])}
    n_random = int(manifest.get("n_random", -1))
    if not encoder or not targets or not shortlist_sizes or not methods or n_random < 0:
        raise ValueError(f"invalid shortlist provenance manifest: {manifest_path}")
    deterministic_methods = methods - {"random"}
    expected = {
        (encoder, target, size, method, -1)
        for target in targets
        for size in shortlist_sizes
        for method in deterministic_methods
    }
    if "random" in methods:
        expected.update({
            (encoder, target, size, "random", replicate)
            for target in targets
            for size in shortlist_sizes
            for replicate in range(n_random)
        })
    observed = {
        (
            row["encoder"], row["target"], int(row["shortlist_size"]),
            row["method"], int(row["replicate"]),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"shortlist result is incomplete or has duplicate/extra rows: {path}")
    if manifest.get("target_test_used_for_selection") is not False:
        raise ValueError(f"shortlist result lacks a no-test-selection declaration: {path}")
    if int(manifest.get("schema_version", 1)) >= 2:
        if (
            manifest.get("target_test_read_during_adaptation_or_selection") is not False
            or manifest.get("selection_frozen_before_target_test") is not True
            or any(
                row.get(field, "") != ""
                for row in rows
                for field in (
                    "selected_test_accuracy", "oracle_selected_test_accuracy",
                    "test_regret_vs_exhaustive_validation_selection",
                )
            )
        ):
            raise ValueError(f"v2 selection result violates target-test isolation: {path}")
    return manifest_path


def validate_heldout_result(path: Path, rows: list[dict[str, str]]) -> list[Path]:
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"missing held-out provenance manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("heldout_results_sha256") != sha256_file(path):
        raise ValueError(f"held-out result checksum mismatch: {path}")
    if int(manifest.get("heldout_result_rows", -1)) != len(rows):
        raise ValueError(f"held-out result row count mismatch: {path}")
    if (
        manifest.get("target_test_access") != "after_selection_manifest_was_frozen"
        or manifest.get("target_test_used_for_selection") is not False
    ):
        raise ValueError(f"held-out result lacks test-isolation provenance: {path}")
    baseline_path = path.parent / str(manifest.get("heldout_stage2_baselines", ""))
    if (
        not baseline_path.is_file()
        or manifest.get("heldout_stage2_baselines_sha256") != sha256_file(baseline_path)
    ):
        raise ValueError(f"held-out Stage-2 baseline checksum mismatch: {baseline_path}")
    with baseline_path.open(newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    if not baseline_rows:
        raise ValueError(f"held-out Stage-2 baseline is empty: {baseline_path}")
    try:
        baseline_keys = [
            (str(row["baseline"]), int(row["adapter_seed"]), str(row["target"]))
            for row in baseline_rows
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid held-out Stage-2 baseline key: {baseline_path}"
        ) from error
    if len(baseline_keys) != len(set(baseline_keys)):
        raise ValueError(f"held-out Stage-2 baseline contains duplicate rows: {baseline_path}")
    for row in baseline_rows:
        try:
            accuracy = float(row["test_accuracy"])
            seconds = float(row["evaluation_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid held-out Stage-2 baseline row: {baseline_path}") from error
        if (
            row.get("encoder") != manifest.get("encoder")
            or not np.isfinite(accuracy)
            or not 0.0 <= accuracy <= 1.0
            or not np.isfinite(seconds)
            or seconds < 0.0
            or not row.get("device")
        ):
            raise ValueError(f"invalid held-out Stage-2 baseline row: {baseline_path}")
    selection_name = manifest.get("selection_manifest")
    if not isinstance(selection_name, str) or not selection_name:
        raise ValueError(f"held-out manifest lacks its selection freeze: {manifest_path}")
    selection_manifest_path = path.parent / selection_name
    if manifest.get("selection_manifest_sha256") != sha256_file(selection_manifest_path):
        raise ValueError(f"selection freeze checksum mismatch: {selection_manifest_path}")
    selection_manifest = json.loads(selection_manifest_path.read_text())
    selection_detail_path = selection_manifest_path.parent / selection_manifest["selection_detail"]
    if manifest.get("selection_detail_sha256") != sha256_file(selection_detail_path):
        raise ValueError(
            f"selection freeze checksum mismatch for detail: {selection_detail_path}"
        )
    with selection_detail_path.open(newline="") as stream:
        selection_rows = list(csv.DictReader(stream))
    validate_detail_result(selection_detail_path, selection_rows)
    protocol = selection_manifest.get("adaptation_protocol")
    adapter_states = selection_manifest.get("adapter_states")
    if not isinstance(protocol, dict) or not isinstance(adapter_states, dict):
        raise ValueError(
            f"selection freeze lacks adaptation provenance: {selection_manifest_path}"
        )
    try:
        seed_start = int(protocol["seed_start"])
        n_seeds = int(protocol["n_seeds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"selection freeze has an invalid seed protocol: {selection_manifest_path}"
        ) from error
    targets = [str(target) for target in selection_manifest.get("targets", [])]
    if n_seeds < 1 or not targets or len(targets) != len(set(targets)):
        raise ValueError(
            f"selection freeze has an invalid target/seed protocol: {selection_manifest_path}"
        )
    seeds = range(seed_start, seed_start + n_seeds)
    expected_baseline_keys = {
        ("frozen_identity", -1, target) for target in targets
    } | {
        ("random_adapter", seed, target)
        for seed in seeds
        for target in targets
    }
    if set(baseline_keys) != expected_baseline_keys or len(baseline_keys) != len(
        expected_baseline_keys
    ):
        raise ValueError(
            f"held-out Stage-2 baselines are incomplete or contain extra rows: {baseline_path}"
        )

    raw_name = manifest.get("raw_evaluations")
    sidecar_name = manifest.get("heldout_run_sidecar")
    if not isinstance(raw_name, str) or not isinstance(sidecar_name, str):
        raise ValueError(f"held-out manifest lacks raw evaluation provenance: {manifest_path}")
    raw_path = path.parent / raw_name
    sidecar_path = path.parent / sidecar_name
    if (
        not raw_path.is_file()
        or manifest.get("raw_evaluations_sha256") != sha256_file(raw_path)
    ):
        raise ValueError(f"held-out raw evaluation checksum mismatch: {raw_path}")
    if (
        not sidecar_path.is_file()
        or manifest.get("heldout_run_sidecar_sha256") != sha256_file(sidecar_path)
    ):
        raise ValueError(f"held-out run sidecar checksum mismatch: {sidecar_path}")
    with raw_path.open(newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if (
        len(raw_rows) != int(manifest.get("raw_evaluation_rows", -1))
        or len(raw_rows) != int(manifest.get("test_evaluation_keys", -1))
    ):
        raise ValueError(f"held-out raw evaluation row count mismatch: {raw_path}")
    run_record = json.loads(sidecar_path.read_text())
    configuration = run_record.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"held-out run sidecar lacks configuration: {sidecar_path}")
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    computed_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    fingerprint = run_record.get("run_fingerprint")
    expected_pairs = sorted({
        (str(row["selected_source"]), str(row["target"]))
        for row in selection_rows
    } | {
        (str(row["oracle_source"]), str(row["target"]))
        for row in selection_rows
    })
    try:
        recorded_pairs = sorted(
            (str(pair[0]), str(pair[1]))
            for pair in configuration.get("evaluation_pairs", [])
            if len(pair) == 2
        )
    except (TypeError, IndexError) as error:
        raise ValueError(f"held-out run has invalid evaluation pairs: {sidecar_path}") from error
    target_test_hashes = configuration.get("target_test_cache_sha256")
    expected_test_keys = {f"{target}:test" for target in targets}
    test_hashes_valid = (
        isinstance(target_test_hashes, dict)
        and set(target_test_hashes) == expected_test_keys
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower())
            for value in target_test_hashes.values()
        )
    )
    if (
        fingerprint != computed_fingerprint
        or configuration.get("selection_manifest_sha256")
        != manifest.get("selection_manifest_sha256")
        or configuration.get("selection_detail_sha256")
        != manifest.get("selection_detail_sha256")
        or configuration.get("encoder") != manifest.get("encoder")
        or configuration.get("target_test_access")
        != "after_selection_manifest_was_frozen"
        or configuration.get("script_sha256") != manifest.get("script_sha256")
        or manifest.get("script_sha256") != selection_manifest.get("script_sha256")
        or configuration.get("targets") != targets
        or recorded_pairs != expected_pairs
        or configuration.get("adapter_states") != adapter_states
        or int(configuration.get("seed_start", -1)) != seed_start
        or int(configuration.get("n_seeds", -1)) != n_seeds
        or configuration.get("width") != protocol.get("width")
        or configuration.get("ridge") != protocol.get("ridge")
        or configuration.get("deterministic_algorithms")
        != protocol.get("deterministic_algorithms")
        or configuration.get("target_train_cache_sha256")
        != protocol.get("target_cache_sha256")
        or configuration.get("target_train_metadata_sha256")
        != protocol.get("target_cache_metadata_sha256")
        or configuration.get("target_train_index_sha256")
        != protocol.get("target_cache_index_sha256")
        or not test_hashes_valid
    ):
        raise ValueError(f"held-out run sidecar is incompatible: {sidecar_path}")
    if {row.get("run_fingerprint") for row in baseline_rows} != {fingerprint}:
        raise ValueError(f"held-out Stage-2 baseline has the wrong run fingerprint: {baseline_path}")

    raw_keys = []
    test_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in raw_rows:
        try:
            seed = int(row["adapter_seed"])
            accuracy = float(row["test_accuracy"])
            seconds = float(row["evaluation_seconds"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid held-out raw evaluation row: {raw_path}") from error
        state = adapter_states.get(f"{row.get('source')}:{seed}")
        if not isinstance(state, dict):
            raise ValueError(f"raw evaluation references an unknown adapter: {raw_path}")
        if (
            row.get("run_fingerprint") != fingerprint
            or row.get("encoder") != manifest.get("encoder")
            or row.get("adapter_state_path") != state.get("path")
            or row.get("adapter_state_file_sha256") != state.get("file_sha256")
            or row.get("adapter_state_sha256") != state.get("state_sha256")
            or not np.isfinite(accuracy)
            or not 0.0 <= accuracy <= 1.0
            or not np.isfinite(seconds)
            or seconds < 0.0
            or not row.get("device")
        ):
            raise ValueError(f"invalid held-out raw evaluation row: {raw_path}")
        key = (row["source"], seed, row["target"])
        raw_keys.append(key)
        test_values[(row["source"], row["target"])].append(accuracy)
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError(f"held-out raw evaluation contains duplicate rows: {raw_path}")
    expected_raw_keys = {
        (source, seed, target)
        for source, target in expected_pairs
        for seed in seeds
    }
    if set(raw_keys) != expected_raw_keys or len(raw_keys) != len(expected_raw_keys):
        raise ValueError(
            f"held-out raw evaluations are incomplete or contain extra rows: {raw_path}"
        )
    if int(manifest.get("test_evaluation_keys", -1)) != len(expected_raw_keys):
        raise ValueError(f"held-out evaluation count is inconsistent: {manifest_path}")

    keys = ("encoder", "target", "method", "replicate", "shortlist_size")
    selected = {tuple(row[field] for field in keys): row for row in selection_rows}
    heldout = {tuple(row[field] for field in keys): row for row in rows}
    if set(selected) != set(heldout) or len(rows) != len(heldout):
        raise ValueError(f"held-out rows do not match the frozen selection: {path}")
    for key, selected_row in selected.items():
        heldout_row = heldout[key]
        changed = [
            field for field, value in selected_row.items()
            if heldout_row.get(field) != value
        ]
        if changed:
            raise ValueError(f"held-out result changed frozen fields {changed}: {path}")
        for field in (
            "oracle_selected_test_accuracy", "selected_test_accuracy",
            "test_regret_vs_exhaustive_validation_selection",
        ):
            value = float(heldout_row[field])
            if not np.isfinite(value):
                raise ValueError(f"invalid held-out metric {field}: {path}")
        selected_values = test_values.get(
            (selected_row["selected_source"], selected_row["target"]), []
        )
        oracle_values = test_values.get(
            (selected_row["oracle_source"], selected_row["target"]), []
        )
        if len(selected_values) != n_seeds or len(oracle_values) != n_seeds:
            raise ValueError(f"held-out result lacks raw selected/oracle evaluations: {path}")
        selected_mean = float(np.mean(selected_values))
        oracle_mean = float(np.mean(oracle_values))
        if (
            not np.isclose(float(heldout_row["selected_test_accuracy"]), selected_mean)
            or not np.isclose(float(heldout_row["oracle_selected_test_accuracy"]), oracle_mean)
            or not np.isclose(
                float(heldout_row["test_regret_vs_exhaustive_validation_selection"]),
                oracle_mean - selected_mean,
            )
            or int(heldout_row.get("heldout_adapter_seeds", -1)) != n_seeds
        ):
            raise ValueError(f"held-out result is inconsistent with raw evaluations: {path}")
    return [
        manifest_path, selection_manifest_path, selection_detail_path, baseline_path,
        raw_path, sidecar_path,
    ]


def read_detail_rows(
    results_dir: Path, expected_encoders: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[Path]]:
    heldout_paths = sorted(results_dir.glob("*/*_heldout_results.csv"))
    paths = heldout_paths or sorted(results_dir.glob("*/*_shortlist_results.csv"))
    if not paths:
        raise ValueError(f"no per-encoder shortlist results found under {results_dir}")
    rows: list[dict[str, str]] = []
    provenance_paths: list[Path] = []
    for path in paths:
        with path.open(newline="") as stream:
            file_rows = list(csv.DictReader(stream))
        if not file_rows:
            raise ValueError(f"empty result file: {path}")
        if heldout_paths:
            provenance_paths.extend(validate_heldout_result(path, file_rows))
        else:
            provenance_paths.append(validate_detail_result(path, file_rows))
        rows.extend(file_rows)
    encoders = sorted({row["encoder"] for row in rows})
    if expected_encoders is not None and encoders != sorted(expected_encoders):
        raise ValueError(f"encoders={encoders}, expected={sorted(expected_encoders)}")
    return rows, paths + provenance_paths


def mean_metric(rows: list[dict[str, str]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["shortlist_size"]))].append(row)

    output = []
    for (method, shortlist_size), values in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0]),
    ):
        encoder_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for value in values:
            encoder_groups[value["encoder"]].append(value)
        passing_encoders = sum(
            mean_metric(encoder_rows, "best_candidate_recall") >= 0.9
            and mean_metric(encoder_rows, "removed_fraction") >= 0.5
            for encoder_rows in encoder_groups.values()
        )
        record: dict[str, object] = {
            "method": method,
            "shortlist_size": shortlist_size,
            "rows": len(values),
            "encoders": len(encoder_groups),
            "encoder_target_pairs": len({
                (value["encoder"], value["target"]) for value in values
            }),
        }
        metrics = METRICS + [
            field for field in HELDOUT_METRICS
            if all(value.get(field, "") != "" for value in values)
        ] + [
            field for field in OPTIONAL_METRICS
            if all(value.get(field, "") != "" for value in values)
        ]
        record.update({f"{field}_mean": mean_metric(values, field) for field in metrics})
        record["passing_encoders"] = passing_encoders
        record["passes_overall"] = bool(
            record["best_candidate_recall_mean"] >= 0.9
            and record["removed_fraction_mean"] >= 0.5
        )
        output.append(record)
    return output


def paired_dpp_vs_rank(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    fixed = {
        (row["encoder"], row["target"], int(row["shortlist_size"]), row["method"]): row
        for row in rows
        if int(row["replicate"]) == -1
    }
    pairs = []
    keys = sorted({key[:3] for key in fixed})
    for encoder, target, shortlist_size in keys:
        dpp = fixed.get((encoder, target, shortlist_size, "dpp_subspace"))
        rank = fixed.get((encoder, target, shortlist_size, "rank_only"))
        if dpp is None or rank is None:
            continue
        dpp_shortlist = set(dpp["shortlist"].split("|"))
        rank_shortlist = set(rank["shortlist"].split("|"))
        pair = {
            "encoder": encoder,
            "target": target,
            "shortlist_size": shortlist_size,
            "oracle_source_family": dpp.get("oracle_source_family", ""),
            "dpp_best_candidate_recall": float(dpp["best_candidate_recall"]),
            "rank_best_candidate_recall": float(rank["best_candidate_recall"]),
            "dpp_validation_regret": float(dpp["validation_regret"]),
            "rank_validation_regret": float(rank["validation_regret"]),
            "dpp_minus_rank_best_recall": (
                float(dpp["best_candidate_recall"])
                - float(rank["best_candidate_recall"])
            ),
            "dpp_minus_rank_top3_recall": (
                float(dpp["top3_candidate_recall"])
                - float(rank["top3_candidate_recall"])
            ),
            "dpp_minus_rank_validation_regret": (
                float(dpp["validation_regret"])
                - float(rank["validation_regret"])
            ),
            "dpp_minus_rank_normalized_validation_regret": (
                float(dpp["normalized_validation_regret"])
                - float(rank["normalized_validation_regret"])
            ),
            "shortlist_jaccard": len(dpp_shortlist & rank_shortlist)
            / len(dpp_shortlist | rank_shortlist),
        }
        if all(
            row.get("test_regret_vs_exhaustive_validation_selection", "") != ""
            for row in (dpp, rank)
        ):
            pair["dpp_minus_rank_test_regret"] = (
                float(dpp["test_regret_vs_exhaustive_validation_selection"])
                - float(rank["test_regret_vs_exhaustive_validation_selection"])
            )
        if "best_candidate_family_recall" in dpp:
            pair.update({
                "dpp_best_candidate_family_recall": float(
                    dpp["best_candidate_family_recall"]
                ),
                "rank_best_candidate_family_recall": float(
                    rank["best_candidate_family_recall"]
                ),
                "dpp_minus_rank_best_family_recall": (
                    float(dpp["best_candidate_family_recall"])
                    - float(rank["best_candidate_family_recall"])
                ),
            })
        pairs.append(pair)
    if not pairs:
        raise ValueError("no paired DPP/rank-only rows found")
    return pairs


def compare(value: float, *, lower_is_better: bool, tolerance: float = 1e-12) -> str:
    if abs(value) <= tolerance:
        return "tie"
    is_win = value < 0 if lower_is_better else value > 0
    return "win" if is_win else "loss"


def summarize_pairs(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[int(pair["shortlist_size"])].append(pair)
    output = []
    for shortlist_size, values in sorted(grouped.items()):
        recall_outcomes = [
            compare(float(value["dpp_minus_rank_best_recall"]), lower_is_better=False)
            for value in values
        ]
        regret_outcomes = [
            compare(float(value["dpp_minus_rank_validation_regret"]), lower_is_better=True)
            for value in values
        ]
        output.append({
            "shortlist_size": shortlist_size,
            "encoder_target_pairs": len(values),
            "best_recall_delta_mean": fmean(
                float(value["dpp_minus_rank_best_recall"]) for value in values
            ),
            "validation_regret_delta_mean": fmean(
                float(value["dpp_minus_rank_validation_regret"]) for value in values
            ),
            "shortlist_jaccard_mean": fmean(
                float(value["shortlist_jaccard"]) for value in values
            ),
            "recall_wins": recall_outcomes.count("win"),
            "recall_ties": recall_outcomes.count("tie"),
            "recall_losses": recall_outcomes.count("loss"),
            "regret_wins": regret_outcomes.count("win"),
            "regret_ties": regret_outcomes.count("tie"),
            "regret_losses": regret_outcomes.count("loss"),
        })
    return output


def cluster_bootstrap_summaries(
    pairs: list[dict[str, object]],
    replicates: int,
    seed: int,
    cluster_specs: list[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    metric_specs = [
        ("dpp_best_candidate_recall", ">=0.9"),
        ("rank_best_candidate_recall", ">=0.9"),
        ("dpp_minus_rank_best_recall", ">0"),
        ("dpp_minus_rank_validation_regret", "<0"),
    ]
    if all("dpp_best_candidate_family_recall" in pair for pair in pairs):
        metric_specs.extend([
            ("dpp_best_candidate_family_recall", ">=0.9"),
            ("rank_best_candidate_family_recall", ">=0.9"),
            ("dpp_minus_rank_best_family_recall", ">0"),
        ])

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for pair in pairs:
        grouped[int(pair["shortlist_size"])].append(pair)
    rng = np.random.default_rng(seed)
    output = []
    for shortlist_size, values in sorted(grouped.items()):
        active_cluster_specs = cluster_specs
        if active_cluster_specs is None:
            active_cluster_specs = [("target", "target")]
            if all(str(value["oracle_source_family"]) for value in values):
                active_cluster_specs.append(
                    ("oracle_source_family", "source_family")
                )
        for cluster_field, cluster_level in active_cluster_specs:
            if not all(str(value.get(cluster_field, "")) for value in values):
                raise ValueError(f"missing cluster field: {cluster_field}")
            labels = sorted({str(value[cluster_field]) for value in values})
            for metric, favorable_rule in metric_specs:
                cluster_means = np.asarray([
                    fmean(
                        float(value[metric])
                        for value in values
                        if str(value[cluster_field]) == label
                    )
                    for label in labels
                ], dtype=np.float64)
                draws = rng.integers(
                    0, len(cluster_means), size=(replicates, len(cluster_means))
                )
                bootstrap = cluster_means[draws].mean(axis=1)
                if favorable_rule == ">=0.9":
                    probability_favorable = float(np.mean(bootstrap >= 0.9))
                elif favorable_rule == ">0":
                    probability_favorable = float(np.mean(bootstrap > 0.0))
                else:
                    probability_favorable = float(np.mean(bootstrap < 0.0))
                output.append({
                    "shortlist_size": shortlist_size,
                    "cluster_level": cluster_level,
                    "cluster_estimand": "equal_cluster_mean",
                    "clusters": len(labels),
                    "observations": len(values),
                    "metric": metric,
                    "estimate": float(cluster_means.mean()),
                    "ci_2_5": float(np.quantile(bootstrap, 0.025)),
                    "ci_97_5": float(np.quantile(bootstrap, 0.975)),
                    "favorable_rule": favorable_rule,
                    "bootstrap_probability_favorable": probability_favorable,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                })
    return output


def leave_one_encoder_out(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    deterministic = [row for row in rows if int(row["replicate"]) == -1]
    encoders = sorted({row["encoder"] for row in deterministic})
    output = []
    for excluded_encoder in encoders:
        retained = [
            row for row in deterministic if row["encoder"] != excluded_encoder
        ]
        for aggregate in aggregate_rows(retained):
            output.append({"excluded_encoder": excluded_encoder, **aggregate})
    return output


def target_summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    deterministic = [row for row in rows if int(row["replicate"]) == -1]
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in deterministic:
        grouped[(row["method"], int(row["shortlist_size"]), row["target"])].append(row)
    output = []
    for (method, shortlist_size, target), values in sorted(grouped.items()):
        metrics = METRICS + [
            field for field in HELDOUT_METRICS
            if all(value.get(field, "") != "" for value in values)
        ] + [
            field for field in OPTIONAL_METRICS
            if all(value.get(field, "") != "" for value in values)
        ]
        output.append({
            "method": method,
            "shortlist_size": shortlist_size,
            "target": target,
            "encoders": len({value["encoder"] for value in values}),
            **{f"{field}_mean": mean_metric(values, field) for field in metrics},
        })
    return output


def stage2_control_pairs(
    results_dir: Path, rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[Path]]:
    """Pair the validation-oracle source adapter with source-independent controls."""

    baseline_paths = sorted(results_dir.glob("*/*_heldout_stage2_baselines.csv"))
    if not baseline_paths:
        raise ValueError(f"no held-out Stage-2 controls found under {results_dir}")
    controls: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in baseline_paths:
        with path.open(newline="") as stream:
            baseline_rows = list(csv.DictReader(stream))
        for row in baseline_rows:
            try:
                value = float(row["test_accuracy"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid held-out control row: {path}") from error
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid held-out control accuracy: {path}")
            controls[(row["encoder"], row["target"])][row["baseline"]].append(value)

    units = sorted({(row["encoder"], row["target"]) for row in rows})
    if set(controls) != set(units):
        raise ValueError("held-out controls do not cover the encoder-target units")
    output = []
    for encoder, target in units:
        unit_controls = controls[(encoder, target)]
        identity = unit_controls.get("frozen_identity", [])
        random_adapter = unit_controls.get("random_adapter", [])
        if len(identity) != 1 or not random_adapter:
            raise ValueError(f"incomplete Stage-2 controls for {encoder}/{target}")
        unit_rows = [
            row for row in rows
            if row["encoder"] == encoder and row["target"] == target
        ]
        oracle_values = np.asarray([
            float(row["oracle_selected_test_accuracy"]) for row in unit_rows
        ], dtype=np.float64)
        if not np.isfinite(oracle_values).all() or not np.allclose(
            oracle_values, oracle_values[0], rtol=0.0, atol=1e-12,
        ):
            raise ValueError(f"held-out oracle differs across methods for {encoder}/{target}")
        oracle = float(oracle_values[0])
        random_mean = float(np.mean(random_adapter))
        output.append({
            "encoder": encoder,
            "target": target,
            "frozen_identity_test_accuracy": identity[0],
            "random_adapter_test_accuracy_mean": random_mean,
            "random_adapter_seeds": len(random_adapter),
            "validation_oracle_source_adapter_test_accuracy": oracle,
            "oracle_minus_frozen_identity": oracle - identity[0],
            "oracle_minus_random_adapter": oracle - random_mean,
        })
    return output, baseline_paths


def summarize_stage2_controls(
    pairs: list[dict[str, object]], replicates: int, seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    summary = [{
        "encoder_target_pairs": len(pairs),
        "encoders": len({str(row["encoder"]) for row in pairs}),
        "targets": len({str(row["target"]) for row in pairs}),
        "frozen_identity_test_accuracy_mean": fmean(
            float(row["frozen_identity_test_accuracy"]) for row in pairs
        ),
        "random_adapter_test_accuracy_mean": fmean(
            float(row["random_adapter_test_accuracy_mean"]) for row in pairs
        ),
        "validation_oracle_source_adapter_test_accuracy_mean": fmean(
            float(row["validation_oracle_source_adapter_test_accuracy"]) for row in pairs
        ),
        "oracle_minus_frozen_identity_mean": fmean(
            float(row["oracle_minus_frozen_identity"]) for row in pairs
        ),
        "oracle_minus_random_adapter_mean": fmean(
            float(row["oracle_minus_random_adapter"]) for row in pairs
        ),
    }]

    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for cluster_field in ("target", "encoder"):
        labels = sorted({str(row[cluster_field]) for row in pairs})
        for metric in ("oracle_minus_frozen_identity", "oracle_minus_random_adapter"):
            cluster_means = np.asarray([
                fmean(
                    float(row[metric]) for row in pairs
                    if str(row[cluster_field]) == label
                )
                for label in labels
            ], dtype=np.float64)
            draws = rng.integers(
                0, len(cluster_means), size=(replicates, len(cluster_means))
            )
            bootstrap = cluster_means[draws].mean(axis=1)
            bootstrap_rows.append({
                "cluster_level": cluster_field,
                "cluster_estimand": "equal_cluster_mean",
                "clusters": len(labels),
                "observations": len(pairs),
                "metric": metric,
                "estimate": float(cluster_means.mean()),
                "ci_2_5": float(np.quantile(bootstrap, 0.025)),
                "ci_97_5": float(np.quantile(bootstrap, 0.975)),
                "favorable_rule": ">0",
                "bootstrap_probability_favorable": float(np.mean(bootstrap > 0.0)),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
            })
    return summary, bootstrap_rows


def selected_vs_frozen_identity(
    rows: list[dict[str, str]], control_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
    identity = {
        (str(row["encoder"]), str(row["target"])):
        float(row["frozen_identity_test_accuracy"])
        for row in control_pairs
    }
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["replicate"]) == -1:
            grouped[(row["method"], int(row["shortlist_size"]))].append(row)
    output = []
    for (method, shortlist_size), values in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0]),
    ):
        deltas = [
            float(row["selected_test_accuracy"])
            - identity[(row["encoder"], row["target"])]
            for row in values
        ]
        output.append({
            "method": method,
            "shortlist_size": shortlist_size,
            "encoder_target_pairs": len(values),
            "selected_test_accuracy_mean": fmean(
                float(row["selected_test_accuracy"]) for row in values
            ),
            "selected_minus_frozen_identity_mean": fmean(deltas),
            "selected_beats_frozen_identity_count": sum(
                delta > 1e-12 for delta in deltas
            ),
            "selected_ties_frozen_identity_count": sum(abs(delta) <= 1e-12 for delta in deltas),
            "selected_loses_to_frozen_identity_count": sum(
                delta < -1e-12 for delta in deltas
            ),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument(
        "--expected-encoders", nargs="+",
        default=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
    )
    args = parser.parse_args()

    rows, input_paths = read_detail_rows(args.results_dir, args.expected_encoders)
    aggregate = aggregate_rows(rows)
    pairs = paired_dpp_vs_rank(rows)
    pair_summary = summarize_pairs(pairs)
    bootstrap_summary = cluster_bootstrap_summaries(
        pairs, args.bootstrap_replicates, args.bootstrap_seed,
    )
    output_rows = {
        "cross_encoder_summary.csv": aggregate,
        "dpp_vs_rank_paired.csv": pairs,
        "dpp_vs_rank_summary.csv": pair_summary,
        "cluster_bootstrap_summary.csv": bootstrap_summary,
        "leave_one_encoder_out.csv": leave_one_encoder_out(rows),
        "target_summary.csv": target_summaries(rows),
    }
    if all(row.get("selected_test_accuracy", "") != "" for row in rows):
        control_pairs, control_paths = stage2_control_pairs(args.results_dir, rows)
        control_summary, control_bootstrap = summarize_stage2_controls(
            control_pairs, args.bootstrap_replicates, args.bootstrap_seed,
        )
        output_rows.update({
            "stage2_control_pairs.csv": control_pairs,
            "stage2_control_summary.csv": control_summary,
            "stage2_control_cluster_bootstrap.csv": control_bootstrap,
            "selected_vs_frozen_identity.csv": selected_vs_frozen_identity(
                rows, control_pairs,
            ),
        })
        input_paths.extend(control_paths)
    for filename, values in output_rows.items():
        write_csv(args.results_dir / filename, values)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            str(path): sha256_file(path) for path in input_paths
        },
        "detail_rows": len(rows),
        "encoders": sorted({row["encoder"] for row in rows}),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "outputs": {
            filename: sha256_file(args.results_dir / filename)
            for filename in output_rows
        },
    }
    temporary = args.results_dir / "aggregation_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.results_dir / "aggregation_manifest.json")


if __name__ == "__main__":
    main()
