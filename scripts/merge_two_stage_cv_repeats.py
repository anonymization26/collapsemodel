#!/usr/bin/env python3
"""Merge repeated target-CV utilities without duplicating adapter seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np

import two_stage_natural_shortlist as natural


KEY_FIELDS = ["encoder", "source", "adapter_seed", "target"]
MEAN_FIELDS = ["training_seconds", "evaluation_seconds"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["target_cv_folds"]) < 2:
            raise ValueError("all inputs must use cross-validation")
        grouped[tuple(row[field] for field in KEY_FIELDS)].append(row)

    merged = []
    expected_partitions = None
    for key, values in sorted(grouped.items()):
        raw_partition_seeds = [
            seed
            for value in values
            for seed in value["target_cv_seed"].split("|")
        ]
        partition_seeds = sorted(set(raw_partition_seeds), key=int)
        if len(partition_seeds) != len(raw_partition_seeds):
            raise ValueError(f"duplicate CV partition for key {key}")
        if expected_partitions is None:
            expected_partitions = partition_seeds
        elif partition_seeds != expected_partitions:
            raise ValueError(f"incomplete CV partitions for key {key}")

        state_hashes = {value["adapter_state_sha256"] for value in values}
        if len(state_hashes) != 1:
            raise ValueError(f"adapter parameters changed across CV partitions for key {key}")
        fold_values = [
            float(fold_accuracy)
            for value in values
            for fold_accuracy in value["validation_fold_accuracies"].split("|")
        ]
        output: dict[str, object] = dict(values[0])
        validation_values = [float(value["validation_accuracy"]) for value in values]
        output["validation_accuracy"] = fmean(validation_values)
        output["validation_accuracy_std"] = float(np.std(fold_values))
        output["validation_fold_accuracies"] = "|".join(
            f"{value:.10f}" for value in fold_values
        )
        output["target_validation_protocol"] = (
            f"repeated_stratified_{values[0]['target_cv_folds']}_fold"
        )
        output["target_cv_seed"] = "|".join(partition_seeds)
        output["target_cv_partitions"] = len(partition_seeds)
        output["validation_partition_accuracies"] = "|".join(
            f"{value:.10f}" for value in validation_values
        )
        for field in MEAN_FIELDS:
            output[field] = fmean(float(value[field]) for value in values)
        output["cv_partition_count"] = len(partition_seeds)
        if all("validation_examples" in value for value in values):
            output["validation_examples"] = sum(
                int(value["validation_examples"]) for value in values
            )
        if all("unique_validation_examples" in value for value in values):
            unique_counts = {int(value["unique_validation_examples"]) for value in values}
            if len(unique_counts) != 1:
                raise ValueError(f"target sample count changed across partitions for key {key}")
            output["unique_validation_examples"] = next(iter(unique_counts))
        output["validation_accuracy_std_across_partitions"] = float(
            np.std(validation_values)
        )
        merged.append(output)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    screening_manifest = json.loads(args.manifest.read_text())
    rows = []
    configurations = []
    input_fingerprints = []
    state_artifacts: dict[tuple[str, ...], tuple[Path, str]] = {}
    for path in args.input:
        current_rows = natural.read_adaptation(
            [path], manifest=screening_manifest, manifest_path=args.manifest,
        )
        rows.extend(current_rows)
        for row in current_rows:
            key = tuple(row[field] for field in KEY_FIELDS)
            checkpoint = path.parent / row["adapter_state_path"]
            state_hash = row["adapter_state_sha256"]
            previous = state_artifacts.get(key)
            if previous is not None and previous[1] != state_hash:
                raise ValueError(f"adapter parameters differ for {key}")
            state_artifacts.setdefault(key, (checkpoint, state_hash))
        sidecar = json.loads(natural.adaptation_run_path(path).read_text())
        configurations.append(sidecar["configuration"])
        input_fingerprints.append(sidecar["run_fingerprint"])
    reference = configurations[0]
    ignored_fields = {"target_cv_seeds"}
    for configuration in configurations[1:]:
        mismatches = [
            field for field in set(reference) | set(configuration)
            if field not in ignored_fields and reference.get(field) != configuration.get(field)
        ]
        if mismatches:
            raise ValueError(f"CV input configurations differ: {sorted(mismatches)}")
    target_cv_seeds = sorted({
        int(seed)
        for configuration in configurations
        for seed in configuration["target_cv_seeds"]
    })
    if len(target_cv_seeds) != sum(
        len(configuration["target_cv_seeds"]) for configuration in configurations
    ):
        raise ValueError("CV input configurations repeat a target partition seed")
    merged = merge_rows(rows)
    merged_configuration = {
        **reference,
        "target_cv_seeds": target_cv_seeds,
        "derived_input_run_fingerprints": input_fingerprints,
    }
    canonical = json.dumps(merged_configuration, sort_keys=True, separators=(",", ":"))
    merged_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    for row in merged:
        row["run_fingerprint"] = merged_fingerprint
        key = tuple(str(row[field]) for field in KEY_FIELDS)
        checkpoint, state_hash = state_artifacts[key]
        row["adapter_state_path"] = str(
            Path(os.path.relpath(checkpoint, args.output.parent))
        )
        row["adapter_state_file_sha256"] = sha256_file(checkpoint)
        row["adapter_state_sha256"] = state_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged[0])
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    natural.write_json_atomic(natural.adaptation_run_path(args.output), {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": merged_fingerprint,
        "configuration": merged_configuration,
    })

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "screening_manifest": str(args.manifest),
        "screening_manifest_sha256": sha256_file(args.manifest),
        "inputs": {str(path): sha256_file(path) for path in args.input},
        "input_sidecars": {
            str(natural.adaptation_run_path(path)): sha256_file(
                natural.adaptation_run_path(path)
            )
            for path in args.input
        },
        "output_run_fingerprint": merged_fingerprint,
        "input_rows": len(rows),
        "output_rows": len(merged),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output} rows={len(merged)}")


if __name__ == "__main__":
    main()
