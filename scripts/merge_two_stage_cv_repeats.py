#!/usr/bin/env python3
"""Merge repeated target-CV utilities without duplicating adapter seeds."""

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
        partition_seeds = sorted({value["target_cv_seed"] for value in values})
        if len(partition_seeds) != len(values):
            raise ValueError(f"duplicate CV partition for key {key}")
        if expected_partitions is None:
            expected_partitions = partition_seeds
        elif partition_seeds != expected_partitions:
            raise ValueError(f"incomplete CV partitions for key {key}")

        test_values = [float(value["test_accuracy"]) for value in values]
        if max(test_values) - min(test_values) > 1e-8:
            raise ValueError(f"test accuracy changed across CV partitions for key {key}")
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
        output["test_accuracy"] = fmean(test_values)
        for field in MEAN_FIELDS:
            output[field] = fmean(float(value[field]) for value in values)
        output["cv_partition_count"] = len(partition_seeds)
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

    rows = []
    for path in args.input:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    merged = merge_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(merged[0])
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "screening_manifest": str(args.manifest),
        "screening_manifest_sha256": sha256_file(args.manifest),
        "inputs": {str(path): sha256_file(path) for path in args.input},
        "input_rows": len(rows),
        "output_rows": len(merged),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.output} rows={len(merged)}")


if __name__ == "__main__":
    main()
