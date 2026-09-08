#!/usr/bin/env python3
"""Merge and validate sharded classical-baseline screening results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, str]] = []
    result_fields: list[str] | None = None
    inventories: dict[str, dict[str, str]] = {}
    manifests = []

    for shard in range(args.shard_count):
        result_path = args.input_dir / f"{args.prefix}_results_shard{shard}.csv"
        inventory_path = args.input_dir / f"method_resource_inventory_shard{shard}.csv"
        manifest_path = args.input_dir / f"{args.prefix}_manifest_shard{shard}.json"
        fields, rows = read_csv(result_path)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"result columns differ in shard {shard}")
        result_rows.extend(rows)
        _, inventory_rows = read_csv(inventory_path)
        for row in inventory_rows:
            method = row["method"]
            if method in inventories:
                stable_fields = [
                    "information_class", "representation", "uses_labels",
                    "uses_family_metadata", "uses_target_data", "bytes_per_pool",
                    "bytes_for_collection",
                ]
                if any(inventories[method][field] != row[field] for field in stable_fields):
                    raise ValueError(f"resource metadata differs for {method} in shard {shard}")
            else:
                inventories[method] = row
        manifests.append(json.loads(manifest_path.read_text()))

    keys = [
        (row["config"], row["method"], int(row["replicate"]))
        for row in result_rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate config-method-replicate rows found")
    configs = sorted({row["config"] for row in result_rows})
    if args.expected_configs is not None and len(configs) != args.expected_configs:
        raise ValueError(f"expected {args.expected_configs} configs, found {len(configs)}")
    counts = defaultdict(int)
    for row in result_rows:
        counts[row["config"]] += 1
    count_values = set(counts.values())
    if len(count_values) != 1:
        raise ValueError(f"config row counts differ: {dict(sorted(counts.items()))}")
    rows_per_config = next(iter(count_values))
    if args.expected_rows_per_config is not None and rows_per_config != args.expected_rows_per_config:
        raise ValueError(
            f"expected {args.expected_rows_per_config} rows/config, found {rows_per_config}"
        )

    result_rows.sort(key=lambda row: (row["config"], row["method"], int(row["replicate"])))
    assert result_fields is not None
    write_csv(args.out_dir / f"{args.prefix}_results.csv", result_fields, result_rows)

    merged_configs: dict[str, object] = {}
    script_hashes = {manifest["script_sha256"] for manifest in manifests}
    if len(script_hashes) != 1:
        raise ValueError(f"shards used different scripts: {sorted(script_hashes)}")
    for manifest in manifests:
        overlap = set(merged_configs).intersection(manifest["configs"])
        if overlap:
            raise ValueError(f"manifest configs repeated across shards: {sorted(overlap)}")
        merged_configs.update(manifest["configs"])
    if set(merged_configs) != set(configs):
        raise ValueError("manifest and result config sets differ")
    merged_manifest = {
        key: value
        for key, value in manifests[0].items()
        if key not in {"start_time_utc", "end_time_utc", "pid", "script_sha256", "shard_index", "configs"}
    }
    merged_manifest.update({
        "start_time_utc": min(manifest["start_time_utc"] for manifest in manifests),
        "end_time_utc": max(manifest["end_time_utc"] for manifest in manifests),
        "script_sha256": manifests[0]["script_sha256"],
        "source_shards": [
            {
                "shard_index": manifest["shard_index"],
                "pid": manifest["pid"],
                "start_time_utc": manifest["start_time_utc"],
                "end_time_utc": manifest["end_time_utc"],
            }
            for manifest in manifests
        ],
        "configs": dict(sorted(merged_configs.items())),
    })
    (args.out_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(merged_manifest, indent=2) + "\n"
    )

    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_by_config_method: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        rows_by_method[row["method"]].append(row)
        rows_by_config_method[(row["config"], row["method"])].append(row)

    inventory_rows = []
    for method in sorted(rows_by_method):
        method_rows = rows_by_method[method]
        metadata = inventories[method]
        information_class = metadata["information_class"]
        requires_selected_feedback = method == "collapse"
        feedback_bytes_per_selected_pool = 0
        feedback_bytes_mean = 0.0
        if requires_selected_feedback:
            top_k = int(manifests[0]["top_k"])
            summary_bytes_per_pool = int(metadata["bytes_per_pool"])
            dimension = (summary_bytes_per_pool // 4 - 2) // top_k
            pool_size = int(manifests[0]["pool_size"])
            feedback_bytes_per_selected_pool = 4 * pool_size * dimension
            feedback_bytes_mean = feedback_bytes_per_selected_pool * float(np.mean([
                int(row["budget"]) for row in method_rows
            ]))
            information_class = "summary+selected-full-feedback"
        selection_seconds = np.asarray(
            [float(row["selection_seconds"]) for row in method_rows], dtype=np.float64,
        )
        metric_seconds = np.asarray(
            [float(row["metric_seconds"]) for row in method_rows], dtype=np.float64,
        )
        preprocessing_seconds = np.asarray(
            [float(row["preprocessing_seconds"]) for row in method_rows], dtype=np.float64,
        )
        inventory_rows.append({
            "method": method,
            "information_class": information_class,
            "representation": metadata["representation"],
            "uses_labels": metadata["uses_labels"],
            "uses_family_metadata": metadata["uses_family_metadata"],
            "uses_target_data": metadata["uses_target_data"],
            "bytes_per_pool": metadata["bytes_per_pool"],
            "bytes_for_collection": metadata["bytes_for_collection"],
            "requires_selected_full_feature_feedback": requires_selected_feedback,
            "feedback_bytes_per_selected_pool": feedback_bytes_per_selected_pool,
            "feedback_bytes_mean": feedback_bytes_mean,
            "selector_total_bytes_mean": int(metadata["bytes_for_collection"]) + feedback_bytes_mean,
            "selection_runs": len({row["config"] for row in method_rows}),
            "metric_runs": len(method_rows),
            "selection_seconds_mean": float(selection_seconds.mean()),
            "selection_seconds_max": float(selection_seconds.max()),
            "shared_all_representation_preprocessing_seconds_mean": float(
                preprocessing_seconds.mean()
            ),
            "end_to_end_upper_seconds_mean": float(
                preprocessing_seconds.mean() + selection_seconds.mean()
            ),
            "metric_seconds_mean": float(metric_seconds.mean()),
        })
    inventory_fields = list(inventory_rows[0])
    write_csv(args.out_dir / "method_resource_inventory.csv", inventory_fields, inventory_rows)

    available_methods = set(rows_by_method)
    reference_method = next(
        (
            method for method in [
                "exhaustive_merged_rank_oracle",
                "exact_merged_rank_greedy",
                "full_merged_rank",
            ]
            if method in available_methods
        ),
        None,
    )
    if reference_method is None:
        raise ValueError("no exact-score or exhaustive merged-rank reference is available")
    reference_by_config = {
        config: float(rows[0]["merged_reff"])
        for (config, method), rows in rows_by_config_method.items()
        if method == reference_method
    }
    rank_by_config = {
        config: float(rows[0]["merged_reff"])
        for (config, method), rows in rows_by_config_method.items()
        if method == "rank_only"
    }
    if set(reference_by_config) != set(configs) or set(rank_by_config) != set(configs):
        raise ValueError(f"{reference_method} or rank_only is missing from some configs")

    summary_rows = []
    for method in sorted(rows_by_method):
        config_values = []
        config_duplicates = []
        gaps = []
        wins_vs_rank = []
        for config in configs:
            method_rows = rows_by_config_method[(config, method)]
            values = np.asarray([float(row["merged_reff"]) for row in method_rows])
            duplicates = np.asarray([float(row["duplicate_count"]) for row in method_rows])
            value = float(values.mean())
            config_values.append(value)
            config_duplicates.append(float(duplicates.mean()))
            gaps.append(reference_by_config[config] - value)
            wins_vs_rank.append(value > rank_by_config[config] + 1e-10)
        summary_rows.append({
            "method": method,
            "configs": len(configs),
            "selections": len(rows_by_method[method]),
            "merged_reff_mean": float(np.mean(config_values)),
            "merged_reff_config_std": float(np.std(config_values, ddof=1)),
            "duplicate_count_mean": float(np.mean(config_duplicates)),
            "reference_method": reference_method,
            "gap_vs_reference_mean": float(np.mean(gaps)),
            "relative_gap_vs_reference_mean": float(np.mean([
                gap / max(reference_by_config[config], 1e-12)
                for config, gap in zip(configs, gaps)
            ])),
            "config_win_rate_vs_rank_only": float(np.mean(wins_vs_rank)),
        })
    summary_rows.sort(key=lambda row: (-float(row["merged_reff_mean"]), str(row["method"])))
    write_csv(
        args.out_dir / f"{args.prefix}_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )
    print(
        f"merged {args.shard_count} shards: configs={len(configs)} "
        f"rows={len(result_rows)} rows_per_config={rows_per_config}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="classic_screening")
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-configs", type=int, default=24)
    parser.add_argument("--expected-rows-per-config", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
