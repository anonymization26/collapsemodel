#!/usr/bin/env python3
"""Merge independently generated E1 seed shards with integrity checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_target_conditioned_e1 as e1  # noqa: E402


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require_unique(
    rows: Iterable[dict[str, str]],
    fields: tuple[str, ...],
    table_name: str,
) -> None:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in seen:
            raise ValueError(f"duplicate {table_name} row for key {key}")
        seen.add(key)


def merge_configs(configs: list[dict[str, object]]) -> dict[str, object]:
    if not configs:
        raise ValueError("at least one shard is required")
    reference = {key: value for key, value in configs[0].items() if key != "seeds"}
    seeds: list[int] = []
    seen_seeds: set[int] = set()
    for config in configs:
        comparable = {key: value for key, value in config.items() if key != "seeds"}
        if comparable != reference:
            raise ValueError("E1 shard configurations differ outside the seed list")
        shard_seeds = config.get("seeds")
        if not isinstance(shard_seeds, list) or not shard_seeds:
            raise ValueError("every E1 shard must contain a nonempty seed list")
        for raw_seed in shard_seeds:
            seed = int(raw_seed)
            if seed in seen_seeds:
                raise ValueError(f"seed {seed} appears in more than one E1 shard")
            seen_seeds.add(seed)
            seeds.append(seed)
    return {"seeds": sorted(seeds), **reference}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_dirs = [path.resolve() for path in args.shards]
    configs = [read_json(path / "config.json") for path in shard_dirs]
    config = merge_configs(configs)

    shared_rows: list[dict[str, str]] = []
    shift_rows: list[dict[str, str]] = []
    source_manifests = []
    worker_seconds = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("status") != "completed":
            raise ValueError(f"incomplete E1 shard: {shard_dir}")
        shared_rows.extend(read_csv(shard_dir / "shared_model_results.csv"))
        shift_rows.extend(read_csv(shard_dir / "conditional_shift_results.csv"))
        worker_seconds.append(float(manifest["elapsed_seconds"]))
        source_manifests.append({
            "directory": str(shard_dir),
            "manifest_sha256": e1.sha256_file(manifest_path),
        })

    shared_key = e1.CONFIGURATION_FIELDS + ("method", "random_repeat")
    shift_key = e1.CONFIGURATION_FIELDS + ("shift", "method")
    require_unique(shared_rows, shared_key, "shared-model")
    require_unique(shift_rows, shift_key, "conditional-shift")

    configuration_count = len({
        tuple(row[field] for field in e1.CONFIGURATION_FIELDS)
        for row in shared_rows
    })
    expected_configuration_count = (
        len(config["seeds"])
        * len(config["dimensions"])
        * len(config["candidate_counts"])
        * len(config["budgets"])
        * len(config["target_samples"])
        * len(config["target_rank_fractions"])
    )
    if configuration_count != expected_configuration_count:
        raise ValueError(
            "merged E1 configuration count is incomplete: "
            f"observed={configuration_count}, expected={expected_configuration_count}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    e1.write_csv(args.out_dir / "shared_model_results.csv", shared_rows)
    e1.write_csv(args.out_dir / "conditional_shift_results.csv", shift_rows)
    summary = e1.summarize(shared_rows, shift_rows)
    created_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "experiment": "E1 target-conditioned synthetic selection (merged)",
        "status": "completed",
        "created_utc": created_utc,
        "config": config,
        "counts": {
            "shards": len(shard_dirs),
            "configurations": configuration_count,
            "shared_model_rows": len(shared_rows),
            "conditional_shift_rows": len(shift_rows),
        },
        "timing": {
            "sum_worker_seconds": sum(worker_seconds),
            "max_shard_seconds": max(worker_seconds),
        },
        "summary": summary,
        "provenance": {
            "git_revision": e1.git_revision(),
            "merge_script_sha256": e1.sha256_file(Path(__file__)),
            "source_manifests": source_manifests,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        key: report[key]
        for key in ("experiment", "status", "created_utc", "config", "counts", "timing", "provenance")
    }
    manifest["outputs"] = [
        "conditional_shift_results.csv",
        "config.json",
        "manifest.json",
        "README.md",
        "shared_model_results.csv",
        "summary.json",
    ]
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    h2 = summary["gates"]["h2_pilot"]
    h5 = summary["gates"]["h5_pilot"]
    readme = f"""# E1 Target-Conditioned Synthetic Selection

Status: completed merge of {len(shard_dirs)} independently generated seed shards.

- Seeds: {len(config['seeds'])}
- Configurations: {configuration_count}
- Shared-model rows: {len(shared_rows)}
- Conditional-shift rows: {len(shift_rows)}
- H2 synthetic gate: {h2['status']}
- Legacy dense-byte H5 pilot gate: {h5['status']}

The merge rejects overlapping seeds, mismatched configurations, duplicate raw
rows, and incomplete Cartesian grids. H5 communication claims must use the
dedicated E5/H5 report, which also compares against packed symmetric Grams.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "shards": len(shard_dirs),
        "configurations": configuration_count,
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
