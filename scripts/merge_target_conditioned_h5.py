#!/usr/bin/env python3
"""Merge independently generated E5/H5 seed shards with integrity checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_target_conditioned_e1 as merge_e1  # noqa: E402
import run_target_conditioned_e1 as e1  # noqa: E402
import run_target_conditioned_h5 as h5  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_dirs = [path.resolve() for path in args.shards]
    configs = [merge_e1.read_json(path / "config.json") for path in shard_dirs]
    config = merge_e1.merge_configs(configs)
    manifests = [
        merge_e1.read_json(path / "manifest.json") for path in shard_dirs
    ]
    source_revision = merge_e1.require_common_revision(manifests, "E5/H5")
    rows: list[dict[str, str]] = []
    source_manifests = []
    worker_seconds = []
    for shard_dir, manifest in zip(shard_dirs, manifests):
        manifest_path = shard_dir / "manifest.json"
        if manifest.get("status") != "completed":
            raise ValueError(f"incomplete E5/H5 shard: {shard_dir}")
        rows.extend(merge_e1.read_csv(shard_dir / "raw.csv"))
        worker_seconds.append(float(manifest["elapsed_seconds"]))
        source_manifests.append({
            "directory": shard_dir.name,
            "manifest_sha256": e1.sha256_file(manifest_path),
        })

    row_key = e1.CONFIGURATION_FIELDS + ("strategy_id",)
    merge_e1.require_unique(rows, row_key, "E5/H5")
    configuration_count = len({
        tuple(row[field] for field in e1.CONFIGURATION_FIELDS)
        for row in rows
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
            "merged E5/H5 configuration count is incomplete: "
            f"observed={configuration_count}, expected={expected_configuration_count}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    e1.write_csv(args.out_dir / "raw.csv", rows)
    summary = h5.summarize(rows, configuration_count)
    created_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "experiment": "E5/H5 synthetic sketch certificate sweep (merged)",
        "status": "completed",
        "created_utc": created_utc,
        "config": config,
        "counts": {
            "shards": len(shard_dirs),
            "configurations": configuration_count,
            "rows": len(rows),
        },
        "timing": {
            "sum_worker_seconds": sum(worker_seconds),
            "max_shard_seconds": max(worker_seconds),
        },
        "summary": summary,
        "provenance": {
            "git_revision": source_revision,
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
        for key in (
            "experiment",
            "status",
            "created_utc",
            "config",
            "counts",
            "timing",
            "provenance",
        )
    }
    manifest["outputs"] = [
        "config.json",
        "manifest.json",
        "raw.csv",
        "README.md",
        "summary.json",
    ]
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    gate = summary["h5_gate"]
    readme = f"""# E5/H5 Sketch Certificate Sweep

Status: completed merge of {len(shard_dirs)} independently generated seed shards.

- Seeds: {len(config['seeds'])}
- Configurations: {configuration_count}
- Raw rows: {len(rows)}
- H5 gate: {gate['status']}
- Gate byte denominator: packed symmetric float64 full-Gram

The merge rejects overlapping seeds, mismatched configurations, duplicate rows,
and incomplete Cartesian grids.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "shards": len(shard_dirs),
        "configurations": configuration_count,
        "h5_gate": gate["status"],
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
