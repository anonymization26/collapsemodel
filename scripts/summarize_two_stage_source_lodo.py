#!/usr/bin/env python3
"""Aggregate leave-one-source-out shortlist results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import summarize_two_stage_cross_encoder as cross


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(root.glob("exclude_*/*/*_shortlist_results.csv"))
    if not paths:
        raise ValueError(f"no leave-one-source result files found under {root}")
    rows = []
    provenance_paths = []
    for path in paths:
        excluded_source = path.relative_to(root).parts[0].removeprefix("exclude_")
        with path.open(newline="") as stream:
            file_rows = list(csv.DictReader(stream))
        if not file_rows:
            raise ValueError(f"empty result file: {path}")
        provenance_paths.append(cross.validate_detail_result(path, file_rows))
        rows.extend({**row, "excluded_source": excluded_source} for row in file_rows)
    return rows, paths + provenance_paths


def aggregate_values(
    scope: str, method: str, values: list[dict[str, str]],
) -> dict[str, object]:
    metrics = cross.METRICS + [
        field for field in cross.OPTIONAL_METRICS
        if all(field in value for value in values)
    ]
    recall = fmean(float(value["best_candidate_recall"]) for value in values)
    removed = fmean(float(value["removed_fraction"]) for value in values)
    return {
        "scope": scope,
        "method": method,
        "shortlist_size": int(values[0]["shortlist_size"]),
        "rows": len(values),
        "encoder_target_exclusion_units": len({
            (value["encoder"], value["target"], value["excluded_source"])
            for value in values
        }),
        "encoders": len({value["encoder"] for value in values}),
        "targets": len({value["target"] for value in values}),
        "excluded_sources": len({value["excluded_source"] for value in values}),
        **{
            f"{metric}_mean": fmean(float(value[metric]) for value in values)
            for metric in metrics
        },
        "passes_90pct_recall_at_50pct_removal": recall >= 0.9 and removed >= 0.5,
    }


def aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for excluded_source in sorted({row["excluded_source"] for row in rows}):
        source_rows = [row for row in rows if row["excluded_source"] == excluded_source]
        for method in sorted({row["method"] for row in source_rows}):
            values = [row for row in source_rows if row["method"] == method]
            output.append(aggregate_values(f"exclude_{excluded_source}", method, values))
    for method in sorted({row["method"] for row in rows}):
        values = [row for row in rows if row["method"] == method]
        output.append(aggregate_values("all_source_lodo", method, values))
    return output


def paired_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for excluded_source in sorted({row["excluded_source"] for row in rows}):
        source_rows = [row for row in rows if row["excluded_source"] == excluded_source]
        for pair in cross.paired_dpp_vs_rank(source_rows):
            output.append({"excluded_source": excluded_source, **pair})
    return output


def leave_one_encoder_out(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    deterministic = [row for row in rows if int(row["replicate"]) == -1]
    output = []
    for encoder in sorted({row["encoder"] for row in deterministic}):
        retained = [row for row in deterministic if row["encoder"] != encoder]
        for method in sorted({row["method"] for row in retained}):
            values = [row for row in retained if row["method"] == method]
            output.append({
                "excluded_encoder": encoder,
                **aggregate_values("all_source_lodo", method, values),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    args = parser.parse_args()

    rows, input_paths = read_rows(args.results_dir)
    selected = [
        row for row in rows if int(row["shortlist_size"]) == args.shortlist_size
    ]
    if not selected:
        raise ValueError(f"no rows for shortlist size {args.shortlist_size}")
    pairs = paired_rows(selected)
    bootstrap = cross.cluster_bootstrap_summaries(
        pairs,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        cluster_specs=[
            ("target", "target"),
            ("oracle_source_family", "source_family"),
            ("excluded_source", "excluded_source"),
        ],
    )
    write_csv(args.results_dir / "source_lodo_summary.csv", aggregate_rows(selected))
    write_csv(args.results_dir / "source_lodo_dpp_vs_rank_paired.csv", pairs)
    write_csv(args.results_dir / "source_lodo_cluster_bootstrap_summary.csv", bootstrap)
    write_csv(
        args.results_dir / "source_lodo_leave_one_encoder_out.csv",
        leave_one_encoder_out(selected),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "shortlist_size": args.shortlist_size,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_estimand": "equal cluster mean",
    }
    (args.results_dir / "source_lodo_aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
