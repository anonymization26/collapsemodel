#!/usr/bin/env python3
"""Aggregate source-domain LODO results at the half-removal operating point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
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


def read_lodo_rows(root: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(root.glob("exclude_*/*/*_shortlist_results.csv"))
    if not paths:
        raise ValueError(f"no LODO result files found under {root}")
    rows = []
    for path in paths:
        excluded_domain = path.relative_to(root).parts[0].removeprefix("exclude_")
        with path.open(newline="") as stream:
            file_rows = list(csv.DictReader(stream))
        if not file_rows:
            raise ValueError(f"empty result file: {path}")
        rows.extend({**row, "excluded_domain": excluded_domain} for row in file_rows)
    return rows, paths


def choose_primary_budgets(rows: list[dict[str, str]]) -> dict[str, int]:
    by_domain: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if int(row["replicate"]) == -1:
            by_domain[row["excluded_domain"]][int(row["shortlist_size"])].append(
                float(row["removed_fraction"])
            )
    budgets = {}
    for domain, by_budget in sorted(by_domain.items()):
        candidates = [
            (fmean(removals), budget)
            for budget, removals in by_budget.items()
            if fmean(removals) >= 0.5 - 1e-12
        ]
        if not candidates:
            raise ValueError(f"no >=50% removal operating point for {domain}")
        _, budgets[domain] = min(
            candidates, key=lambda value: (value[0] - 0.5, -value[1])
        )
    return budgets


def primary_rows(
    rows: list[dict[str, str]], budgets: dict[str, int],
) -> list[dict[str, str]]:
    return [
        row for row in rows
        if int(row["shortlist_size"]) == budgets[row["excluded_domain"]]
    ]


def aggregate_group(
    scope: str,
    method: str,
    values: list[dict[str, str]],
    budgets: dict[str, int],
) -> dict[str, object]:
    metrics = cross.METRICS + [
        field for field in cross.OPTIONAL_METRICS
        if all(field in value for value in values)
    ]
    recall = fmean(float(value["best_candidate_recall"]) for value in values)
    removed = fmean(float(value["removed_fraction"]) for value in values)
    domains = sorted({value["excluded_domain"] for value in values})
    return {
        "scope": scope,
        "method": method,
        "primary_shortlist_sizes": "|".join(
            f"{domain}:{budgets[domain]}" for domain in domains
        ),
        "rows": len(values),
        "encoder_target_domain_units": len({
            (value["encoder"], value["target"], value["excluded_domain"])
            for value in values
        }),
        "encoders": len({value["encoder"] for value in values}),
        "targets": len({value["target"] for value in values}),
        "excluded_domains": len(domains),
        **{
            f"{metric}_mean": fmean(float(value[metric]) for value in values)
            for metric in metrics
        },
        "passes_90pct_recall_at_50pct_removal": recall >= 0.9 and removed >= 0.5,
    }


def aggregate_primary(
    rows: list[dict[str, str]], budgets: dict[str, int],
) -> list[dict[str, object]]:
    output = []
    for domain in sorted(budgets):
        domain_rows = [row for row in rows if row["excluded_domain"] == domain]
        for method in sorted({row["method"] for row in domain_rows}):
            values = [row for row in domain_rows if row["method"] == method]
            output.append(aggregate_group(f"exclude_{domain}", method, values, budgets))
    for method in sorted({row["method"] for row in rows}):
        values = [row for row in rows if row["method"] == method]
        output.append(aggregate_group("all_lodo", method, values, budgets))
    return output


def paired_primary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output = []
    for domain in sorted({row["excluded_domain"] for row in rows}):
        domain_rows = [row for row in rows if row["excluded_domain"] == domain]
        for pair in cross.paired_dpp_vs_rank(domain_rows):
            output.append({"excluded_domain": domain, **pair})
    return output


def leave_one_encoder_out(
    rows: list[dict[str, str]], budgets: dict[str, int],
) -> list[dict[str, object]]:
    deterministic = [row for row in rows if int(row["replicate"]) == -1]
    output = []
    for encoder in sorted({row["encoder"] for row in deterministic}):
        retained = [row for row in deterministic if row["encoder"] != encoder]
        for method in sorted({row["method"] for row in retained}):
            values = [row for row in retained if row["method"] == method]
            output.append({
                "excluded_encoder": encoder,
                **aggregate_group("all_lodo", method, values, budgets),
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
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    args = parser.parse_args()

    all_rows, input_paths = read_lodo_rows(args.results_dir)
    budgets = choose_primary_budgets(all_rows)
    selected_rows = primary_rows(all_rows, budgets)
    pairs = paired_primary(selected_rows)
    bootstrap_pairs = [
        {**pair, "shortlist_size": 0} for pair in pairs
    ]
    bootstrap = cross.cluster_bootstrap_summaries(
        bootstrap_pairs,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        cluster_specs=[
            ("target", "target"),
            ("oracle_source_family", "source_family"),
            ("excluded_domain", "excluded_domain"),
        ],
    )
    for row in bootstrap:
        row["operating_point"] = "per_domain_closest_at_least_50pct_removal"

    write_csv(
        args.results_dir / "lodo_primary_summary.csv",
        aggregate_primary(selected_rows, budgets),
    )
    write_csv(args.results_dir / "lodo_dpp_vs_rank_paired.csv", pairs)
    write_csv(args.results_dir / "lodo_cluster_bootstrap_summary.csv", bootstrap)
    write_csv(
        args.results_dir / "lodo_leave_one_encoder_out.csv",
        leave_one_encoder_out(selected_rows, budgets),
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "primary_shortlist_sizes": budgets,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_estimand": "equal cluster mean",
    }
    (args.results_dir / "lodo_aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
