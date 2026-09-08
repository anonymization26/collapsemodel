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
    "test_regret_vs_exhaustive_validation_selection",
    "removed_fraction",
]
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


def read_detail_rows(results_dir: Path) -> tuple[list[dict[str, str]], list[Path]]:
    paths = sorted(results_dir.glob("*/*_shortlist_results.csv"))
    if not paths:
        raise ValueError(f"no per-encoder shortlist results found under {results_dir}")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as stream:
            file_rows = list(csv.DictReader(stream))
        if not file_rows:
            raise ValueError(f"empty result file: {path}")
        rows.extend(file_rows)
    return rows, paths


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
            field for field in OPTIONAL_METRICS
            if all(field in value for value in values)
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
            "dpp_minus_rank_test_regret": (
                float(dpp["test_regret_vs_exhaustive_validation_selection"])
                - float(rank["test_regret_vs_exhaustive_validation_selection"])
            ),
            "shortlist_jaccard": len(dpp_shortlist & rank_shortlist)
            / len(dpp_shortlist | rank_shortlist),
        }
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
            field for field in OPTIONAL_METRICS
            if all(field in value for value in values)
        ]
        output.append({
            "method": method,
            "shortlist_size": shortlist_size,
            "target": target,
            "encoders": len({value["encoder"] for value in values}),
            **{f"{field}_mean": mean_metric(values, field) for field in metrics},
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

    rows, input_paths = read_detail_rows(args.results_dir)
    aggregate = aggregate_rows(rows)
    pairs = paired_dpp_vs_rank(rows)
    pair_summary = summarize_pairs(pairs)
    bootstrap_summary = cluster_bootstrap_summaries(
        pairs, args.bootstrap_replicates, args.bootstrap_seed,
    )
    write_csv(args.results_dir / "cross_encoder_summary.csv", aggregate)
    write_csv(args.results_dir / "dpp_vs_rank_paired.csv", pairs)
    write_csv(args.results_dir / "dpp_vs_rank_summary.csv", pair_summary)
    write_csv(args.results_dir / "cluster_bootstrap_summary.csv", bootstrap_summary)
    write_csv(args.results_dir / "leave_one_encoder_out.csv", leave_one_encoder_out(rows))
    write_csv(args.results_dir / "target_summary.csv", target_summaries(rows))
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
    }
    (args.results_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
