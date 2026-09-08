#!/usr/bin/env python3
"""Measure Stage-2 utility-oracle stability across target train/validation splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

import numpy as np


SOURCE_FAMILY_ALIASES = {
    "cifar100": "cifar100_shared_images",
    "cifar100_coarse": "cifar100_shared_images",
    "mnist": "handwritten_digits",
    "usps": "handwritten_digits",
    "organamnist": "organmnist_views",
    "organcmnist": "organmnist_views",
    "organsmnist": "organmnist_views",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_seed_utility(directory: Path) -> tuple[dict[tuple[str, str, str], float], list[Path]]:
    paths = sorted(directory.glob("*/adaptation_results.csv"))
    if not paths:
        raise ValueError(f"no adaptation results found under {directory}")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                grouped[(row["encoder"], row["target"], row["source"])].append(
                    float(row["validation_accuracy"])
                )
    seed_counts = {len(values) for values in grouped.values()}
    if len(seed_counts) != 1:
        raise ValueError(f"unequal adapter seed counts in {directory}: {seed_counts}")
    return {key: fmean(values) for key, values in grouped.items()}, paths


def read_joint_cv_utility(
    directory: Path,
) -> tuple[dict[str, dict[tuple[str, str, str], float]], list[Path]]:
    """Read per-partition utility from adapters trained once in a joint CV run."""
    paths = sorted(directory.glob("*/adaptation_results.csv"))
    if not paths:
        raise ValueError(f"no adaptation results found under {directory}")
    grouped: dict[str, dict[tuple[str, str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    expected_partition_seeds: list[str] | None = None
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                partition_seeds = row["target_cv_seed"].split("|")
                partition_values = row["validation_partition_accuracies"].split("|")
                if len(partition_seeds) != len(partition_values):
                    raise ValueError(
                        f"partition seed/value mismatch in {path}: "
                        f"{len(partition_seeds)} != {len(partition_values)}"
                    )
                if len(partition_seeds) < 2:
                    raise ValueError(f"joint CV requires at least two partitions: {path}")
                if int(row["target_cv_partitions"]) != len(partition_seeds):
                    raise ValueError(f"incorrect target_cv_partitions in {path}")
                if expected_partition_seeds is None:
                    expected_partition_seeds = partition_seeds
                elif partition_seeds != expected_partition_seeds:
                    raise ValueError(f"inconsistent CV partition seeds in {path}")
                key = (row["encoder"], row["target"], row["source"])
                for partition_seed, value in zip(partition_seeds, partition_values):
                    grouped[partition_seed][key].append(float(value))

    adapter_seed_counts = {
        len(values)
        for utility in grouped.values()
        for values in utility.values()
    }
    if len(adapter_seed_counts) != 1:
        raise ValueError(f"unequal adapter seed counts in {directory}: {adapter_seed_counts}")
    utility_by_partition = {
        partition_seed: {
            key: fmean(values) for key, values in utility.items()
        }
        for partition_seed, utility in grouped.items()
    }
    return utility_by_partition, paths


def parse_seed_directory(specification: str) -> tuple[str, Path]:
    split_seed, separator, directory = specification.partition("=")
    if not separator or not split_seed or not directory:
        raise ValueError(
            f"invalid seed directory '{specification}'; expected SEED=PATH"
        )
    return split_seed, Path(directory)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    ranks_a = average_ranks(values_a)
    ranks_b = average_ranks(values_b)
    centered_a = ranks_a - ranks_a.mean()
    centered_b = ranks_b - ranks_b.mean()
    denominator = float(np.linalg.norm(centered_a) * np.linalg.norm(centered_b))
    if denominator == 0.0:
        return float("nan")
    return float(centered_a @ centered_b / denominator)


def analyze_utility(
    utility_by_split: dict[str, dict[tuple[str, str, str], float]],
    tie_tolerance: float,
    source_families: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_families = source_families or {}

    def family(source: str) -> str:
        return source_families.get(source, source)

    split_seeds = sorted(utility_by_split)
    reference_keys = set(utility_by_split[split_seeds[0]])
    for split_seed, utility in utility_by_split.items():
        if set(utility) != reference_keys:
            missing = len(reference_keys - set(utility))
            extra = len(set(utility) - reference_keys)
            raise ValueError(
                f"utility keys differ for split {split_seed}: missing={missing}, extra={extra}"
            )

    grouped_keys: dict[tuple[str, str], list[str]] = defaultdict(list)
    for encoder, target, source in reference_keys:
        grouped_keys[(encoder, target)].append(source)

    rows = []
    all_correlations = []
    all_margins = []
    all_ranges = []
    all_source_split_std = []
    for (encoder, target), source_names in sorted(grouped_keys.items()):
        sources = sorted(source_names)
        vectors = {
            split_seed: np.asarray([
                utility_by_split[split_seed][(encoder, target, source)]
                for source in sources
            ])
            for split_seed in split_seeds
        }
        correlations = [
            spearman(vectors[first], vectors[second])
            for first, second in itertools.combinations(split_seeds, 2)
        ]
        margins = []
        ranges = []
        primary_oracles = []
        oracle_tiers = []
        for split_seed in split_seeds:
            values = vectors[split_seed]
            ranking = sorted(
                range(len(sources)), key=lambda index: (-values[index], sources[index]),
            )
            best = float(values[ranking[0]])
            second = float(values[ranking[1]])
            primary_oracles.append(sources[ranking[0]])
            oracle_tiers.append({
                sources[index]
                for index in ranking
                if float(values[index]) >= best - tie_tolerance
            })
            margins.append(best - second)
            ranges.append(best - float(values[ranking[-1]]))
        shared_oracles = set.intersection(*oracle_tiers)
        primary_oracle_families = [family(source) for source in primary_oracles]
        oracle_family_tiers = [
            {family(source) for source in tier} for tier in oracle_tiers
        ]
        shared_oracle_families = set.intersection(*oracle_family_tiers)
        source_split_std = [
            float(np.std([
                utility_by_split[split_seed][(encoder, target, source)]
                for split_seed in split_seeds
            ]))
            for source in sources
        ]
        finite_correlations = [value for value in correlations if np.isfinite(value)]
        row = {
            "encoder": encoder,
            "target": target,
            "candidate_sources": len(sources),
            "primary_oracles": "|".join(
                f"{seed}:{source}" for seed, source in zip(split_seeds, primary_oracles)
            ),
            "distinct_primary_oracles": len(set(primary_oracles)),
            "shared_tie_optimal_sources": "|".join(sorted(shared_oracles)),
            "has_shared_tie_optimal_source": bool(shared_oracles),
            "primary_oracle_families": "|".join(
                f"{seed}:{oracle_family}"
                for seed, oracle_family in zip(split_seeds, primary_oracle_families)
            ),
            "distinct_primary_oracle_families": len(set(primary_oracle_families)),
            "shared_tie_optimal_families": "|".join(sorted(shared_oracle_families)),
            "has_shared_tie_optimal_family": bool(shared_oracle_families),
            "pairwise_spearman_mean": (
                fmean(finite_correlations) if finite_correlations else float("nan")
            ),
            "pairwise_spearman_min": (
                min(finite_correlations) if finite_correlations else float("nan")
            ),
            "top1_margin_mean": fmean(margins),
            "top1_margin_min": min(margins),
            "utility_range_mean": fmean(ranges),
            "source_split_std_mean": fmean(source_split_std),
        }
        rows.append(row)
        all_correlations.extend(finite_correlations)
        all_margins.extend(margins)
        all_ranges.extend(ranges)
        all_source_split_std.extend(source_split_std)

    summary = {
        "split_seeds": split_seeds,
        "encoder_target_pairs": len(rows),
        "stable_primary_oracle_pairs": sum(
            int(row["distinct_primary_oracles"]) == 1 for row in rows
        ),
        "shared_tie_optimal_source_pairs": sum(
            bool(row["has_shared_tie_optimal_source"]) for row in rows
        ),
        "stable_primary_oracle_family_pairs": sum(
            int(row["distinct_primary_oracle_families"]) == 1 for row in rows
        ),
        "shared_tie_optimal_family_pairs": sum(
            bool(row["has_shared_tie_optimal_family"]) for row in rows
        ),
        "pairwise_spearman_mean": fmean(all_correlations),
        "pairwise_spearman_median": median(all_correlations),
        "pairwise_spearman_min": min(all_correlations),
        "top1_margin_mean": fmean(all_margins),
        "top1_margin_median": median(all_margins),
        "top1_margin_at_most_1e_3": sum(value <= 1e-3 for value in all_margins),
        "top1_margin_at_most_5e_3": sum(value <= 5e-3 for value in all_margins),
        "top1_margin_count": len(all_margins),
        "utility_range_mean": fmean(all_ranges),
        "utility_range_median": median(all_ranges),
        "source_split_std_mean": fmean(all_source_split_std),
        "source_split_std_median": median(all_source_split_std),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--baseline-dir", type=Path)
    input_group.add_argument("--joint-cv-dir", type=Path)
    parser.add_argument("--baseline-seed", default="20260905")
    parser.add_argument("--sensitivity-dir", type=Path)
    parser.add_argument(
        "--additional-dir", action="append", default=[], metavar="SEED=PATH",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    if args.joint_cv_dir is not None:
        if args.sensitivity_dir is not None or args.additional_dir:
            parser.error("--joint-cv-dir cannot be combined with split directories")
        utility_by_split, input_paths = read_joint_cv_utility(args.joint_cv_dir)
        input_protocol = "same_adapter_repeated_stratified_cv"
    else:
        utility_by_split = {}
        input_paths = []
        baseline, baseline_paths = read_seed_utility(args.baseline_dir)
        utility_by_split[args.baseline_seed] = baseline
        input_paths.extend(baseline_paths)
        seed_directories = []
        if args.sensitivity_dir is not None:
            seed_directories.extend(
                (split_dir.name.removeprefix("split_"), split_dir)
                for split_dir in sorted(args.sensitivity_dir.glob("split_*"))
            )
        seed_directories.extend(
            parse_seed_directory(specification)
            for specification in args.additional_dir
        )
        for split_seed, split_dir in seed_directories:
            if split_seed in utility_by_split:
                raise ValueError(f"duplicate split seed: {split_seed}")
            utility, paths = read_seed_utility(split_dir)
            utility_by_split[split_seed] = utility
            input_paths.extend(paths)
        input_protocol = "independently_trained_split_runs"
    if len(utility_by_split) < 2:
        raise ValueError("at least two target split seeds are required")

    rows, summary = analyze_utility(
        utility_by_split, args.tie_tolerance, SOURCE_FAMILY_ALIASES,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "oracle_stability.csv"
    with detail_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    summary["input_protocol"] = input_protocol
    summary["script_sha256"] = sha256_file(Path(__file__))
    summary["input_sha256"] = {str(path): sha256_file(path) for path in input_paths}
    (args.out_dir / "oracle_stability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
