#!/usr/bin/env python3
"""Cross-encoder controlled-overlap sweep for the Stage-1 stopping rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

import two_stage_classic_baselines as classic


DATASETS = ("cifar10", "cifar100", "dtd", "eurosat", "svhn")
DEFAULT_SEEDS = tuple(range(20260910, 20260920))
DEFAULT_OVERLAPS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
DEFAULT_BUDGETS = (3, 5, 8, 10)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_set(
    root: Path, encoder: str, dataset: str, sample_tag: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    stem = f"{encoder}__{dataset}__n{sample_tag}"
    npz_path = root / f"{stem}.npz"
    metadata_path = root / f"{stem}.json"
    metadata = json.loads(metadata_path.read_text())
    if sha256_file(npz_path) != metadata["feature_file_sha256"]:
        raise ValueError(f"feature checksum mismatch: {npz_path}")
    with np.load(npz_path) as payload:
        features = payload["H"].astype(np.float32)
        indices = payload["indices"].astype(np.int64)
    if list(features.shape) != metadata["feature_shape"] or len(indices) != len(features):
        raise ValueError(f"feature metadata mismatch: {npz_path}")
    return classic.normalize_rows(features), indices, metadata


def make_collection(
    feature_sets: dict[str, np.ndarray],
    source_indices: dict[str, np.ndarray],
    pool_size: int,
    pools_per_dataset: int,
    overlap: float,
    seed: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Build disjoint base pools and one controlled alias per dataset."""

    pools: dict[str, np.ndarray] = {}
    pool_indices: dict[str, np.ndarray] = {}
    dataset_family: dict[str, str] = {}
    lineage: dict[str, str] = {}
    alias_parent: dict[str, str] = {}
    required = (pools_per_dataset + 1) * pool_size

    for dataset_offset, dataset in enumerate(DATASETS):
        features = feature_sets[dataset]
        if len(features) < required:
            raise ValueError(
                f"{dataset} has {len(features)} rows; need at least {required}"
            )
        rng = np.random.default_rng(seed + dataset_offset * 10_000)
        permutation = rng.permutation(len(features))
        base_positions: dict[str, np.ndarray] = {}
        for pool_index in range(pools_per_dataset):
            name = f"{dataset}__pool{pool_index}"
            start = pool_index * pool_size
            positions = permutation[start : start + pool_size]
            base_positions[name] = positions
            pools[name] = features[positions]
            pool_indices[name] = source_indices[dataset][positions]
            dataset_family[name] = dataset
            lineage[name] = name

        parent_index = (seed + dataset_offset) % pools_per_dataset
        parent = f"{dataset}__pool{parent_index}"
        parent_positions = base_positions[parent]
        fresh_positions = permutation[
            pools_per_dataset * pool_size : (pools_per_dataset + 1) * pool_size
        ]
        shared_count = int(round(pool_size * overlap))
        shared = (
            rng.choice(parent_positions, shared_count, replace=False)
            if shared_count
            else np.empty(0, dtype=np.int64)
        )
        fresh_count = pool_size - shared_count
        fresh = (
            rng.choice(fresh_positions, fresh_count, replace=False)
            if fresh_count
            else np.empty(0, dtype=np.int64)
        )
        alias_positions = np.concatenate([shared, fresh])
        rng.shuffle(alias_positions)
        alias = f"{dataset}__alias"
        pools[alias] = features[alias_positions]
        pool_indices[alias] = source_indices[dataset][alias_positions]
        dataset_family[alias] = dataset
        lineage[alias] = parent
        alias_parent[alias] = parent

    return pools, pool_indices, dataset_family, lineage, alias_parent


def deterministic_argmax(candidates: Iterable[str], score) -> str:
    return max(sorted(candidates), key=score)


def collapse_sequence(
    features: dict[str, np.ndarray],
    ranks: dict[str, float],
    nuclear: dict[str, float],
    subspaces: dict[str, np.ndarray],
    top_k: int,
    max_budget: int,
) -> list[str]:
    selected = [deterministic_argmax(features, lambda name: ranks[name])]
    current = features[selected[0]]
    while len(selected) < max_budget:
        current_stats = classic.pool_statistics({"current": current}, top_k)
        current_rank = current_stats[0]["current"]
        current_nuclear = current_stats[1]["current"]
        current_subspace = current_stats[3]["current"]

        def score(name: str) -> float:
            cosines = np.linalg.svd(
                current_subspace @ subspaces[name].T, compute_uv=False,
            )
            alpha = float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))
            gamma = nuclear[name] / max(current_nuclear, 1e-12)
            return classic.collapse_predict(
                current_rank, ranks[name], gamma, alpha,
            )

        remaining = [name for name in features if name not in selected]
        choice = deterministic_argmax(remaining, score)
        selected.append(choice)
        current = np.concatenate([current, features[choice]], axis=0)
    return selected


def lineage_deduplicated_rank_sequence(
    ranks: dict[str, float], lineage: dict[str, str], max_budget: int,
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for name in sorted(ranks, key=lambda item: (-ranks[item], item)):
        if lineage[name] in seen:
            continue
        selected.append(name)
        seen.add(lineage[name])
        if len(selected) == max_budget:
            return selected
    raise ValueError("not enough independent lineages for requested budget")


lineage_oracle_sequence = lineage_deduplicated_rank_sequence


def full_rank_sequence(
    features: dict[str, np.ndarray], max_budget: int,
) -> list[str]:
    return classic.full_merged_rank_greedy(features, max_budget)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return float("nan")
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def alignment_detection(
    similarity: np.ndarray, names: list[str], lineage: dict[str, str],
) -> tuple[float, float]:
    """Global lineage detection; cross-dataset negatives make this an easy diagnostic."""

    labels: list[int] = []
    scores: list[float] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            labels.append(int(lineage[names[left]] == lineage[names[right]]))
            scores.append(float(similarity[left, right]))
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    return binary_auc(label_array, score_array), average_precision(label_array, score_array)


def matched_parent_detection(
    similarity: np.ndarray,
    names: list[str],
    dataset_family: dict[str, str],
    alias_parent: dict[str, str],
) -> dict[str, float]:
    """Evaluate alias-parent recovery only against same-dataset base pools."""

    name_to_index = {name: index for index, name in enumerate(names)}
    labels: list[int] = []
    scores: list[float] = []
    reciprocal_ranks: list[float] = []
    top_one: list[float] = []
    for alias, parent in sorted(alias_parent.items()):
        candidates = [
            name for name in names
            if name != alias
            and name not in alias_parent
            and dataset_family[name] == dataset_family[alias]
        ]
        if parent not in candidates:
            raise ValueError(f"parent {parent} is absent from matched candidates for {alias}")
        ranked = sorted(
            candidates,
            key=lambda name: (-float(similarity[name_to_index[alias], name_to_index[name]]), name),
        )
        parent_rank = ranked.index(parent) + 1
        reciprocal_ranks.append(1.0 / parent_rank)
        top_one.append(float(parent_rank == 1))
        for candidate in candidates:
            labels.append(int(candidate == parent))
            scores.append(float(similarity[name_to_index[alias], name_to_index[candidate]]))
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "parent_matched_auroc": binary_auc(label_array, score_array),
        "parent_matched_auprc": average_precision(label_array, score_array),
        "parent_top1_accuracy": float(np.mean(top_one)) if top_one else float("nan"),
        "parent_mean_reciprocal_rank": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
        ),
    }


def selection_metrics(
    selected: list[str],
    features: dict[str, np.ndarray],
    pool_indices: dict[str, np.ndarray],
    dataset_family: dict[str, str],
    lineage: dict[str, str],
    alias_parent: dict[str, str],
) -> dict[str, float | int]:
    merged = np.concatenate([features[name] for name in selected], axis=0)
    parent_pairs = sum(
        int(alias in selected and parent in selected)
        for alias, parent in alias_parent.items()
    )
    overlap_fractions: list[float] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if dataset_family[left] != dataset_family[right]:
                overlap_fractions.append(0.0)
                continue
            intersection = np.intersect1d(
                pool_indices[left], pool_indices[right], assume_unique=True,
            ).size
            overlap_fractions.append(intersection / len(pool_indices[left]))
    return {
        "merged_reff": classic.effective_rank(merged),
        "unique_lineages": len({lineage[name] for name in selected}),
        "lineage_duplicate_count": len(selected) - len({lineage[name] for name in selected}),
        "unique_datasets": len({dataset_family[name] for name in selected}),
        "selected_alias_parent_pairs": parent_pairs,
        "max_selected_sample_overlap": max(overlap_fractions, default=0.0),
        "mean_selected_sample_overlap": float(np.mean(overlap_fractions)) if overlap_fractions else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_collection(
    args: argparse.Namespace,
    feature_sets: dict[str, np.ndarray],
    source_indices: dict[str, np.ndarray],
    seed: int,
    overlap: float,
) -> list[dict[str, object]]:
    collection_started = time.perf_counter()
    pools, pool_indices, dataset_family, lineage, alias_parent = make_collection(
        feature_sets,
        source_indices,
        args.pool_size,
        args.pools_per_dataset,
        overlap,
        seed,
    )
    names = sorted(pools)
    stats_started = time.perf_counter()
    ranks, nuclear, _, subspaces = classic.pool_statistics(pools, args.top_k)
    similarity = classic.similarity_matrix(
        pools, names, "subspace", {}, subspaces,
    )
    stats_seconds = time.perf_counter() - stats_started
    designated_parent_alignment = matched_parent_detection(
        similarity, names, dataset_family, alias_parent,
    )
    if overlap > 0.0:
        global_alignment_auroc, global_alignment_auprc = alignment_detection(
            similarity, names, lineage,
        )
        matched_alignment = designated_parent_alignment
    else:
        global_alignment_auroc = global_alignment_auprc = float("nan")
        matched_alignment = {key: float("nan") for key in designated_parent_alignment}
    max_budget = max(args.budgets)

    sequences: dict[str, tuple[list[str], float]] = {}

    def timed(method: str, function) -> None:
        started = time.perf_counter()
        sequences[method] = function(), time.perf_counter() - started

    timed(
        "rank_only",
        lambda: sorted(ranks, key=lambda name: (-ranks[name], name))[:max_budget],
    )
    timed(
        "collapse",
        lambda: collapse_sequence(
            pools, ranks, nuclear, subspaces, args.top_k, max_budget,
        ),
    )
    timed(
        "dpp_subspace",
        lambda: classic.dpp_greedy(similarity, names, ranks, max_budget),
    )
    timed(
        "facility_subspace",
        lambda: classic.facility_location(similarity, names, max_budget),
    )
    timed(
        "lineage_deduplicated_rank",
        lambda: lineage_deduplicated_rank_sequence(ranks, lineage, max_budget),
    )
    if args.include_full_rank:
        timed("exact_merged_rank_greedy", lambda: full_rank_sequence(pools, max_budget))

    rank_sequence = sequences["rank_only"][0]
    collapse_order = sequences["collapse"][0]
    rows: list[dict[str, object]] = []
    metrics_by_key: dict[tuple[str, int], dict[str, float | int]] = {}
    for method, (sequence, selection_seconds) in sequences.items():
        for budget in args.budgets:
            selected = sequence[:budget]
            metric_started = time.perf_counter()
            metrics = selection_metrics(
                selected,
                pools,
                pool_indices,
                dataset_family,
                lineage,
                alias_parent,
            )
            metric_seconds = time.perf_counter() - metric_started
            metrics_by_key[(method, budget)] = metrics
            rows.append({
                "encoder": args.encoder,
                "construction_seed": seed,
                "overlap": overlap,
                "budget": budget,
                "method": method,
                "replicate": 0,
                "selected": "|".join(selected),
                **metrics,
                "selection_seconds_for_max_budget": selection_seconds,
                "metric_seconds": metric_seconds,
                "shared_statistics_seconds": stats_seconds,
                "alignment_global_lineage_auroc": global_alignment_auroc,
                "alignment_global_lineage_auprc": global_alignment_auprc,
                "alignment_parent_matched_auroc": matched_alignment["parent_matched_auroc"],
                "alignment_parent_matched_auprc": matched_alignment["parent_matched_auprc"],
                "alignment_parent_top1_accuracy": matched_alignment["parent_top1_accuracy"],
                "alignment_parent_mean_reciprocal_rank": matched_alignment[
                    "parent_mean_reciprocal_rank"
                ],
                "alignment_parent_ground_truth_defined": overlap > 0.0,
                "alignment_zero_overlap_designated_parent_top1_accuracy": (
                    designated_parent_alignment["parent_top1_accuracy"]
                    if overlap == 0.0 else float("nan")
                ),
                "collapse_differs_from_rank": collapse_order[:budget] != rank_sequence[:budget],
            })

    rng = np.random.default_rng(seed + int(round(overlap * 1000)) + 991_337)
    for replicate in range(args.n_random):
        order = rng.permutation(names).tolist()
        for budget in args.budgets:
            selected = order[:budget]
            metric_started = time.perf_counter()
            metrics = selection_metrics(
                selected,
                pools,
                pool_indices,
                dataset_family,
                lineage,
                alias_parent,
            )
            rows.append({
                "encoder": args.encoder,
                "construction_seed": seed,
                "overlap": overlap,
                "budget": budget,
                "method": "random",
                "replicate": replicate,
                "selected": "|".join(selected),
                **metrics,
                "selection_seconds_for_max_budget": 0.0,
                "metric_seconds": time.perf_counter() - metric_started,
                "shared_statistics_seconds": stats_seconds,
                "alignment_global_lineage_auroc": global_alignment_auroc,
                "alignment_global_lineage_auprc": global_alignment_auprc,
                "alignment_parent_matched_auroc": matched_alignment["parent_matched_auroc"],
                "alignment_parent_matched_auprc": matched_alignment["parent_matched_auprc"],
                "alignment_parent_top1_accuracy": matched_alignment["parent_top1_accuracy"],
                "alignment_parent_mean_reciprocal_rank": matched_alignment[
                    "parent_mean_reciprocal_rank"
                ],
                "alignment_parent_ground_truth_defined": overlap > 0.0,
                "alignment_zero_overlap_designated_parent_top1_accuracy": (
                    designated_parent_alignment["parent_top1_accuracy"]
                    if overlap == 0.0 else float("nan")
                ),
                "collapse_differs_from_rank": collapse_order[:budget] != rank_sequence[:budget],
            })

    for row in rows:
        budget = int(row["budget"])
        collapse_reff = float(metrics_by_key[("collapse", budget)]["merged_reff"])
        rank_reff = float(metrics_by_key[("rank_only", budget)]["merged_reff"])
        dpp_reff = float(metrics_by_key[("dpp_subspace", budget)]["merged_reff"])
        row["collapse_minus_rank_reff"] = collapse_reff - rank_reff
        row["dpp_minus_rank_reff"] = dpp_reff - rank_reff
        row["collection_seconds"] = time.perf_counter() - collection_started
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
        required=True,
    )
    parser.add_argument("--sample-tag", default="5000")
    parser.add_argument("--pool-size", type=int, default=250)
    parser.add_argument("--pools-per-dataset", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--construction-seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--overlaps", type=float, nargs="+", default=DEFAULT_OVERLAPS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--include-full-rank", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-prefix", default="controlled_stopping")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if max(args.budgets) > len(DATASETS) * args.pools_per_dataset:
        raise ValueError("budget exceeds the number of independent base lineages")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets: dict[str, np.ndarray] = {}
    source_indices: dict[str, np.ndarray] = {}
    feature_metadata: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        features, indices, metadata = load_feature_set(
            args.feature_dir, args.encoder, dataset, args.sample_tag,
        )
        feature_sets[dataset] = features
        source_indices[dataset] = indices
        feature_metadata[dataset] = metadata

    all_collections = [
        (seed, overlap)
        for seed in args.construction_seeds
        for overlap in args.overlaps
    ]
    assigned = [
        item for index, item in enumerate(all_collections)
        if index % args.shard_count == args.shard_index
    ]
    rows: list[dict[str, object]] = []
    started_utc = utc_now()
    for collection_index, (seed, overlap) in enumerate(assigned, 1):
        current = run_collection(
            args, feature_sets, source_indices, seed, overlap,
        )
        rows.extend(current)
        print(
            f"[{collection_index}/{len(assigned)}] encoder={args.encoder} "
            f"seed={seed} overlap={overlap:.2f} rows={len(current)} "
            f"seconds={current[-1]['collection_seconds']:.3f} "
            f"collapse_minus_rank={current[-1]['collapse_minus_rank_reff']:.6f}",
            flush=True,
        )

    suffix = f"_{args.encoder}_shard{args.shard_index}of{args.shard_count}"
    result_path = args.out_dir / f"{args.output_prefix}{suffix}.csv"
    manifest_path = args.out_dir / f"{args.output_prefix}{suffix}.json"
    write_csv(result_path, rows)
    script_path = Path(__file__)
    manifest = {
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "encoder": args.encoder,
        "feature_dir": str(args.feature_dir),
        "sample_tag": args.sample_tag,
        "datasets": list(DATASETS),
        "pool_size": args.pool_size,
        "pools_per_dataset": args.pools_per_dataset,
        "construction_seeds": args.construction_seeds,
        "overlaps": args.overlaps,
        "budgets": args.budgets,
        "top_k": args.top_k,
        "n_random": args.n_random,
        "include_full_rank": args.include_full_rank,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_collections": assigned,
        "rows": len(rows),
        "result_sha256": sha256_file(result_path),
        "script_sha256": sha256_file(script_path),
        "classic_script_sha256": sha256_file(
            script_path.with_name("two_stage_classic_baselines.py")
        ),
        "feature_metadata": feature_metadata,
    }
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"wrote {result_path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
