#!/usr/bin/env python3
"""Strict summary-only controlled-overlap sweep for two-stage selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import two_stage_classic_baselines as classic
import two_stage_multiencoder_controlled as common


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pool_summaries(
    features: dict[str, np.ndarray], top_k: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Compatibility view of coherent statistics derived from rank-L sketches."""

    _, ranks, nuclear, subspaces, sketches = complete_pool_summaries(features, top_k)
    return ranks, nuclear, subspaces, sketches


def complete_pool_summaries(
    features: dict[str, np.ndarray], top_k: int,
) -> tuple[
    dict[str, classic.GramSketch],
    dict[str, float],
    dict[str, float],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Create rank-L factors plus exact source-local scalar summaries per pool."""

    ranks: dict[str, float] = {}
    nuclear: dict[str, float] = {}
    subspaces: dict[str, np.ndarray] = {}
    sketches: dict[str, np.ndarray] = {}
    summaries = classic.pool_gram_sketches(features, top_k)
    for name, summary in summaries.items():
        _, _, subspace = sketch_statistics(summary.factor, top_k)
        ranks[name] = summary.marginal_effective_rank
        nuclear[name] = summary.marginal_nuclear_mass
        subspaces[name] = subspace
        sketches[name] = summary.factor
    return summaries, ranks, nuclear, subspaces, sketches


def sketch_statistics(sketch: np.ndarray, top_k: int) -> tuple[float, float, np.ndarray]:
    matrix = sketch.astype(np.float64, copy=False)
    singular, vh = classic.stable_singular_values_and_vh(matrix)
    tolerance = max(
        float(singular[0]) * np.finfo(np.float64).eps * max(matrix.shape) * 8.0,
        np.finfo(np.float64).tiny,
    ) if singular.size else 0.0
    positive = singular > tolerance
    singular = singular[positive]
    vh = vh[positive]
    if singular.size == 0:
        return 0.0, 0.0, np.zeros((0, matrix.shape[1]), dtype=np.float64)
    effective_rank = classic.effective_rank_from_singular_values(singular)
    return effective_rank, float(singular.sum()), vh[: min(top_k, len(vh))]


def collapse_sketch_sequence(
    ranks: dict[str, float],
    nuclear: dict[str, float],
    subspaces: dict[str, np.ndarray],
    sketches: dict[str, np.ndarray],
    top_k: int,
    max_budget: int,
) -> list[str]:
    """Select using only transmitted pool summaries, never selected full features."""

    selected = [common.deterministic_argmax(ranks, lambda name: ranks[name])]
    current_sketch = sketches[selected[0]]
    while len(selected) < max_budget:
        current_rank, current_nuclear, current_subspace = sketch_statistics(
            current_sketch, top_k,
        )

        def score(name: str) -> float:
            cosines = np.linalg.svd(
                current_subspace @ subspaces[name].T, compute_uv=False,
            )
            alignment = float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))
            gamma = nuclear[name] / max(current_nuclear, 1e-12)
            return classic.collapse_predict(
                current_rank, ranks[name], gamma, alignment,
            )

        remaining = [name for name in ranks if name not in selected]
        choice = common.deterministic_argmax(remaining, score)
        selected.append(choice)
        current_sketch = np.concatenate(
            [current_sketch, sketches[choice]], axis=0,
        )
    return selected


def summary_bytes(dimension: int, top_k: int) -> dict[str, int]:
    return {
        "rank_only": 8,
        "rank_l_gram": 8 * (top_k * dimension + 3),
        "collapse_sketch": 8 * (top_k * dimension + 2),
        "dpp_subspace": 8 * (1 + top_k * dimension),
        "facility_subspace": 8 * (top_k * dimension),
        "agglomerative_subspace": 8 * (top_k * dimension),
        "lineage_deduplicated_rank": 8,
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
    pools, pool_indices, dataset_family, lineage, alias_parent = common.make_collection(
        feature_sets,
        source_indices,
        args.pool_size,
        args.pools_per_dataset,
        overlap,
        seed,
    )
    names = sorted(pools)
    stats_started = time.perf_counter()
    summaries, ranks, nuclear, subspaces, sketches = complete_pool_summaries(
        pools, args.top_k,
    )
    similarity = classic.similarity_matrix(
        pools, names, "subspace", {}, subspaces,
    )
    statistics_seconds = time.perf_counter() - stats_started
    designated_parent_alignment = common.matched_parent_detection(
        similarity, names, dataset_family, alias_parent,
    )
    if overlap > 0.0:
        global_alignment_auroc, global_alignment_auprc = common.alignment_detection(
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
    rank_l_started = time.perf_counter()
    rank_l_order, rank_l_diagnostics = classic.rank_l_gram_greedy(
        summaries, max_budget, return_diagnostics=True,
    )
    sequences["rank_l_gram"] = rank_l_order, time.perf_counter() - rank_l_started
    timed(
        "collapse_sketch",
        lambda: collapse_sketch_sequence(
            ranks, nuclear, subspaces, sketches, args.top_k, max_budget,
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
        "agglomerative_subspace",
        lambda: classic.agglomerative_medoids(similarity, names, max_budget),
    )
    timed(
        "lineage_deduplicated_rank",
        lambda: common.lineage_deduplicated_rank_sequence(ranks, lineage, max_budget),
    )
    include_full_rank = bool(getattr(args, "include_full_rank", False))
    n_random = int(getattr(args, "n_random", 0))
    if include_full_rank:
        timed(
            "exact_merged_rank_greedy",
            lambda: common.full_rank_sequence(pools, max_budget),
        )

    rank_order = sequences["rank_only"][0]
    collapse_order = sequences["collapse_sketch"][0]
    rows: list[dict[str, object]] = []
    metrics_by_key: dict[tuple[str, int], dict[str, float | int]] = {}
    for method, (sequence, selection_seconds) in sequences.items():
        for budget in args.budgets:
            selected = sequence[:budget]
            metric_started = time.perf_counter()
            metrics = common.selection_metrics(
                selected,
                pools,
                pool_indices,
                dataset_family,
                lineage,
                alias_parent,
            )
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
                "metric_seconds": time.perf_counter() - metric_started,
                "shared_statistics_seconds": statistics_seconds,
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
                "collapse_differs_from_rank": collapse_order[:budget] != rank_order[:budget],
                "rank_l_differs_from_rank": rank_l_order[:budget] != rank_order[:budget],
                "rank_l_certified_steps": sum(
                    int(item["certified_exact_greedy_choice"])
                    for item in rank_l_diagnostics[:budget]
                ),
                "rank_l_all_steps_certified": all(
                    bool(item["certified_exact_greedy_choice"])
                    for item in rank_l_diagnostics[:budget]
                ),
                "rank_l_max_chosen_log_error_bound": max(
                    float(item["log_error_bound"])
                    for item in rank_l_diagnostics[:budget]
                ),
            })

    rng = np.random.default_rng(seed + int(round(overlap * 1000)) + 991_337)
    for replicate in range(n_random):
        sequence = rng.permutation(names).tolist()
        for budget in args.budgets:
            selected = sequence[:budget]
            metric_started = time.perf_counter()
            metrics = common.selection_metrics(
                selected, pools, pool_indices, dataset_family, lineage, alias_parent,
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
                "shared_statistics_seconds": statistics_seconds,
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
                "collapse_differs_from_rank": collapse_order[:budget] != rank_order[:budget],
                "rank_l_differs_from_rank": rank_l_order[:budget] != rank_order[:budget],
                "rank_l_certified_steps": sum(
                    int(item["certified_exact_greedy_choice"])
                    for item in rank_l_diagnostics[:budget]
                ),
                "rank_l_all_steps_certified": all(
                    bool(item["certified_exact_greedy_choice"])
                    for item in rank_l_diagnostics[:budget]
                ),
                "rank_l_max_chosen_log_error_bound": max(
                    float(item["log_error_bound"])
                    for item in rank_l_diagnostics[:budget]
                ),
            })

    for row in rows:
        budget = int(row["budget"])
        rank_reff = float(metrics_by_key[("rank_only", budget)]["merged_reff"])
        collapse_reff = float(metrics_by_key[("collapse_sketch", budget)]["merged_reff"])
        dpp_reff = float(metrics_by_key[("dpp_subspace", budget)]["merged_reff"])
        rank_l_reff = float(metrics_by_key[("rank_l_gram", budget)]["merged_reff"])
        diagnostic = rank_l_diagnostics[budget - 1]
        approximate_log_rank = math.log(max(
            float(diagnostic["approximate_effective_rank"]), 1.0,
        ))
        exact_log_rank = math.log(max(rank_l_reff, 1.0))
        selected_log_error = abs(approximate_log_rank - exact_log_rank)
        selected_log_bound = float(diagnostic["log_error_bound"])
        row["collapse_minus_rank_reff"] = collapse_reff - rank_reff
        row["dpp_minus_rank_reff"] = dpp_reff - rank_reff
        row["rank_l_minus_rank_reff"] = rank_l_reff - rank_reff
        row["rank_l_selected_approximate_log_reff"] = approximate_log_rank
        row["rank_l_selected_exact_log_reff"] = exact_log_rank
        row["rank_l_selected_absolute_log_error"] = selected_log_error
        row["rank_l_selected_log_error_bound"] = selected_log_bound
        row["rank_l_selected_interval_covers_exact"] = (
            selected_log_error <= selected_log_bound + 1e-12
        )
        if include_full_rank:
            exact_order = sequences["exact_merged_rank_greedy"][0]
            row["rank_l_matches_exact_greedy_prefix"] = (
                rank_l_order[:budget] == exact_order[:budget]
            )
        else:
            row["rank_l_matches_exact_greedy_prefix"] = ""
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
    parser.add_argument(
        "--construction-seeds", type=int, nargs="+", default=common.DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--overlaps", type=float, nargs="+", default=common.DEFAULT_OVERLAPS,
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=common.DEFAULT_BUDGETS,
    )
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--include-full-rank", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-prefix", default="summary_stopping")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if max(args.budgets) > len(common.DATASETS) * args.pools_per_dataset:
        raise ValueError("budget exceeds the number of independent base lineages")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets: dict[str, np.ndarray] = {}
    source_indices: dict[str, np.ndarray] = {}
    feature_metadata: dict[str, dict[str, object]] = {}
    for dataset in common.DATASETS:
        features, indices, metadata = common.load_feature_set(
            args.feature_dir, args.encoder, dataset, args.sample_tag,
        )
        feature_sets[dataset] = features
        source_indices[dataset] = indices
        feature_metadata[dataset] = metadata

    provenance_fields = (
        "checkpoint", "checkpoint_file_sha256", "model_state_sha256",
        "preprocess_sha256", "encoder_loader_sha256",
    )
    first_metadata = feature_metadata[common.DATASETS[0]]
    encoder_provenance = {
        field: first_metadata[field] for field in provenance_fields
    }
    for dataset, metadata in feature_metadata.items():
        mismatches = [
            field for field in provenance_fields
            if metadata.get(field) != encoder_provenance[field]
        ]
        if mismatches:
            raise ValueError(
                f"encoder provenance differs for {dataset}: {mismatches}"
            )

    all_collections = [
        (seed, overlap)
        for seed in args.construction_seeds
        for overlap in args.overlaps
    ]
    assigned = [
        item for index, item in enumerate(all_collections)
        if index % args.shard_count == args.shard_index
    ]
    started_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
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
    dependency_path = script_path.with_name("two_stage_multiencoder_controlled.py")
    dimension = next(iter(feature_sets.values())).shape[1]
    bytes_per_pool = summary_bytes(dimension, args.top_k)
    bytes_per_pool["random"] = 0
    if args.include_full_rank:
        bytes_per_pool["exact_merged_rank_greedy"] = (
            4 * args.pool_size * dimension
        )
    manifest = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "encoder": args.encoder,
        "encoder_provenance": encoder_provenance,
        "feature_dir": str(args.feature_dir),
        "sample_tag": args.sample_tag,
        "stage1_cache_policy": "unlabeled_random_only",
        "datasets": list(common.DATASETS),
        "feature_index_sha256": {
            dataset: metadata["computed_index_sha256"]
            for dataset, metadata in feature_metadata.items()
        },
        "feature_sampling_protocol": {
            dataset: {
                field: metadata[field]
                for field in (
                    "dataset", "split", "source_samples", "requested_samples",
                    "sample_seed", "sampling", "stage1_sampling_reads_labels",
                    "source_file_sha256",
                )
            }
            for dataset, metadata in feature_metadata.items()
        },
        "pool_size": args.pool_size,
        "pools_per_dataset": args.pools_per_dataset,
        "construction_seeds": args.construction_seeds,
        "overlaps": args.overlaps,
        "budgets": args.budgets,
        "top_k": args.top_k,
        "n_random": args.n_random,
        "include_full_rank": args.include_full_rank,
        "primary_method": "rank_l_gram",
        "legacy_collapse_method": "collapse_sketch",
        "marginal_scalar_mode": "exact_source_local",
        "rank_l_mode": "direct_composable_psd_factor_with_tail_bounds",
        "collapse_mode": "legacy_summary_only_weighted_topk_sketch",
        "full_features_used_for_selection": args.include_full_rank,
        "full_features_used_for_evaluation_only": not args.include_full_rank,
        "bytes_per_pool": bytes_per_pool,
        "selected_pool_feedback_bytes": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_collections": assigned,
        "rows": len(rows),
        "result_sha256": sha256_file(result_path),
        "script_sha256": sha256_file(script_path),
        "common_script_sha256": sha256_file(dependency_path),
        "classic_script_sha256": sha256_file(
            script_path.with_name("two_stage_classic_baselines.py")
        ),
        "feature_metadata": feature_metadata,
    }
    temporary = manifest_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(f"wrote {result_path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
