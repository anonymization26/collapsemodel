#!/usr/bin/env python3
"""Validate and merge cross-encoder controlled-overlap sweep shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


COMPATIBILITY_FIELDS = (
    "feature_dir",
    "sample_tag",
    "stage1_cache_policy",
    "datasets",
    "feature_index_sha256",
    "feature_sampling_protocol",
    "pool_size",
    "pools_per_dataset",
    "construction_seeds",
    "overlaps",
    "budgets",
    "top_k",
    "n_random",
    "include_full_rank",
    "primary_method",
    "legacy_collapse_method",
    "marginal_scalar_mode",
    "rank_l_mode",
    "collapse_mode",
    "full_features_used_for_selection",
    "full_features_used_for_evaluation_only",
    "selected_pool_feedback_bytes",
)

CONTROLLED_METHODS = {
    "rank_only",
    "rank_l_gram",
    "collapse_sketch",
    "dpp_subspace",
    "facility_subspace",
    "agglomerative_subspace",
    "lineage_deduplicated_rank",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def compatibility_signature(payload: dict[str, object]) -> dict[str, object]:
    """Return protocol fields that must agree across encoder manifests."""

    missing = [field for field in COMPATIBILITY_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"manifest lacks corrected-protocol fields: {missing}")
    return {field: payload[field] for field in COMPATIBILITY_FIELDS}


def validate_encoder_provenance(payload: dict[str, object]) -> dict[str, object]:
    provenance = payload.get("encoder_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("manifest lacks encoder provenance")
    required = (
        "checkpoint", "checkpoint_file_sha256", "model_state_sha256",
        "preprocess_sha256", "encoder_loader_sha256",
    )
    missing = [field for field in required if field not in provenance]
    if missing:
        raise ValueError(f"encoder provenance lacks fields: {missing}")
    for field in ("model_state_sha256", "preprocess_sha256", "encoder_loader_sha256"):
        if not valid_sha256(provenance[field]):
            raise ValueError(f"invalid encoder provenance hash: {field}")
    checkpoint_hash = provenance["checkpoint_file_sha256"]
    if checkpoint_hash is not None and not valid_sha256(checkpoint_hash):
        raise ValueError("invalid checkpoint file hash")
    return provenance


def register_result_keys(
    rows: list[dict[str, str]],
    seen_keys: set[tuple[str, str, str, str, str, str]],
) -> None:
    """Reject duplicate method results within or across shards."""

    for row in rows:
        key = (
            row["encoder"],
            row["construction_seed"],
            row["overlap"],
            row["budget"],
            row["method"],
            row["replicate"],
        )
        if key in seen_keys:
            raise ValueError(f"duplicate result key: {key}")
        seen_keys.add(key)


def validate_manifest_shard(
    payload: dict[str, object], rows: list[dict[str, str]],
) -> None:
    """Prove that one result shard exactly covers its declared assignment."""

    shard_index = int(payload.get("shard_index", -1))
    shard_count = int(payload.get("shard_count", 0))
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard index/count: {shard_index}/{shard_count}")
    construction_seeds = [int(value) for value in payload["construction_seeds"]]
    overlaps = [float(value) for value in payload["overlaps"]]
    budgets = [int(value) for value in payload["budgets"]]
    if (
        len(construction_seeds) != len(set(construction_seeds))
        or len(overlaps) != len(set(overlaps))
        or len(budgets) != len(set(budgets))
    ):
        raise ValueError("manifest seeds, overlaps, and budgets must be unique")
    all_collections = [
        (seed, overlap) for seed in construction_seeds for overlap in overlaps
    ]
    expected_assigned = [
        item for index, item in enumerate(all_collections)
        if index % shard_count == shard_index
    ]
    assigned = [
        (int(item[0]), float(item[1]))
        for item in payload.get("assigned_collections", [])
    ]
    if assigned != expected_assigned:
        raise ValueError(
            f"shard {shard_index}/{shard_count} has an invalid collection assignment"
        )

    methods = set(CONTROLLED_METHODS)
    if bool(payload["include_full_rank"]):
        methods.add("exact_merged_rank_greedy")
    n_random = int(payload["n_random"])
    if n_random < 0:
        raise ValueError("n_random must be non-negative")
    expected_keys = {
        (str(payload["encoder"]), str(seed), str(overlap), str(budget), method, "0")
        for seed, overlap in assigned
        for budget in budgets
        for method in methods
    }
    expected_keys.update({
        (
            str(payload["encoder"]), str(seed), str(overlap), str(budget),
            "random", str(replicate),
        )
        for seed, overlap in assigned
        for budget in budgets
        for replicate in range(n_random)
    })
    observed_keys = {
        (
            row["encoder"], str(int(row["construction_seed"])),
            str(float(row["overlap"])), str(int(row["budget"])),
            row["method"], str(int(row["replicate"])),
        )
        for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        missing = sorted(expected_keys - observed_keys)[:5]
        extra = sorted(observed_keys - expected_keys)[:5]
        raise ValueError(
            f"controlled shard is incomplete or has extra rows; missing={missing}, extra={extra}"
        )
    for row in rows:
        selected = row.get("selected", "").split("|")
        budget = int(row["budget"])
        if len(selected) != budget or len(set(selected)) != budget:
            raise ValueError("controlled result has an invalid selected set")


def validate_encoder_shards(
    encoder: str, payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Reject missing shards and inconsistent encoder-specific inputs."""

    shard_counts = {int(payload["shard_count"]) for payload in payloads}
    if len(shard_counts) != 1:
        raise ValueError(f"shard counts differ for {encoder}")
    shard_count = next(iter(shard_counts))
    shard_indices = [int(payload["shard_index"]) for payload in payloads]
    if sorted(shard_indices) != list(range(shard_count)):
        raise ValueError(
            f"shards are incomplete for {encoder}: got {sorted(shard_indices)}, "
            f"expected {list(range(shard_count))}"
        )
    provenance_values = [validate_encoder_provenance(payload) for payload in payloads]
    if any(value != provenance_values[0] for value in provenance_values[1:]):
        raise ValueError(f"encoder provenance differs across shards for {encoder}")
    feature_metadata = [payload.get("feature_metadata") for payload in payloads]
    if any(value != feature_metadata[0] for value in feature_metadata[1:]):
        raise ValueError(f"feature metadata differs across shards for {encoder}")
    return provenance_values[0]


def summarize_methods(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["encoder"], row["method"], int(row["budget"]))].append(row)

    output: list[dict[str, object]] = []
    for (encoder, method, budget), group in sorted(grouped.items()):
        reff = [float(row["merged_reff"]) for row in group]
        output.append({
            "encoder": encoder,
            "method": method,
            "budget": budget,
            "configurations": len(group),
            "merged_reff_mean": mean(reff),
            "merged_reff_std": sample_std(reff),
            "unique_lineages_mean": mean([float(row["unique_lineages"]) for row in group]),
            "lineage_duplicate_count_mean": mean([
                float(row["lineage_duplicate_count"]) for row in group
            ]),
            "unique_datasets_mean": mean([float(row["unique_datasets"]) for row in group]),
            "selected_alias_parent_pairs_mean": mean([
                float(row["selected_alias_parent_pairs"]) for row in group
            ]),
            "max_selected_sample_overlap_mean": mean([
                float(row["max_selected_sample_overlap"]) for row in group
            ]),
            "selection_seconds_for_max_budget_mean": mean([
                float(row["selection_seconds_for_max_budget"]) for row in group
            ]),
            "metric_seconds_mean": mean([float(row["metric_seconds"]) for row in group]),
        })
    return output


def summarize_pairwise(
    rows: list[dict[str, str]], primary_method: str,
) -> list[dict[str, object]]:
    lookup: dict[tuple[str, int, float, int, str], float] = {}
    for row in rows:
        if int(row["replicate"]) != 0:
            continue
        key = (
            row["encoder"],
            int(row["construction_seed"]),
            float(row["overlap"]),
            int(row["budget"]),
            row["method"],
        )
        lookup[key] = float(row["merged_reff"])

    grouped: dict[tuple[str, float, int], list[tuple[float, float, float]]] = defaultdict(list)
    bases = sorted({key[:4] for key in lookup})
    for encoder, seed, overlap, budget in bases:
        rank_key = (encoder, seed, overlap, budget, "rank_only")
        primary_key = (encoder, seed, overlap, budget, primary_method)
        dpp_key = (encoder, seed, overlap, budget, "dpp_subspace")
        if not all(key in lookup for key in (rank_key, primary_key, dpp_key)):
            raise ValueError(f"missing paired methods for {(encoder, seed, overlap, budget)}")
        rank = lookup[rank_key]
        grouped[(encoder, overlap, budget)].append((
            rank,
            lookup[primary_key] - rank,
            lookup[dpp_key] - rank,
        ))

    output: list[dict[str, object]] = []
    for (encoder, overlap, budget), values in sorted(grouped.items()):
        rank = np.asarray([value[0] for value in values], dtype=np.float64)
        primary = np.asarray([value[1] for value in values], dtype=np.float64)
        dpp = np.asarray([value[2] for value in values], dtype=np.float64)
        tolerance = 1e-6 * np.maximum(np.abs(rank), 1.0)
        output.append({
            "encoder": encoder,
            "primary_method": primary_method,
            "overlap": overlap,
            "budget": budget,
            "seeds": len(values),
            "primary_minus_rank_mean": float(primary.mean()),
            "primary_minus_rank_std": float(primary.std(ddof=1)) if len(primary) > 1 else 0.0,
            "primary_win_count": int((primary > tolerance).sum()),
            "primary_tie_count": int((np.abs(primary) <= tolerance).sum()),
            "primary_loss_count": int((primary < -tolerance).sum()),
            "primary_win_rate": float((primary > tolerance).mean()),
            "dpp_minus_rank_mean": float(dpp.mean()),
            "dpp_minus_rank_std": float(dpp.std(ddof=1)) if len(dpp) > 1 else 0.0,
            "dpp_win_count": int((dpp > tolerance).sum()),
            "dpp_tie_count": int((np.abs(dpp) <= tolerance).sum()),
            "dpp_loss_count": int((dpp < -tolerance).sum()),
            "dpp_win_rate": float((dpp > tolerance).mean()),
        })
    return output


def summarize_stopping_decision(
    rows: list[dict[str, str]], primary_method: str,
) -> list[dict[str, object]]:
    lookup: dict[tuple[str, int, float, int, str], float] = {}
    for row in rows:
        if int(row["replicate"]) != 0:
            continue
        lookup[(
            row["encoder"],
            int(row["construction_seed"]),
            float(row["overlap"]),
            int(row["budget"]),
            row["method"],
        )] = float(row["merged_reff"])

    paired: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    bases = sorted({key[:4] for key in lookup})
    for encoder, seed, overlap, budget in bases:
        rank = lookup[(encoder, seed, overlap, budget, "rank_only")]
        primary_delta = lookup[(encoder, seed, overlap, budget, primary_method)] - rank
        dpp_delta = lookup[(encoder, seed, overlap, budget, "dpp_subspace")] - rank
        paired[encoder].append((rank, primary_delta, dpp_delta))

    output: list[dict[str, object]] = []
    for encoder, values in sorted(paired.items()):
        rank = np.asarray([value[0] for value in values], dtype=np.float64)
        primary = np.asarray([value[1] for value in values], dtype=np.float64)
        dpp = np.asarray([value[2] for value in values], dtype=np.float64)
        tolerance = 1e-6 * np.maximum(np.abs(rank), 1.0)
        output.append({
            "encoder": encoder,
            "primary_method": primary_method,
            "configurations": len(values),
            "rank_reff_mean": float(rank.mean()),
            "primary_minus_rank_mean": float(primary.mean()),
            "primary_relative_gain_mean": float(np.mean(primary / rank)),
            "primary_win_count": int((primary > tolerance).sum()),
            "primary_tie_count": int((np.abs(primary) <= tolerance).sum()),
            "primary_loss_count": int((primary < -tolerance).sum()),
            "primary_win_rate": float((primary > tolerance).mean()),
            "dpp_minus_rank_mean": float(dpp.mean()),
            "dpp_relative_gain_mean": float(np.mean(dpp / rank)),
            "dpp_win_count": int((dpp > tolerance).sum()),
            "dpp_tie_count": int((np.abs(dpp) <= tolerance).sum()),
            "dpp_loss_count": int((dpp < -tolerance).sum()),
            "dpp_win_rate": float((dpp > tolerance).mean()),
        })
    return output


def summarize_alignment(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    unique: dict[tuple[str, int, float], tuple[float, ...]] = {}
    for row in rows:
        key = (
            row["encoder"],
            int(row["construction_seed"]),
            float(row["overlap"]),
        )
        if "alignment_global_lineage_auroc" in row:
            value = (
                float(row["alignment_global_lineage_auroc"]),
                float(row["alignment_global_lineage_auprc"]),
                float(row["alignment_parent_matched_auroc"]),
                float(row["alignment_parent_matched_auprc"]),
                float(row["alignment_parent_top1_accuracy"]),
                float(row["alignment_parent_mean_reciprocal_rank"]),
                float(row["alignment_zero_overlap_designated_parent_top1_accuracy"]),
            )
        else:
            value = (
                float(row["alignment_duplicate_auroc"]),
                float(row["alignment_duplicate_auprc"]),
                float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
            )
        if key in unique and not np.allclose(unique[key], value, equal_nan=True):
            raise ValueError(f"inconsistent alignment metrics for {key}")
        unique[key] = value

    grouped: dict[tuple[str, float], list[tuple[float, ...]]] = defaultdict(list)
    for (encoder, _, overlap), value in unique.items():
        grouped[(encoder, overlap)].append(value)
    output: list[dict[str, object]] = []
    for (encoder, overlap), values in sorted(grouped.items()):
        output.append({
            "encoder": encoder,
            "overlap": overlap,
            "seeds": len(values),
            "alignment_global_lineage_auroc_mean": mean([value[0] for value in values]),
            "alignment_global_lineage_auroc_std": sample_std([value[0] for value in values]),
            "alignment_global_lineage_auprc_mean": mean([value[1] for value in values]),
            "alignment_global_lineage_auprc_std": sample_std([value[1] for value in values]),
            "alignment_parent_matched_auroc_mean": mean([value[2] for value in values]),
            "alignment_parent_matched_auroc_std": sample_std([value[2] for value in values]),
            "alignment_parent_matched_auprc_mean": mean([value[3] for value in values]),
            "alignment_parent_matched_auprc_std": sample_std([value[3] for value in values]),
            "alignment_parent_top1_accuracy_mean": mean([value[4] for value in values]),
            "alignment_parent_top1_accuracy_std": sample_std([value[4] for value in values]),
            "alignment_parent_mean_reciprocal_rank_mean": mean([
                value[5] for value in values
            ]),
            "alignment_parent_mean_reciprocal_rank_std": sample_std([
                value[5] for value in values
            ]),
            "alignment_zero_overlap_designated_parent_top1_accuracy_mean": mean([
                value[6] for value in values
            ]),
        })
    return output


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected CSV boolean, got {value!r}")


def summarize_rank_l_diagnostics(
    rows: list[dict[str, str]], primary_method: str,
) -> list[dict[str, object]]:
    primary_rows = [
        row for row in rows
        if row["method"] == primary_method and int(row["replicate"]) == 0
    ]
    if not primary_rows or "rank_l_selected_absolute_log_error" not in primary_rows[0]:
        return []

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in primary_rows:
        grouped[row["encoder"]].append(row)

    output: list[dict[str, object]] = []
    for encoder, group in sorted(grouped.items()):
        errors = [float(row["rank_l_selected_absolute_log_error"]) for row in group]
        bounds = [float(row["rank_l_selected_log_error_bound"]) for row in group]
        coverage = [
            parse_bool(row["rank_l_selected_interval_covers_exact"])
            for row in group
        ]
        maximum_budget = max(int(row["budget"]) for row in group)
        terminal = [row for row in group if int(row["budget"]) == maximum_budget]
        exact_prefix_values = [
            row["rank_l_matches_exact_greedy_prefix"] for row in group
            if row["rank_l_matches_exact_greedy_prefix"]
        ]
        output.append({
            "encoder": encoder,
            "primary_method": primary_method,
            "configurations": len(group),
            "selected_interval_coverage_rate": mean([float(value) for value in coverage]),
            "selected_absolute_log_error_mean": mean(errors),
            "selected_absolute_log_error_max": max(errors),
            "selected_log_error_bound_mean": mean(bounds),
            "selected_log_error_bound_min": min(bounds),
            "selected_log_error_bound_max": max(bounds),
            "max_budget": maximum_budget,
            "max_budget_runs": len(terminal),
            "certified_steps_total": sum(
                int(row["rank_l_certified_steps"]) for row in terminal
            ),
            "evaluated_steps_total": len(terminal) * maximum_budget,
            "all_steps_certified_run_rate": mean([
                float(parse_bool(row["rank_l_all_steps_certified"]))
                for row in terminal
            ]),
            "exact_greedy_prefix_match_rate": (
                mean([float(parse_bool(value)) for value in exact_prefix_values])
                if exact_prefix_values else float("nan")
            ),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="controlled_stopping")
    parser.add_argument(
        "--primary-method",
        "--collapse-method",
        dest="primary_method",
        default="rank_l_gram",
        help="Primary Stage-1 method (the old --collapse-method spelling is an alias).",
    )
    parser.add_argument(
        "--expected-encoders",
        nargs="+",
        default=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
    )
    args = parser.parse_args()

    manifests = sorted(args.input_dir.glob(f"{args.prefix}_*_shard*of*.json"))
    if not manifests:
        raise FileNotFoundError(f"no manifests in {args.input_dir}")
    all_rows: list[dict[str, str]] = []
    manifest_payloads: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        result_path = manifest_path.with_suffix(".csv")
        if sha256_file(result_path) != payload["result_sha256"]:
            raise ValueError(f"result checksum mismatch: {result_path}")
        rows = read_rows(result_path)
        if len(rows) != payload["rows"]:
            raise ValueError(f"row count mismatch: {result_path}")
        validate_manifest_shard(payload, rows)
        register_result_keys(rows, seen_keys)
        all_rows.extend(rows)
        manifest_payloads.append(payload)

    encoders = sorted({row["encoder"] for row in all_rows})
    if encoders != sorted(args.expected_encoders):
        raise ValueError(f"encoders={encoders}, expected={sorted(args.expected_encoders)}")
    if len({payload["script_sha256"] for payload in manifest_payloads}) != 1:
        raise ValueError("controlled sweep script hashes differ")
    if len({payload["classic_script_sha256"] for payload in manifest_payloads}) != 1:
        raise ValueError("classic baseline script hashes differ")
    if len({payload["common_script_sha256"] for payload in manifest_payloads}) != 1:
        raise ValueError("controlled collection script hashes differ")
    payloads_by_encoder: dict[str, list[dict[str, object]]] = defaultdict(list)
    for payload in manifest_payloads:
        payloads_by_encoder[str(payload["encoder"])].append(payload)
    encoder_provenance = {
        encoder: validate_encoder_shards(encoder, payloads)
        for encoder, payloads in sorted(payloads_by_encoder.items())
    }
    protocol_signatures = [compatibility_signature(payload) for payload in manifest_payloads]
    reference_signature = protocol_signatures[0]
    for payload, signature in zip(manifest_payloads[1:], protocol_signatures[1:]):
        if signature != reference_signature:
            mismatches = [
                field for field in COMPATIBILITY_FIELDS
                if signature[field] != reference_signature[field]
            ]
            raise ValueError(
                "corrected-protocol fields differ for "
                f"{payload.get('encoder', 'unknown')}: {mismatches}"
            )
    if args.primary_method != reference_signature["primary_method"]:
        raise ValueError(
            f"requested primary method {args.primary_method!r} does not match "
            f"manifest primary method {reference_signature['primary_method']!r}"
        )
    if args.primary_method not in {row["method"] for row in all_rows}:
        raise ValueError(f"primary method is absent from results: {args.primary_method}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.prefix}_results.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.csv"
    pairwise_path = args.out_dir / f"{args.prefix}_pairwise.csv"
    alignment_path = args.out_dir / f"{args.prefix}_alignment.csv"
    decision_path = args.out_dir / f"{args.prefix}_decision.csv"
    rank_l_path = args.out_dir / f"{args.prefix}_rank_l_diagnostics.csv"
    write_rows(merged_path, all_rows)
    write_rows(summary_path, summarize_methods(all_rows))
    write_rows(pairwise_path, summarize_pairwise(all_rows, args.primary_method))
    write_rows(alignment_path, summarize_alignment(all_rows))
    write_rows(
        decision_path,
        summarize_stopping_decision(all_rows, args.primary_method),
    )
    rank_l_rows = summarize_rank_l_diagnostics(all_rows, args.primary_method)
    if rank_l_rows:
        write_rows(rank_l_path, rank_l_rows)

    merged_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoders": encoders,
        "rows": len(all_rows),
        "source_manifests": [path.name for path in manifests],
        "source_manifest_sha256": {
            path.name: sha256_file(path) for path in manifests
        },
        "controlled_script_sha256": manifest_payloads[0]["script_sha256"],
        "common_script_sha256": manifest_payloads[0]["common_script_sha256"],
        "classic_script_sha256": manifest_payloads[0]["classic_script_sha256"],
        "encoder_provenance": encoder_provenance,
        "primary_method": args.primary_method,
        "win_tie_loss_tolerance": "abs(delta) <= 1e-6 * max(abs(rank_reff), 1)",
        "merged_result_sha256": sha256_file(merged_path),
        "summary_sha256": sha256_file(summary_path),
        "pairwise_sha256": sha256_file(pairwise_path),
        "alignment_sha256": sha256_file(alignment_path),
        "decision_sha256": sha256_file(decision_path),
        "rank_l_diagnostics_sha256": (
            sha256_file(rank_l_path) if rank_l_rows else None
        ),
    }
    manifest_path = args.out_dir / f"{args.prefix}_manifest.json"
    temporary = manifest_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(merged_manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    print(
        f"merged manifests={len(manifests)} encoders={len(encoders)} rows={len(all_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
