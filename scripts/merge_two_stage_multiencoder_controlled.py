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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    rows: list[dict[str, str]], collapse_method: str,
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
        collapse_key = (encoder, seed, overlap, budget, collapse_method)
        dpp_key = (encoder, seed, overlap, budget, "dpp_subspace")
        if not all(key in lookup for key in (rank_key, collapse_key, dpp_key)):
            raise ValueError(f"missing paired methods for {(encoder, seed, overlap, budget)}")
        rank = lookup[rank_key]
        grouped[(encoder, overlap, budget)].append((
            rank,
            lookup[collapse_key] - rank,
            lookup[dpp_key] - rank,
        ))

    output: list[dict[str, object]] = []
    for (encoder, overlap, budget), values in sorted(grouped.items()):
        rank = np.asarray([value[0] for value in values], dtype=np.float64)
        collapse = np.asarray([value[1] for value in values], dtype=np.float64)
        dpp = np.asarray([value[2] for value in values], dtype=np.float64)
        tolerance = 1e-6 * np.maximum(np.abs(rank), 1.0)
        output.append({
            "encoder": encoder,
            "overlap": overlap,
            "budget": budget,
            "seeds": len(values),
            "collapse_minus_rank_mean": float(collapse.mean()),
            "collapse_minus_rank_std": float(collapse.std(ddof=1)) if len(collapse) > 1 else 0.0,
            "collapse_win_count": int((collapse > tolerance).sum()),
            "collapse_tie_count": int((np.abs(collapse) <= tolerance).sum()),
            "collapse_loss_count": int((collapse < -tolerance).sum()),
            "collapse_win_rate": float((collapse > tolerance).mean()),
            "dpp_minus_rank_mean": float(dpp.mean()),
            "dpp_minus_rank_std": float(dpp.std(ddof=1)) if len(dpp) > 1 else 0.0,
            "dpp_win_count": int((dpp > tolerance).sum()),
            "dpp_tie_count": int((np.abs(dpp) <= tolerance).sum()),
            "dpp_loss_count": int((dpp < -tolerance).sum()),
            "dpp_win_rate": float((dpp > tolerance).mean()),
        })
    return output


def summarize_stopping_decision(
    rows: list[dict[str, str]], collapse_method: str,
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
        collapse_delta = lookup[(encoder, seed, overlap, budget, collapse_method)] - rank
        dpp_delta = lookup[(encoder, seed, overlap, budget, "dpp_subspace")] - rank
        paired[encoder].append((rank, collapse_delta, dpp_delta))

    output: list[dict[str, object]] = []
    for encoder, values in sorted(paired.items()):
        rank = np.asarray([value[0] for value in values], dtype=np.float64)
        collapse = np.asarray([value[1] for value in values], dtype=np.float64)
        dpp = np.asarray([value[2] for value in values], dtype=np.float64)
        tolerance = 1e-6 * np.maximum(np.abs(rank), 1.0)
        output.append({
            "encoder": encoder,
            "configurations": len(values),
            "rank_reff_mean": float(rank.mean()),
            "collapse_minus_rank_mean": float(collapse.mean()),
            "collapse_relative_gain_mean": float(np.mean(collapse / rank)),
            "collapse_win_count": int((collapse > tolerance).sum()),
            "collapse_tie_count": int((np.abs(collapse) <= tolerance).sum()),
            "collapse_loss_count": int((collapse < -tolerance).sum()),
            "collapse_win_rate": float((collapse > tolerance).mean()),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="controlled_stopping")
    parser.add_argument("--collapse-method", default="collapse")
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
        for row in rows:
            key = (
                row["encoder"], row["construction_seed"], row["overlap"],
                row["budget"], row["method"], row["replicate"],
            )
            if key in seen_keys:
                raise ValueError(f"duplicate result key: {key}")
            seen_keys.add(key)
        all_rows.extend(rows)
        manifest_payloads.append(payload)

    encoders = sorted({row["encoder"] for row in all_rows})
    if encoders != sorted(args.expected_encoders):
        raise ValueError(f"encoders={encoders}, expected={sorted(args.expected_encoders)}")
    if len({payload["script_sha256"] for payload in manifest_payloads}) != 1:
        raise ValueError("controlled sweep script hashes differ")
    if len({payload["classic_script_sha256"] for payload in manifest_payloads}) != 1:
        raise ValueError("classic baseline script hashes differ")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.prefix}_results.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.csv"
    pairwise_path = args.out_dir / f"{args.prefix}_pairwise.csv"
    alignment_path = args.out_dir / f"{args.prefix}_alignment.csv"
    decision_path = args.out_dir / f"{args.prefix}_decision.csv"
    write_rows(merged_path, all_rows)
    write_rows(summary_path, summarize_methods(all_rows))
    write_rows(pairwise_path, summarize_pairwise(all_rows, args.collapse_method))
    write_rows(alignment_path, summarize_alignment(all_rows))
    write_rows(
        decision_path,
        summarize_stopping_decision(all_rows, args.collapse_method),
    )

    merged_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "encoders": encoders,
        "rows": len(all_rows),
        "source_manifests": [path.name for path in manifests],
        "controlled_script_sha256": manifest_payloads[0]["script_sha256"],
        "classic_script_sha256": manifest_payloads[0]["classic_script_sha256"],
        "collapse_method": args.collapse_method,
        "win_tie_loss_tolerance": "abs(delta) <= 1e-6 * max(abs(rank_reff), 1)",
        "merged_result_sha256": sha256_file(merged_path),
        "summary_sha256": sha256_file(summary_path),
        "pairwise_sha256": sha256_file(pairwise_path),
        "alignment_sha256": sha256_file(alignment_path),
        "decision_sha256": sha256_file(decision_path),
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
