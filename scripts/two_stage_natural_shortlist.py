#!/usr/bin/env python3
"""Natural-pool shortlist recall with exhaustive fixed-feature adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# CPU-only screen/summarize commands must not auto-import torch_npu.  The adapt
# command loads it explicitly after the Ascend environment has been configured.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
import torch.nn as nn

import two_stage_classic_baselines as classic


SOURCES = [
    "bloodmnist", "breastmnist", "cifar10", "cifar100", "cifar100_coarse",
    "dermamnist", "fashion_mnist", "mnist", "octmnist", "organamnist",
    "organcmnist", "organsmnist", "pathmnist", "pneumoniamnist",
    "rendered_sst2", "retinamnist", "stl10", "svhn", "tiny_imagenet",
    "tissuemnist", "usps",
]
TARGETS = ["beans", "dtd", "eurosat", "flowers102", "food101", "gtsrb", "oxford_pets"]
SHORTLIST_SIZES = [3, 5, 10]

ENCODER_PROVENANCE_FIELDS = (
    "checkpoint",
    "checkpoint_file_sha256",
    "model_state_sha256",
    "preprocess_sha256",
    "encoder_loader_sha256",
)

SOURCE_DOMAINS = {
    **{name: "medical" for name in [
        "bloodmnist", "breastmnist", "dermamnist", "octmnist", "organamnist",
        "organcmnist", "organsmnist", "pathmnist", "pneumoniamnist",
        "retinamnist", "tissuemnist",
    ]},
    **{name: "digits_characters" for name in ["fashion_mnist", "mnist", "svhn", "usps"]},
    **{name: "general_vision" for name in [
        "cifar10", "cifar100", "cifar100_coarse", "stl10", "tiny_imagenet",
    ]},
    "rendered_sst2": "rendered_text",
}

SOURCE_FAMILIES = {name: name for name in SOURCES}
SOURCE_FAMILIES.update({
    "cifar100": "cifar100_shared_images",
    "cifar100_coarse": "cifar100_shared_images",
    "mnist": "handwritten_digits",
    "usps": "handwritten_digits",
    "organamnist": "organmnist_views",
    "organcmnist": "organmnist_views",
    "organsmnist": "organmnist_views",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(name: str, base: int) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-8)


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        features = normalize_rows(archive["H"])
        labels = np.asarray(archive["y"], dtype=np.int64).reshape(-1)
    if len(features) != len(labels):
        raise ValueError(f"feature/label length mismatch in {path}")
    return features, labels.astype(np.int64, copy=False)


def load_features(path: Path) -> np.ndarray:
    """Load only H so Stage-1 remains valid for feature-only, label-free archives."""

    with np.load(path, allow_pickle=False) as archive:
        return normalize_rows(archive["H"])


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def cache_encoder_provenance(metadata: dict[str, object]) -> dict[str, object]:
    missing = [field for field in ENCODER_PROVENANCE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"cache metadata lacks encoder provenance: {missing}")
    for field in ("model_state_sha256", "preprocess_sha256", "encoder_loader_sha256"):
        if not _valid_sha256(metadata[field]):
            raise ValueError(f"cache metadata has invalid encoder provenance: {field}")
    checkpoint_hash = metadata["checkpoint_file_sha256"]
    if checkpoint_hash is not None and not _valid_sha256(checkpoint_hash):
        raise ValueError("cache metadata has an invalid checkpoint file hash")
    return {field: metadata[field] for field in ENCODER_PROVENANCE_FIELDS}


def _audit_feature_archive(
    path: Path, metadata: dict[str, object], require_labels: bool,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if "H" not in archive or "indices" not in archive:
            raise ValueError(f"feature archive lacks H or indices: {path}")
        features = np.asarray(archive["H"])
        indices = np.asarray(archive["indices"], dtype=np.int64)
        labels = np.asarray(archive["y"]).reshape(-1) if "y" in archive else None
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError(f"feature archive has invalid features: {path}")
    if (
        indices.ndim != 1
        or len(indices) != len(features)
        or len(np.unique(indices)) != len(indices)
        or np.any(indices < 0)
    ):
        raise ValueError(f"feature archive has invalid sample indices: {path}")
    if require_labels and (labels is None or len(labels) != len(features)):
        raise ValueError(f"Stage-2 cache has invalid labels: {path}")
    declared_shape = metadata.get("feature_shape")
    if declared_shape is not None and list(features.shape) != declared_shape:
        raise ValueError(f"feature metadata shape mismatch: {path}")
    return {
        "computed_index_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "computed_feature_shape": list(features.shape),
    }


def stage1_cache_metadata(
    path: Path,
    allow_label_informed: bool,
    *,
    expected_encoder: str | None = None,
    expected_dataset: str | None = None,
    expected_samples: int | None = None,
) -> dict[str, object]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        if allow_label_informed:
            return {
                "sampling": "unverified",
                "label_independent_sampling": False,
                "computed_feature_file_sha256": sha256_file(path),
                "metadata_file_sha256": "",
            }
        raise ValueError(f"Stage-1 cache has no provenance metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    feature_sha256 = sha256_file(path)
    declared_sha256 = metadata.get("feature_file_sha256")
    if declared_sha256 is None and not allow_label_informed:
        raise ValueError(f"Stage-1 cache metadata has no feature checksum: {metadata_path}")
    if declared_sha256 is not None and declared_sha256 != feature_sha256:
        raise ValueError(f"Stage-1 cache checksum mismatch: {path}")
    expected_identity = {
        "variant": expected_encoder,
        "dataset": expected_dataset,
        "split": "train" if expected_dataset is not None else None,
        "requested_samples": expected_samples,
    }
    mismatches = [
        field for field, expected in expected_identity.items()
        if expected is not None and metadata.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"Stage-1 cache identity mismatch in {metadata_path}: {mismatches}")
    sampling = str(metadata.get("sampling", "unknown"))
    label_independent = (
        sampling == "unlabeled_random"
        and metadata.get("stage1_sampling_reads_labels") is False
    )
    if not label_independent and not allow_label_informed:
        raise ValueError(
            f"Stage-1 cache {path} used label-informed or unknown sampling={sampling!r}; "
            "regenerate it with --sampling unlabeled_random"
        )
    archive_audit = _audit_feature_archive(path, metadata, require_labels=False)
    provenance = None
    try:
        provenance = cache_encoder_provenance(metadata)
    except ValueError:
        if not allow_label_informed:
            raise
    return {
        **metadata,
        **archive_audit,
        "encoder_provenance": provenance,
        "label_independent_sampling": label_independent,
        "computed_feature_file_sha256": feature_sha256,
        "metadata_file_sha256": sha256_file(metadata_path),
    }


def stage2_cache_metadata(
    path: Path,
    *,
    expected_encoder: str,
    expected_dataset: str,
    expected_split: str,
    expected_samples: int,
    label_read_field: str,
) -> dict[str, object]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(f"Stage-2 cache has no provenance metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected_identity = {
        "variant": expected_encoder,
        "dataset": expected_dataset,
        "split": expected_split,
        "requested_samples": expected_samples,
    }
    mismatches = [
        field for field, expected in expected_identity.items()
        if metadata.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"Stage-2 cache identity mismatch in {metadata_path}: {mismatches}")
    if (
        metadata.get("sampling") != "unlabeled_random"
        or metadata.get(label_read_field) is not False
    ):
        raise ValueError(
            f"Stage-2 cache subset is not label-independent: {metadata_path}"
        )
    feature_sha256 = sha256_file(path)
    if metadata.get("feature_file_sha256") != feature_sha256:
        raise ValueError(f"Stage-2 cache checksum mismatch: {path}")
    audit = _audit_feature_archive(path, metadata, require_labels=True)
    return {
        **metadata,
        **audit,
        "encoder_provenance": cache_encoder_provenance(metadata),
        "computed_feature_file_sha256": feature_sha256,
        "metadata_file_sha256": sha256_file(metadata_path),
    }


def source_path(root: Path, encoder: str, source: str, samples: int) -> Path:
    return root / f"{encoder}__{source}__train__n{samples}.npz"


def target_path(
    root: Path, encoder: str, target: str, split: str, samples: int,
) -> Path:
    return root / f"{encoder}__{target}__{split}__n{samples}.npz"


def sample_unlabeled(features: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= len(features):
        return features
    indices = np.random.default_rng(seed).choice(len(features), count, replace=False)
    return features[indices]


def stratified_sample_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if count < len(classes):
        raise ValueError(
            f"cannot represent {len(classes)} classes with a {count}-sample Stage-2 budget"
        )
    exact = count * counts.astype(np.float64) / len(labels)
    allocation = np.maximum(np.floor(exact).astype(np.int64), 1)
    while int(allocation.sum()) > count:
        candidates = np.flatnonzero(allocation > 1)
        index = candidates[np.argmin(exact[candidates] - allocation[candidates])]
        allocation[index] -= 1
    while int(allocation.sum()) < count:
        candidates = np.flatnonzero(allocation < counts)
        index = candidates[np.argmax(exact[candidates] - allocation[candidates])]
        allocation[index] += 1
    rng = np.random.default_rng(seed)
    selected = [
        rng.choice(np.flatnonzero(labels == label), int(size), replace=False)
        for label, size in zip(classes, allocation)
    ]
    indices = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(indices)
    return indices


def effective_rank_from_scatter(scatter: np.ndarray) -> float:
    return classic.effective_rank_from_scatter(scatter)


def source_scatters(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: value.astype(np.float64, copy=False).T @ value.astype(np.float64, copy=False)
        for name, value in features.items()
    }


def full_rank_greedy_from_scatters(
    scatters: dict[str, np.ndarray],
    budget: int,
    source_rows: dict[str, int] | None = None,
) -> list[str]:
    selected: list[str] = []
    current = np.zeros_like(next(iter(scatters.values())), dtype=np.float64)
    current_rows = 0
    while len(selected) < budget:
        remaining = sorted(set(scatters) - set(selected))

        def score(name: str) -> float:
            rows = (
                current_rows + source_rows[name]
                if source_rows is not None
                else current.shape[0]
            )
            return classic.effective_rank_from_scatter(
                current + scatters[name],
                source_shape=(rows, current.shape[0]),
            )

        choice = max(
            remaining,
            key=score,
        )
        selected.append(choice)
        current = current + scatters[choice]
        if source_rows is not None:
            current_rows += source_rows[choice]
    return selected


def full_rank_greedy(features: dict[str, np.ndarray], budget: int) -> list[str]:
    return full_rank_greedy_from_scatters(
        source_scatters(features),
        budget,
        source_rows={name: len(value) for name, value in features.items()},
    )


def run_screen(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = {}
    source_rows = []
    encoder_provenance: dict[str, object] | None = None
    for source in args.sources:
        path = source_path(args.source_dir, args.encoder, source, args.cache_samples)
        cache_metadata = stage1_cache_metadata(
            path,
            args.allow_label_informed_cache,
            expected_encoder=args.encoder,
            expected_dataset=source,
            expected_samples=args.cache_samples,
        )
        current_provenance = cache_metadata.get("encoder_provenance")
        if current_provenance is not None:
            if encoder_provenance is None:
                encoder_provenance = current_provenance
            elif current_provenance != encoder_provenance:
                raise ValueError(f"encoder provenance differs for source {source}")
        full_features = load_features(path)
        sampled = sample_unlabeled(
            full_features, args.stage1_samples, stable_seed(source, args.sample_seed),
        )
        if args.source_weighting == "equal_source":
            sampled = sampled / np.sqrt(len(sampled))
        features[source] = sampled
        source_rows.append({
            "source": source,
            "domain": SOURCE_DOMAINS[source],
            "family": SOURCE_FAMILIES[source],
            "cache_rows": len(full_features),
            "stage1_rows": len(sampled),
            "feature_dim": sampled.shape[1],
            "stage1_total_squared_energy": float(np.sum(
                sampled.astype(np.float64) ** 2
            )),
            "cache_path": str(path),
            "cache_sha256": cache_metadata["computed_feature_file_sha256"],
            "cache_metadata_path": str(path.with_suffix(".json")),
            "cache_metadata_sha256": cache_metadata["metadata_file_sha256"],
            "cache_index_sha256": cache_metadata.get("computed_index_sha256", ""),
            "cache_sampling": cache_metadata.get("sampling", "unknown"),
            "label_independent_cache_sampling": cache_metadata[
                "label_independent_sampling"
            ],
        })

    names = sorted(features)
    stats_started = time.perf_counter()
    ranks, _, centroids, subspaces = classic.pool_statistics(features, args.top_k)
    gram_sketches = classic.pool_gram_sketches(features, args.top_k)
    for row in source_rows:
        summary = gram_sketches[str(row["source"])]
        row.update({
            "sketch_rank": summary.factor.shape[0],
            "tail_nuclear_bound": summary.tail_nuclear_bound,
            "tail_squared_bound": summary.tail_squared_bound,
            "tail_rank_bound": summary.tail_rank_bound,
        })
    subspace_similarity = classic.similarity_matrix(
        features, names, "subspace", centroids, subspaces,
    )
    summary_seconds = time.perf_counter() - stats_started

    max_shortlist = max(args.shortlist_sizes)
    method_sequences: dict[str, list[str]] = {}
    runtimes: dict[str, float] = {}

    def select(method: str, function) -> None:
        started = time.perf_counter()
        selected = function()
        runtimes[method] = time.perf_counter() - started
        classic.validate_selection(selected, names, max_shortlist)
        method_sequences[method] = selected

    select("rank_only", lambda: classic.rank_only(ranks, max_shortlist))
    rank_l_started = time.perf_counter()
    rank_l_order, rank_l_diagnostics = classic.rank_l_gram_greedy(
        gram_sketches, max_shortlist, return_diagnostics=True,
    )
    runtimes["rank_l_gram"] = time.perf_counter() - rank_l_started
    classic.validate_selection(rank_l_order, names, max_shortlist)
    method_sequences["rank_l_gram"] = rank_l_order
    select(
        "facility_subspace",
        lambda: classic.facility_location(subspace_similarity, names, max_shortlist),
    )
    select(
        "kcenter_subspace",
        lambda: classic.k_center(subspace_similarity, names, ranks, max_shortlist),
    )
    select(
        "dpp_subspace",
        lambda: classic.dpp_greedy(subspace_similarity, names, ranks, max_shortlist),
    )
    select(
        "pool_vendi_subspace",
        lambda: classic.pool_vendi_greedy(
            subspace_similarity, names, ranks, max_shortlist,
        ),
    )
    select(
        "agglomerative_subspace",
        lambda: classic.agglomerative_medoids(
            subspace_similarity, names, max_shortlist,
        ),
    )
    centroid_matrix = np.stack([centroids[name] for name in names])
    select(
        "leverage_centroid",
        lambda: classic.leverage_score_selection(
            centroid_matrix, names, max_shortlist,
        ),
    )
    exact_started = time.perf_counter()
    scatters = source_scatters(features)
    source_rows_by_name = {name: len(value) for name, value in features.items()}
    exact_order = full_rank_greedy_from_scatters(
        scatters,
        max_shortlist,
        source_rows=source_rows_by_name,
    )
    runtimes["exact_merged_rank_greedy"] = time.perf_counter() - exact_started
    classic.validate_selection(exact_order, names, max_shortlist)
    method_sequences["exact_merged_rank_greedy"] = exact_order

    rank_l_interval_diagnostics = []
    rank_l_scatter = np.zeros_like(next(iter(scatters.values())), dtype=np.float64)
    rank_l_rows = 0
    for step, diagnostic in enumerate(rank_l_diagnostics, 1):
        selected_name = rank_l_order[step - 1]
        rank_l_scatter = rank_l_scatter + scatters[selected_name]
        rank_l_rows += source_rows_by_name[selected_name]
        exact_reff = classic.effective_rank_from_scatter(
            rank_l_scatter,
            source_shape=(rank_l_rows, rank_l_scatter.shape[0]),
        )
        approximate_log_reff = float(np.log(max(
            float(diagnostic["approximate_effective_rank"]), 1.0,
        )))
        exact_log_reff = float(np.log(max(exact_reff, 1.0)))
        absolute_log_error = float(abs(approximate_log_reff - exact_log_reff))
        log_error_bound = float(diagnostic["log_error_bound"])
        rank_l_interval_diagnostics.append({
            **diagnostic,
            "exact_effective_rank_of_selected_prefix": exact_reff,
            "exact_log_effective_rank_of_selected_prefix": exact_log_reff,
            "absolute_log_error_of_selected_prefix": absolute_log_error,
            "selected_prefix_interval_covers_exact": bool(
                absolute_log_error <= log_error_bound + 1e-12
            ),
        })

    selections: dict[str, dict[str, list[str]]] = {}
    for shortlist_size in args.shortlist_sizes:
        current = {
            method: selected[:shortlist_size]
            for method, selected in method_sequences.items()
        }
        selections[str(shortlist_size)] = current
        print(
            f"L={shortlist_size} "
            + " ".join(f"{method}={','.join(value)}" for method, value in current.items()),
            flush=True,
        )

    dimension = next(iter(features.values())).shape[1]
    inventory = [
        {
            "method": "rank_only", "information_class": "scalar_summary",
            "bytes_per_pool": 4, "uses_labels": False, "uses_target_data": False,
        },
        {
            "method": "rank_l_gram", "information_class": "rank_l_psd_summary",
            "bytes_per_pool": 8 * (args.top_k * dimension + 3),
            "uses_labels": False, "uses_target_data": False,
        },
        *[
            {
                "method": method, "information_class": "top_k_subspace_summary",
                "bytes_per_pool": 4 * (args.top_k * dimension + 1),
                "uses_labels": False, "uses_target_data": False,
            }
            for method in [
                "facility_subspace", "kcenter_subspace", "dpp_subspace",
                "pool_vendi_subspace", "agglomerative_subspace",
            ]
        ],
        {
            "method": "leverage_centroid", "information_class": "centroid_summary",
            "bytes_per_pool": 4 * dimension,
            "uses_labels": False, "uses_target_data": False,
        },
        {
            "method": "exact_merged_rank_greedy", "information_class": "full_features",
            "bytes_per_pool": 4 * args.stage1_samples * dimension,
            "uses_labels": False, "uses_target_data": False,
        },
        {
            "method": "random", "information_class": "none",
            "bytes_per_pool": 0, "uses_labels": False, "uses_target_data": False,
        },
    ]
    with (args.out_dir / f"{args.encoder}_source_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    with (args.out_dir / f"{args.encoder}_method_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    manifest = {
        "created_utc": utc_now(),
        "pid": os.getpid(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classic_script_sha256": sha256_file(
            Path(__file__).with_name("two_stage_classic_baselines.py")
        ),
        "encoder": args.encoder,
        "source_dir": str(args.source_dir),
        "sources": args.sources,
        "source_domains": {
            source: SOURCE_DOMAINS[source] for source in args.sources
        },
        "source_families": {
            source: SOURCE_FAMILIES[source] for source in args.sources
        },
        "excluded_source_domains": args.exclude_source_domains,
        "targets": TARGETS,
        "cache_samples": args.cache_samples,
        "stage1_samples": args.stage1_samples,
        "sample_seed": args.sample_seed,
        "source_weighting": args.source_weighting,
        "marginal_scalar_mode": "exact_source_local",
        "stage1_reads_labels": False,
        "stage1_requires_label_independent_cache": not args.allow_label_informed_cache,
        "encoder_provenance": encoder_provenance,
        "top_k": args.top_k,
        "shortlist_sizes": args.shortlist_sizes,
        "summary_preprocessing_seconds": summary_seconds,
        "selections": selections,
        "selection_seconds_to_max_shortlist": runtimes,
        "rank_l_diagnostics": rank_l_interval_diagnostics,
        "rank_l_exact_greedy_prefix_match": {
            str(size): (
                method_sequences["rank_l_gram"][:size]
                == method_sequences["exact_merged_rank_greedy"][:size]
            )
            for size in args.shortlist_sizes
        },
        "source_cache_sha256": {
            str(row["source"]): str(row["cache_sha256"]) for row in source_rows
        },
        "source_cache_metadata_sha256": {
            str(row["source"]): str(row["cache_metadata_sha256"])
            for row in source_rows
        },
        "source_cache_index_sha256": {
            str(row["source"]): str(row["cache_index_sha256"])
            for row in source_rows
        },
    }
    path = args.out_dir / f"{args.encoder}_screening_manifest.json"
    write_json_atomic(path, manifest)
    print(f"wrote {path}", flush=True)


class Adapter(nn.Module):
    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width, bias=False),
            nn.LayerNorm(width),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def split_train_validation(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError(f"class {label} has fewer than two target training examples")
        cut = min(max(1, int(0.8 * len(indices))), len(indices) - 1)
        train_indices.extend(indices[:cut])
        validation_indices.extend(indices[cut:])
    return np.asarray(train_indices), np.asarray(validation_indices)


def stratified_kfold_indices(
    labels: np.ndarray, folds: int, seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("stratified K-fold requires at least two folds")
    rng = np.random.default_rng(seed)
    validation_parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < folds:
            raise ValueError(
                f"class {label} has {len(indices)} examples, fewer than {folds} folds"
            )
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, folds)):
            validation_parts[fold].append(part)

    all_indices = np.arange(len(labels), dtype=np.int64)
    splits = []
    for parts in validation_parts:
        validation = np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
        is_validation = np.zeros(len(labels), dtype=bool)
        is_validation[validation] = True
        train = all_indices[~is_validation]
        splits.append((train, validation))
    return splits


def target_validation_partitions(
    labels: np.ndarray, folds: int, seeds: list[int],
) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    partitions = []
    for seed in seeds:
        if folds == 1:
            splits = [split_train_validation(labels, seed)]
        else:
            splits = stratified_kfold_indices(labels, folds, seed)
        partitions.append((seed, splits))
    return partitions


def ridge_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_features: torch.Tensor,
    eval_labels: torch.Tensor,
    ridge: float,
) -> float:
    classes = int(train_labels.max().item()) + 1
    one_hot = torch.nn.functional.one_hot(train_labels, classes).float()
    if train_features.shape[0] < train_features.shape[1]:
        identity = torch.eye(train_features.shape[0], device=train_features.device)
        weights = train_features.T @ torch.linalg.solve(
            train_features @ train_features.T + ridge * identity, one_hot,
        )
    else:
        identity = torch.eye(train_features.shape[1], device=train_features.device)
        weights = torch.linalg.solve(
            train_features.T @ train_features + ridge * identity,
            train_features.T @ one_hot,
        )
    predictions = (eval_features @ weights).argmax(1)
    return float((predictions == eval_labels).float().mean().cpu())


def remap_target_labels(
    train_labels: np.ndarray, test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(train_labels)
    train = np.searchsorted(classes, train_labels)
    test = np.searchsorted(classes, test_labels)
    if np.any(test >= len(classes)):
        raise ValueError("target test split contains labels absent from train")
    if not np.array_equal(classes[test], test_labels):
        raise ValueError("target test split contains labels absent from train")
    return train.astype(np.int64), test.astype(np.int64)


def fit_adapter(
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    seed: int,
    width: int,
    steps: int,
    batch_size: int,
) -> Adapter:
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    adapter = Adapter(features.shape[1], width).to(device)
    head = nn.Linear(width, int(labels.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-4,
    )
    feature_tensor = torch.from_numpy(features).to(device)
    label_tensor = torch.from_numpy(labels).to(device)
    rng = np.random.default_rng(seed)
    adapter.train()
    for _ in range(steps):
        indices = torch.from_numpy(rng.integers(0, len(features), size=batch_size)).to(device)
        loss = torch.nn.functional.cross_entropy(
            head(adapter(feature_tensor[indices])), label_tensor[indices],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    adapter.eval()
    return adapter


def encode(adapter: Adapter, features: np.ndarray, device: torch.device) -> torch.Tensor:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), 2048):
            batch = torch.from_numpy(features[start : start + 2048]).to(device)
            outputs.append(adapter(batch))
    return torch.cat(outputs)


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_adapter_state(
    adapter: Adapter, output: Path, encoder: str, source: str, seed: int,
) -> tuple[str, str, str]:
    state_dir = output.parent / "adapter_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{encoder}__{source}__seed{seed}.pt"
    state_path = state_dir / filename
    temporary = state_path.with_suffix(".tmp.pt")
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()},
        temporary,
    )
    temporary.replace(state_path)
    relative = state_path.relative_to(output.parent).as_posix()
    return relative, sha256_file(state_path), module_state_sha256(adapter)


def adaptation_run_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.run.json")


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv_atomic(
    path: Path, fields: list[str], rows: list[dict[str, object] | dict[str, str]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def adaptation_provenance(
    args: argparse.Namespace,
    manifest: dict[str, object],
    assigned_sources: list[str],
) -> dict[str, object]:
    source_metadata = {
        source: stage2_cache_metadata(
            source_path(args.source_dir, args.encoder, source, args.cache_samples),
            expected_encoder=args.encoder,
            expected_dataset=source,
            expected_split="train",
            expected_samples=args.cache_samples,
            label_read_field="stage1_sampling_reads_labels",
        )
        for source in assigned_sources
    }
    target_metadata = {
        f"{target}:train": stage2_cache_metadata(
            target_path(args.target_dir, args.encoder, target, "train", args.cache_samples),
            expected_encoder=args.encoder,
            expected_dataset=target,
            expected_split="train",
            expected_samples=args.cache_samples,
            label_read_field="stage2_sampling_reads_labels",
        )
        for target in manifest["targets"]
    }
    all_metadata = [*source_metadata.values(), *target_metadata.values()]
    encoder_provenance = all_metadata[0]["encoder_provenance"]
    if any(
        metadata["encoder_provenance"] != encoder_provenance
        for metadata in all_metadata[1:]
    ):
        raise ValueError("Stage-2 source/target caches use different encoder provenance")
    if manifest.get("encoder_provenance") != encoder_provenance:
        raise ValueError("Stage-2 cache provenance does not match the Stage-1 manifest")
    source_hashes = {
        source: metadata["computed_feature_file_sha256"]
        for source, metadata in source_metadata.items()
    }
    target_hashes = {
        key: metadata["computed_feature_file_sha256"]
        for key, metadata in target_metadata.items()
    }
    configuration = {
        "schema_version": 4,
        "script_sha256": sha256_file(Path(__file__)),
        "screening_manifest_sha256": sha256_file(args.manifest),
        "encoder": args.encoder,
        "encoder_provenance": encoder_provenance,
        "assigned_sources": assigned_sources,
        "targets": manifest["targets"],
        "source_cache_sha256": source_hashes,
        "target_cache_sha256": target_hashes,
        "source_cache_metadata_sha256": {
            source: metadata["metadata_file_sha256"]
            for source, metadata in source_metadata.items()
        },
        "target_cache_metadata_sha256": {
            key: metadata["metadata_file_sha256"]
            for key, metadata in target_metadata.items()
        },
        "source_cache_index_sha256": {
            source: metadata["computed_index_sha256"]
            for source, metadata in source_metadata.items()
        },
        "target_cache_index_sha256": {
            key: metadata["computed_index_sha256"]
            for key, metadata in target_metadata.items()
        },
        "target_test_access": "forbidden_during_adaptation_and_selection",
        "cache_samples": args.cache_samples,
        "adapter_samples": args.adapter_samples,
        "allow_short_source": args.allow_short_source,
        "source_sample_seed": args.source_sample_seed,
        "target_cv_folds": args.target_cv_folds,
        "target_cv_seeds": args.target_cv_seeds,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "width": args.width,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "ridge": args.ridge,
        "adapter_state_format": "torch_state_dict_v1",
        "deterministic_algorithms": args.deterministic,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return {
        "created_utc": utc_now(),
        "run_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "configuration": configuration,
    }


def validate_resume_groups(
    completed: set[tuple[str, int, str]],
    sources: list[str],
    seeds: range,
    targets: set[str],
) -> None:
    allowed = {(source, seed) for source in sources for seed in seeds}
    observed: dict[tuple[str, int], set[str]] = {}
    for source, seed, target in completed:
        if (source, seed) not in allowed or target not in targets:
            raise ValueError(
                f"resume output contains a row outside this run: {(source, seed, target)}"
            )
        observed.setdefault((source, seed), set()).add(target)
    for key, completed_targets in observed.items():
        if completed_targets != targets:
            missing = sorted(targets - completed_targets)
            raise ValueError(
                f"partial adapter group {key}; refusing nondeterministic resume, missing {missing}"
            )


def evaluate_target_representation(
    encoded_train: torch.Tensor,
    data: dict[str, object],
    ridge: float,
    device: torch.device,
) -> dict[str, object]:
    train_labels_tensor = torch.from_numpy(data["train_labels"]).to(device)
    partition_accuracies: list[float] = []
    all_fold_accuracies: list[float] = []
    validation_examples = 0
    for _, folds in data["ridge_partitions"]:
        fold_accuracies: list[float] = []
        fold_sizes: list[int] = []
        for train_indices_np, validation_indices_np in folds:
            train_indices = torch.from_numpy(train_indices_np).to(device)
            validation_indices = torch.from_numpy(validation_indices_np).to(device)
            fold_accuracies.append(ridge_accuracy(
                encoded_train[train_indices],
                train_labels_tensor[train_indices],
                encoded_train[validation_indices],
                train_labels_tensor[validation_indices],
                ridge,
            ))
            fold_sizes.append(len(validation_indices_np))
        partition_accuracies.append(float(np.average(fold_accuracies, weights=fold_sizes)))
        all_fold_accuracies.extend(fold_accuracies)
        validation_examples += sum(fold_sizes)
    return {
        "validation_accuracy": float(np.mean(partition_accuracies)),
        "validation_accuracy_std": float(np.std(all_fold_accuracies)),
        "validation_fold_accuracies": "|".join(
            f"{value:.10f}" for value in all_fold_accuracies
        ),
        "validation_partition_accuracies": "|".join(
            f"{value:.10f}" for value in partition_accuracies
        ),
        "validation_accuracy_std_across_partitions": float(np.std(partition_accuracies)),
        "validation_examples": validation_examples,
        "unique_validation_examples": len(data["train_labels"]),
    }


def run_stage2_baselines(
    args: argparse.Namespace,
    targets: dict[str, dict[str, object]],
    device: torch.device,
    run_fingerprint: str,
) -> Path:
    """Evaluate source-independent controls under the identical target protocol."""

    output = args.output.with_name(f"{args.output.stem}_stage2_baselines.csv")
    expected_rows = len(targets) * (1 + args.n_seeds)
    if output.exists():
        with output.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
        if (
            len(existing) == expected_rows
            and {row.get("run_fingerprint") for row in existing} == {run_fingerprint}
        ):
            return output
        raise ValueError(f"Stage-2 baseline output exists with incompatible provenance: {output}")

    rows: list[dict[str, object]] = []
    for target, data in targets.items():
        started = time.perf_counter()
        identity_train = torch.from_numpy(data["train_features"]).to(device)
        metrics = evaluate_target_representation(
            identity_train, data, args.ridge, device,
        )
        torch.npu.synchronize()
        rows.append({
            "run_fingerprint": run_fingerprint,
            "encoder": args.encoder,
            "baseline": "frozen_identity",
            "adapter_seed": -1,
            "target": target,
            **metrics,
            "target_validation_protocol": data["validation_protocol"],
            "target_cv_folds": args.target_cv_folds,
            "target_cv_partitions": len(args.target_cv_seeds),
            "target_cv_seed": "|".join(map(str, args.target_cv_seeds)),
            "width": data["train_features"].shape[1],
            "ridge": args.ridge,
            "evaluation_seconds": time.perf_counter() - started,
            "device": str(device),
        })
        for adapter_seed in range(args.seed_start, args.seed_start + args.n_seeds):
            started = time.perf_counter()
            torch.manual_seed(adapter_seed)
            torch.npu.manual_seed_all(adapter_seed)
            np.random.seed(adapter_seed)
            random.seed(adapter_seed)
            adapter = Adapter(data["train_features"].shape[1], args.width).to(device)
            adapter.eval()
            encoded_train = encode(adapter, data["train_features"], device)
            metrics = evaluate_target_representation(
                encoded_train, data, args.ridge, device,
            )
            torch.npu.synchronize()
            rows.append({
                "run_fingerprint": run_fingerprint,
                "encoder": args.encoder,
                "baseline": "random_adapter",
                "adapter_seed": adapter_seed,
                "target": target,
                **metrics,
                "target_validation_protocol": data["validation_protocol"],
                "target_cv_folds": args.target_cv_folds,
                "target_cv_partitions": len(args.target_cv_seeds),
                "target_cv_seed": "|".join(map(str, args.target_cv_seeds)),
                "width": args.width,
                "ridge": args.ridge,
                "evaluation_seconds": time.perf_counter() - started,
                "device": str(device),
            })
    write_csv_atomic(output, list(rows[0]), rows)
    return output


def run_adapt(args: argparse.Namespace) -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("the adapt phase requires torch_npu on an Ascend host") from error

    manifest = json.loads(args.manifest.read_text())
    if manifest["encoder"] != args.encoder:
        raise ValueError(f"manifest encoder {manifest['encoder']} != {args.encoder}")
    sources = manifest["sources"]
    if args.shard_count > len(sources):
        raise ValueError(
            f"shard-count {args.shard_count} exceeds the {len(sources)} manifest sources"
        )
    assigned_sources = [
        source for index, source in enumerate(sources)
        if index % args.shard_count == args.shard_index
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_record = adaptation_provenance(args, manifest, assigned_sources)
    run_path = adaptation_run_path(args.output)
    existing_run = json.loads(run_path.read_text()) if run_path.exists() else None
    has_output = args.output.exists() and args.output.stat().st_size > 0
    if has_output and existing_run is None:
        raise ValueError(
            f"{args.output} has no provenance sidecar; use a new output path"
        )
    if existing_run is not None and (
        existing_run.get("run_fingerprint") != run_record["run_fingerprint"]
    ):
        raise ValueError("adaptation resume fingerprint does not match current inputs/config")
    if existing_run is None:
        write_json_atomic(run_path, run_record)

    targets: dict[str, dict[str, object]] = {}
    for target in manifest["targets"]:
        train_features, train_labels = load_cache(target_path(
            args.target_dir, args.encoder, target, "train", args.cache_samples,
        ))
        _, train_labels = np.unique(train_labels, return_inverse=True)
        train_labels = train_labels.astype(np.int64, copy=False)
        ridge_partitions = target_validation_partitions(
            train_labels, args.target_cv_folds, args.target_cv_seeds,
        )
        base_protocol = (
            "stratified_holdout_80_20" if args.target_cv_folds == 1
            else f"stratified_{args.target_cv_folds}_fold"
        )
        targets[target] = {
            "train_features": train_features,
            "train_labels": train_labels,
            "ridge_partitions": ridge_partitions,
            "validation_protocol": (
                base_protocol if len(args.target_cv_seeds) == 1
                else f"repeated_{base_protocol}"
            ),
        }

    fields = [
        "run_fingerprint", "encoder", "source", "source_domain", "adapter_seed", "target",
        "validation_accuracy", "validation_accuracy_std", "validation_fold_accuracies",
        "validation_partition_accuracies", "validation_accuracy_std_across_partitions",
        "validation_examples", "unique_validation_examples", "target_validation_protocol",
        "target_cv_folds", "target_cv_partitions", "target_cv_seed",
        "source_samples", "steps", "width", "batch_size", "ridge",
        "deterministic_algorithms", "adapter_state_path", "adapter_state_file_sha256",
        "adapter_state_sha256", "training_seconds", "evaluation_seconds", "device",
    ]
    existing_rows: list[dict[str, object] | dict[str, str]] = []
    if has_output:
        with args.output.open(newline="") as stream:
            existing_rows.extend(csv.DictReader(stream))
    completed = {
        (str(row["source"]), int(row["adapter_seed"]), str(row["target"]))
        for row in existing_rows
    }
    if len(completed) != len(existing_rows):
        raise ValueError("adaptation resume contains duplicate rows")
    if existing_rows:
        if (
            {row.get("run_fingerprint") for row in existing_rows}
            != {run_record["run_fingerprint"]}
            or {row.get("encoder") for row in existing_rows} != {args.encoder}
        ):
            raise ValueError("adaptation resume rows have incompatible provenance")
        validate_adaptation_row_protocol(
            existing_rows,
            run_record["configuration"],
            args.output,
        )
    seed_range = range(args.seed_start, args.seed_start + args.n_seeds)
    validate_resume_groups(completed, assigned_sources, seed_range, set(targets))

    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    if args.shard_index == 0 and not args.skip_stage2_baselines:
        baseline_path = run_stage2_baselines(
            args, targets, device, str(run_record["run_fingerprint"]),
        )
        print(f"wrote/validated {baseline_path}", flush=True)
    for source in assigned_sources:
        source_features, source_labels = load_cache(source_path(
            args.source_dir, args.encoder, source, args.cache_samples,
        ))
        if len(source_features) < args.adapter_samples and not args.allow_short_source:
            raise ValueError(
                f"source {source} has {len(source_features)} rows, below the fixed "
                f"Stage-2 budget {args.adapter_samples}"
            )
        source_indices = stratified_sample_indices(
            source_labels,
            min(args.adapter_samples, len(source_features)),
            stable_seed(source, args.source_sample_seed),
        )
        source_features = source_features[source_indices]
        source_labels = source_labels[source_indices]
        _, source_labels = np.unique(source_labels, return_inverse=True)
        source_labels = source_labels.astype(np.int64, copy=False)
        for adapter_seed in seed_range:
            if all((source, adapter_seed, target) in completed for target in targets):
                continue
            training_started = time.perf_counter()
            adapter = fit_adapter(
                source_features, source_labels, device, adapter_seed,
                args.width, args.steps, args.batch_size,
            )
            torch.npu.synchronize()
            training_seconds = time.perf_counter() - training_started
            state_path, state_file_sha256, state_sha256 = save_adapter_state(
                adapter, args.output, args.encoder, source, adapter_seed,
            )
            adapter_rows: list[dict[str, object]] = []
            for target, data in targets.items():
                evaluation_started = time.perf_counter()
                encoded_train = encode(adapter, data["train_features"], device)
                metrics = evaluate_target_representation(
                    encoded_train, data, args.ridge, device,
                )
                torch.npu.synchronize()
                adapter_rows.append({
                    "run_fingerprint": run_record["run_fingerprint"],
                    "encoder": args.encoder,
                    "source": source,
                    "source_domain": SOURCE_DOMAINS[source],
                    "adapter_seed": adapter_seed,
                    "target": target,
                    **metrics,
                    "target_validation_protocol": data["validation_protocol"],
                    "target_cv_folds": args.target_cv_folds,
                    "target_cv_partitions": len(args.target_cv_seeds),
                    "target_cv_seed": "|".join(map(str, args.target_cv_seeds)),
                    "source_samples": len(source_features),
                    "steps": args.steps,
                    "width": args.width,
                    "batch_size": args.batch_size,
                    "ridge": args.ridge,
                    "deterministic_algorithms": args.deterministic,
                    "adapter_state_path": state_path,
                    "adapter_state_file_sha256": state_file_sha256,
                    "adapter_state_sha256": state_sha256,
                    "training_seconds": training_seconds,
                    "evaluation_seconds": time.perf_counter() - evaluation_started,
                    "device": str(device),
                })
            existing_rows.extend(adapter_rows)
            write_csv_atomic(args.output, fields, existing_rows)
            completed.update((source, adapter_seed, target) for target in targets)
            print(
                f"source={source} seed={adapter_seed} train={training_seconds:.2f}s "
                f"targets={len(targets)}",
                flush=True,
            )


ADAPTATION_COMPATIBILITY_FIELDS = (
    "schema_version",
    "script_sha256",
    "screening_manifest_sha256",
    "encoder",
    "encoder_provenance",
    "targets",
    "target_cache_sha256",
    "target_cache_metadata_sha256",
    "target_cache_index_sha256",
    "target_test_access",
    "cache_samples",
    "adapter_samples",
    "allow_short_source",
    "source_sample_seed",
    "target_cv_folds",
    "target_cv_seeds",
    "seed_start",
    "n_seeds",
    "width",
    "steps",
    "batch_size",
    "ridge",
    "adapter_state_format",
    "deterministic_algorithms",
    "shard_count",
)


def _parse_accuracy_list(value: str, field: str, path: Path) -> list[float]:
    try:
        values = [float(item) for item in value.split("|") if item != ""]
    except ValueError as error:
        raise ValueError(f"invalid {field} in {path}") from error
    if not values or any(not np.isfinite(item) or not 0.0 <= item <= 1.0 for item in values):
        raise ValueError(f"invalid {field} in {path}")
    return values


def validate_adaptation_row_protocol(
    rows: list[dict[str, str]],
    configuration: dict[str, object],
    path: Path,
) -> None:
    """Bind CSV claims to the signed run configuration and basic metric invariants."""

    folds = int(configuration["target_cv_folds"])
    seeds = [int(seed) for seed in configuration["target_cv_seeds"]]
    partition_count = len(seeds)
    expected_seed_text = "|".join(map(str, seeds))
    base_protocol = "stratified_holdout_80_20" if folds == 1 else f"stratified_{folds}_fold"
    expected_protocol = (
        base_protocol if partition_count == 1 else f"repeated_{base_protocol}"
    )
    required_fields = {
        "target_validation_protocol", "target_cv_folds", "target_cv_partitions",
        "target_cv_seed", "validation_accuracy", "validation_accuracy_std",
        "validation_fold_accuracies", "validation_partition_accuracies",
        "validation_accuracy_std_across_partitions", "validation_examples",
        "unique_validation_examples", "source_samples", "steps",
        "width", "batch_size", "ridge", "deterministic_algorithms",
        "adapter_state_path", "adapter_state_file_sha256", "adapter_state_sha256",
        "training_seconds", "evaluation_seconds",
    }
    expected_static = {
        "target_validation_protocol": expected_protocol,
        "target_cv_folds": str(folds),
        "target_cv_partitions": str(partition_count),
        "target_cv_seed": expected_seed_text,
        "steps": str(configuration["steps"]),
        "width": str(configuration["width"]),
        "batch_size": str(configuration["batch_size"]),
        "deterministic_algorithms": str(configuration["deterministic_algorithms"]),
    }
    expected_samples = int(configuration["adapter_samples"])
    allow_short = bool(configuration["allow_short_source"])
    for row in rows:
        if row.get("test_accuracy", "") != "":
            raise ValueError(f"adaptation row accessed target test before selection: {path}")
        missing = sorted(field for field in required_fields if row.get(field, "") == "")
        if missing:
            raise ValueError(f"adaptation row lacks audited fields {missing}: {path}")
        state_relative = Path(row["adapter_state_path"])
        if state_relative.is_absolute() or ".." in state_relative.parts:
            raise ValueError(f"adapter state path escapes its run directory: {path}")
        state_path = path.parent / state_relative
        if (
            not state_path.is_file()
            or not _valid_sha256(row["adapter_state_file_sha256"])
            or sha256_file(state_path) != row["adapter_state_file_sha256"]
            or not _valid_sha256(row["adapter_state_sha256"])
        ):
            raise ValueError(f"adapter state provenance mismatch: {state_path}")
        mismatches = [
            field for field, expected in expected_static.items()
            if row[field] != expected
        ]
        if mismatches:
            raise ValueError(f"adaptation row/config mismatch {mismatches}: {path}")
        try:
            if not np.isclose(float(row["ridge"]), float(configuration["ridge"])):
                raise ValueError
            source_samples = int(row["source_samples"])
            validation = float(row["validation_accuracy"])
            validation_std = float(row["validation_accuracy_std"])
            partition_std = float(row["validation_accuracy_std_across_partitions"])
            validation_examples = int(row["validation_examples"])
            unique_examples = int(row["unique_validation_examples"])
            timings = [float(row["training_seconds"]), float(row["evaluation_seconds"])]
        except (TypeError, ValueError) as error:
            raise ValueError(f"adaptation row contains invalid numeric fields: {path}") from error
        if source_samples < 1 or source_samples > expected_samples:
            raise ValueError(f"adaptation source sample count exceeds its budget: {path}")
        if not allow_short and source_samples != expected_samples:
            raise ValueError(f"adaptation source sample count differs from its fixed budget: {path}")
        if not np.isfinite(validation) or not 0.0 <= validation <= 1.0:
            raise ValueError(f"adaptation accuracy is outside [0, 1]: {path}")
        if any(not np.isfinite(value) or value < 0.0 for value in (validation_std, partition_std, *timings)):
            raise ValueError(f"adaptation dispersion/timing field is invalid: {path}")
        fold_values = _parse_accuracy_list(
            row["validation_fold_accuracies"], "validation_fold_accuracies", path,
        )
        partition_values = _parse_accuracy_list(
            row["validation_partition_accuracies"],
            "validation_partition_accuracies", path,
        )
        if len(fold_values) != folds * partition_count:
            raise ValueError(f"adaptation fold count does not match configuration: {path}")
        if len(partition_values) != partition_count:
            raise ValueError(f"adaptation partition count does not match configuration: {path}")
        if not np.isclose(validation, np.mean(partition_values), atol=1e-8):
            raise ValueError(f"adaptation validation mean is inconsistent: {path}")
        if not np.isclose(validation_std, np.std(fold_values), atol=1e-8):
            raise ValueError(f"adaptation fold dispersion is inconsistent: {path}")
        if not np.isclose(partition_std, np.std(partition_values), atol=1e-8):
            raise ValueError(f"adaptation partition dispersion is inconsistent: {path}")
        if validation_examples < 1 or unique_examples < 1:
            raise ValueError(f"adaptation validation sample counts are invalid: {path}")
        if folds > 1 and validation_examples != unique_examples * partition_count:
            raise ValueError(f"cross-validation coverage is incomplete: {path}")
        if folds == 1 and validation_examples > unique_examples * partition_count:
            raise ValueError(f"holdout coverage exceeds the target split: {path}")


def read_adaptation(
    paths: list[Path],
    *,
    manifest: dict[str, object],
    manifest_path: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    configurations: list[dict[str, object]] = []
    observed_shards: set[int] = set()
    assigned_sources: set[str] = set()
    for path in paths:
        with path.open(newline="") as stream:
            current = list(csv.DictReader(stream))
        if not current:
            raise ValueError(f"adaptation input is empty: {path}")
        fingerprints = {row.get("run_fingerprint", "") for row in current}
        if len(fingerprints) != 1 or "" in fingerprints:
            raise ValueError(f"missing or mixed adaptation provenance in {path}")
        run_path = adaptation_run_path(path)
        if not run_path.exists():
            raise ValueError(f"adaptation provenance sidecar is missing: {run_path}")
        run_record = json.loads(run_path.read_text())
        configuration = run_record.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError(f"adaptation sidecar lacks configuration: {run_path}")
        canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        computed_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        recorded_fingerprint = run_record.get("run_fingerprint")
        if (
            recorded_fingerprint != computed_fingerprint
            or recorded_fingerprint != next(iter(fingerprints))
        ):
            raise ValueError(f"adaptation provenance mismatch: {path}")
        missing = [
            field for field in ADAPTATION_COMPATIBILITY_FIELDS
            if field not in configuration
        ]
        missing.extend(
            field for field in (
                "shard_index", "assigned_sources", "source_cache_sha256",
                "source_cache_metadata_sha256", "source_cache_index_sha256",
            )
            if field not in configuration
        )
        if missing:
            raise ValueError(f"adaptation sidecar lacks corrected fields: {missing}")
        if configuration["screening_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError(f"adaptation input uses a different screening manifest: {path}")
        if configuration["encoder"] != manifest["encoder"]:
            raise ValueError(f"adaptation encoder does not match manifest: {path}")
        if configuration["encoder_provenance"] != manifest.get("encoder_provenance"):
            raise ValueError(f"adaptation encoder provenance does not match manifest: {path}")
        if configuration["targets"] != manifest["targets"]:
            raise ValueError(f"adaptation targets do not match manifest: {path}")

        shard_index = int(configuration["shard_index"])
        shard_count = int(configuration["shard_count"])
        if not 0 <= shard_index < shard_count or shard_index in observed_shards:
            raise ValueError(f"invalid or duplicate adaptation shard: {shard_index}/{shard_count}")
        observed_shards.add(shard_index)
        current_sources = list(configuration.get("assigned_sources", []))
        if len(current_sources) != len(set(current_sources)):
            raise ValueError(f"duplicate assigned source in {run_path}")
        overlap = assigned_sources & set(current_sources)
        if overlap:
            raise ValueError(f"sources occur in multiple adaptation shards: {sorted(overlap)}")
        assigned_sources.update(current_sources)
        for key in (
            "source_cache_sha256", "source_cache_metadata_sha256",
            "source_cache_index_sha256",
        ):
            values = configuration.get(key, {})
            if (
                not isinstance(values, dict)
                or set(values) != set(current_sources)
                or any(not _valid_sha256(value) for value in values.values())
            ):
                raise ValueError(f"{key} does not cover assigned sources in {run_path}")

        expected_target_keys = {
            f"{target}:train" for target in manifest["targets"]
        }
        for key in (
            "target_cache_sha256", "target_cache_metadata_sha256",
            "target_cache_index_sha256",
        ):
            values = configuration.get(key, {})
            if (
                not isinstance(values, dict)
                or set(values) != expected_target_keys
                or any(not _valid_sha256(value) for value in values.values())
            ):
                raise ValueError(f"{key} does not cover target-train caches in {run_path}")

        expected_seeds = set(range(
            int(configuration["seed_start"]),
            int(configuration["seed_start"]) + int(configuration["n_seeds"]),
        ))
        expected_targets = set(manifest["targets"])
        expected_keys = {
            (source, seed, target)
            for source in current_sources
            for seed in expected_seeds
            for target in expected_targets
        }
        current_keys = {
            (row["source"], int(row["adapter_seed"]), row["target"])
            for row in current
        }
        if {row["encoder"] for row in current} != {manifest["encoder"]}:
            raise ValueError(f"adaptation rows use the wrong encoder: {path}")
        if current_keys != expected_keys or len(current) != len(expected_keys):
            raise ValueError(f"adaptation shard is incomplete or contains extra rows: {path}")
        validate_adaptation_row_protocol(current, configuration, path)
        configurations.append(configuration)
        rows.extend(current)

    reference = configurations[0]
    if reference["script_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("adaptation results were produced by a different script revision")
    for configuration in configurations[1:]:
        mismatches = [
            field for field in ADAPTATION_COMPATIBILITY_FIELDS
            if configuration[field] != reference[field]
        ]
        if mismatches:
            raise ValueError(f"adaptation shard configurations differ: {mismatches}")
    shard_count = int(reference["shard_count"])
    if observed_shards != set(range(shard_count)):
        raise ValueError(
            f"adaptation shards are incomplete: got {sorted(observed_shards)}, "
            f"expected {list(range(shard_count))}"
        )
    if assigned_sources != set(manifest["sources"]):
        raise ValueError("adaptation shards do not cover the manifest source set")
    keys = [(row["encoder"], row["source"], row["adapter_seed"], row["target"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate encoder/source/seed/target rows in adaptation inputs")
    return rows


def utility_tier(
    ranking: list[str],
    utility: dict[str, dict[str, float | int]],
    rank: int,
    tolerance: float,
) -> set[str]:
    if rank < 1 or rank > len(ranking):
        raise ValueError("rank must index the candidate ranking")
    threshold = float(utility[ranking[rank - 1]]["validation"])
    return {
        source for source in ranking
        if float(utility[source]["validation"]) >= threshold - tolerance
    }


def tolerance_column(tolerance: float) -> str:
    text = f"{tolerance:.12g}".replace("-", "m").replace(".", "p")
    return f"best_candidate_recall_tol_{text}"


def paired_bootstrap_confidence_tier(
    ranking: list[str],
    source_values: dict[str, list[tuple[float, int]]],
    confidence: float,
    replicates: int,
    seed: int,
) -> set[str]:
    """Return sources not detectably worse than the observed winner across shared seeds."""

    if not 0.0 < confidence < 1.0 or replicates < 1:
        raise ValueError("bootstrap confidence and replicate count are invalid")
    winner = ranking[0]
    winner_values = {seed_id: validation for validation, seed_id in source_values[winner]}
    result = {winner}
    rng = np.random.default_rng(seed)
    for source in ranking[1:]:
        candidate_values = {
            seed_id: validation for validation, seed_id in source_values[source]
        }
        if set(candidate_values) != set(winner_values):
            raise ValueError(f"adapter seeds differ between {winner} and {source}")
        paired_seeds = sorted(winner_values)
        if len(paired_seeds) < 2:
            result.add(source)
            continue
        differences = np.asarray([
            winner_values[seed_id] - candidate_values[seed_id]
            for seed_id in paired_seeds
        ])
        sample_indices = rng.integers(
            0, len(differences), size=(replicates, len(differences)),
        )
        bootstrap_means = differences[sample_indices].mean(axis=1)
        lower = float(np.quantile(bootstrap_means, (1.0 - confidence) / 2.0))
        if lower <= 0.0:
            result.add(source)
    return result


def adaptation_artifacts_for_selection(
    paths: list[Path], output_dir: Path,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    configurations = [
        json.loads(adaptation_run_path(path).read_text())["configuration"]
        for path in paths
    ]
    reference = configurations[0]
    source_maps: dict[str, dict[str, str]] = {
        field: {} for field in (
            "source_cache_sha256", "source_cache_metadata_sha256",
            "source_cache_index_sha256",
        )
    }
    for configuration in configurations:
        for field, merged in source_maps.items():
            for source, value in configuration[field].items():
                if source in merged and merged[source] != value:
                    raise ValueError(f"adaptation {field} differs for source {source}")
                merged[source] = value

    adapter_states: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = f"{row['source']}:{int(row['adapter_seed'])}"
                resolved = path.parent / row["adapter_state_path"]
                record = {
                    "path": os.path.relpath(resolved, output_dir),
                    "file_sha256": row["adapter_state_file_sha256"],
                    "state_sha256": row["adapter_state_sha256"],
                }
                if key in adapter_states and adapter_states[key] != record:
                    raise ValueError(f"adapter checkpoint differs across target rows: {key}")
                adapter_states[key] = record

    protocol_fields = (
        "schema_version", "script_sha256", "screening_manifest_sha256", "encoder",
        "encoder_provenance", "targets", "target_cache_sha256",
        "target_cache_metadata_sha256", "target_cache_index_sha256",
        "target_test_access", "cache_samples", "adapter_samples", "allow_short_source",
        "source_sample_seed", "target_cv_folds", "target_cv_seeds", "seed_start",
        "n_seeds", "width", "steps", "batch_size", "ridge",
        "adapter_state_format", "deterministic_algorithms",
    )
    protocol = {field: reference[field] for field in protocol_fields}
    protocol.update(source_maps)
    protocol["sources"] = sorted(source_maps["source_cache_sha256"])
    expected_state_keys = {
        f"{source}:{seed}"
        for source in protocol["sources"]
        for seed in range(
            int(reference["seed_start"]),
            int(reference["seed_start"]) + int(reference["n_seeds"]),
        )
    }
    if set(adapter_states) != expected_state_keys:
        raise ValueError("adapter checkpoints do not cover every source/seed pair")
    return protocol, adapter_states


def run_summarize(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    adaptation_manifest_path = getattr(args, "adaptation_manifest", None) or args.manifest
    adaptation_manifest = json.loads(adaptation_manifest_path.read_text())
    if (
        adaptation_manifest.get("encoder") != manifest.get("encoder")
        or adaptation_manifest.get("targets") != manifest.get("targets")
        or adaptation_manifest.get("encoder_provenance") != manifest.get("encoder_provenance")
    ):
        raise ValueError("adaptation and evaluation manifests are incompatible")
    if not set(manifest["sources"]).issubset(set(adaptation_manifest["sources"])):
        raise ValueError("evaluation sources are not a subset of the adaptation manifest")
    rows = read_adaptation(
        args.adaptation_csv,
        manifest=adaptation_manifest,
        manifest_path=adaptation_manifest_path,
    )
    adaptation_protocol, adapter_states = adaptation_artifacts_for_selection(
        args.adaptation_csv, args.out_dir,
    )
    manifest_sources = set(manifest["sources"])
    manifest_targets = set(manifest["targets"])
    equivalence_tolerances = sorted(set(
        float(value) for value in getattr(
            args, "equivalence_tolerances", [0.0, 0.001, 0.002, 0.005],
        )
    ))
    source_families = manifest.get(
        "source_families",
        {source: SOURCE_FAMILIES[source] for source in manifest_sources},
    )
    grouped: dict[tuple[str, str], dict[str, list[tuple[float, int]]]] = {}
    for row in rows:
        if (
            row["encoder"] != manifest["encoder"]
            or row["source"] not in manifest_sources
            or row["target"] not in manifest_targets
        ):
            continue
        key = row["encoder"], row["target"]
        grouped.setdefault(key, {}).setdefault(row["source"], []).append((
            float(row["validation_accuracy"]), int(row["adapter_seed"]),
        ))
    observed_targets = {target for _, target in grouped}
    if observed_targets != manifest_targets:
        missing = sorted(manifest_targets - observed_targets)
        extra = sorted(observed_targets - manifest_targets)
        raise ValueError(f"adaptation targets do not match manifest; missing={missing}, extra={extra}")

    detailed = []
    for (encoder, target), source_values in sorted(grouped.items()):
        if set(source_values) != set(manifest["sources"]):
            missing = sorted(set(manifest["sources"]) - set(source_values))
            raise ValueError(f"{encoder}/{target} missing exhaustive sources: {missing}")
        utility = {
            source: {
                "validation": float(np.mean([value[0] for value in values])),
                "seeds": len(values),
            }
            for source, values in source_values.items()
        }
        seed_counts = {value["seeds"] for value in utility.values()}
        if len(seed_counts) != 1:
            raise ValueError(f"{encoder}/{target} has unequal adapter seed counts: {seed_counts}")
        ranking = sorted(
            utility, key=lambda source: (-utility[source]["validation"], source),
        )
        oracle_source = ranking[0]
        oracle_validation = utility[oracle_source]["validation"]
        worst_validation = utility[ranking[-1]]["validation"]
        oracle_sources = utility_tier(ranking, utility, 1, args.tie_tolerance)
        equivalence_oracles = {
            tolerance: utility_tier(ranking, utility, 1, tolerance)
            for tolerance in equivalence_tolerances
        }
        confidence_oracles = paired_bootstrap_confidence_tier(
            ranking,
            source_values,
            float(getattr(args, "confidence_level", 0.95)),
            int(getattr(args, "bootstrap_replicates", 5_000)),
            stable_seed(
                f"confidence:{encoder}:{target}",
                int(getattr(args, "random_seed", 20260905)),
            ),
        )
        top_three = utility_tier(
            ranking, utility, min(3, len(ranking)), args.tie_tolerance,
        )
        oracle_families = {source_families[source] for source in oracle_sources}
        top_three_families = {source_families[source] for source in top_three}
        candidate_families = set(source_families.values())

        for shortlist_size in manifest["shortlist_sizes"]:
            selections = manifest["selections"][str(shortlist_size)]
            random_rng = np.random.default_rng(
                stable_seed(f"{encoder}:{target}:L{shortlist_size}", args.random_seed),
            )
            evaluated = [(method, -1, selected) for method, selected in selections.items()]
            for replicate in range(args.n_random):
                selected = sorted(random_rng.choice(
                    manifest["sources"], shortlist_size, replace=False,
                ).tolist())
                evaluated.append(("random", replicate, selected))

            for method, replicate, shortlist in evaluated:
                shortlist_families = {source_families[source] for source in shortlist}
                selected_source = min(
                    shortlist,
                    key=lambda source: (-utility[source]["validation"], source),
                )
                validation = utility[selected_source]["validation"]
                scale = max(oracle_validation - worst_validation, 1e-12)
                result = {
                    "encoder": encoder,
                    "target": target,
                    "method": method,
                    "replicate": replicate,
                    "shortlist_size": shortlist_size,
                    "candidate_count": len(manifest["sources"]),
                    "removed_fraction": 1.0 - shortlist_size / len(manifest["sources"]),
                    "best_candidate_recall": int(bool(oracle_sources & set(shortlist))),
                    "best_candidate_confidence_set_recall": int(bool(
                        confidence_oracles & set(shortlist)
                    )),
                    "top3_candidate_recall": len(top_three & set(shortlist)) / len(top_three),
                    "best_candidate_family_recall": int(bool(
                        oracle_families & shortlist_families
                    )),
                    "top3_candidate_family_recall": (
                        len(top_three_families & shortlist_families)
                        / len(top_three_families)
                    ),
                    "oracle_source": oracle_source,
                    "oracle_sources": "|".join(sorted(oracle_sources)),
                    "oracle_source_count": len(oracle_sources),
                    "oracle_confidence_sources": "|".join(sorted(confidence_oracles)),
                    "oracle_confidence_source_count": len(confidence_oracles),
                    "oracle_source_family": source_families[oracle_source],
                    "oracle_source_families": "|".join(sorted(oracle_families)),
                    "selected_source": selected_source,
                    "selected_source_family": source_families[selected_source],
                    "shortlist": "|".join(shortlist),
                    "shortlist_families": "|".join(sorted(shortlist_families)),
                    "candidate_family_count": len(candidate_families),
                    "selected_family_count": len(shortlist_families),
                    "removed_family_fraction": (
                        1.0 - len(shortlist_families) / len(candidate_families)
                    ),
                    "oracle_validation_accuracy": oracle_validation,
                    "selected_validation_accuracy": validation,
                    "validation_regret": oracle_validation - validation,
                    "normalized_validation_regret": (oracle_validation - validation) / scale,
                    "adapter_seeds": next(iter(seed_counts)),
                    "primary_tie_tolerance": args.tie_tolerance,
                    "oracle_validation_margin_to_second": (
                        oracle_validation - utility[ranking[1]]["validation"]
                        if len(ranking) > 1 else float("nan")
                    ),
                    "stage2_evaluations": shortlist_size,
                    "exhaustive_stage2_evaluations": len(manifest["sources"]),
                }
                for tolerance, equivalent_sources in equivalence_oracles.items():
                    result[tolerance_column(tolerance)] = int(bool(
                        equivalent_sources & set(shortlist)
                    ))
                detailed.append(result)

    if not detailed:
        raise ValueError("no adaptation rows matched the screening manifest")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / f"{manifest['encoder']}_shortlist_results.csv"
    with detail_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)

    grouped_results: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in detailed:
        grouped_results.setdefault((str(row["method"]), int(row["shortlist_size"])), []).append(row)
    summary_rows = []
    for (method, shortlist_size), values in sorted(grouped_results.items()):
        summary = {
            "encoder": manifest["encoder"],
            "method": method,
            "shortlist_size": shortlist_size,
            "rows": len(values),
            "targets": len(set(str(value["target"]) for value in values)),
            "best_candidate_recall_mean": float(np.mean([
                float(value["best_candidate_recall"]) for value in values
            ])),
            "best_candidate_confidence_set_recall_mean": float(np.mean([
                float(value["best_candidate_confidence_set_recall"]) for value in values
            ])),
            "top3_candidate_recall_mean": float(np.mean([
                float(value["top3_candidate_recall"]) for value in values
            ])),
            "best_candidate_family_recall_mean": float(np.mean([
                float(value["best_candidate_family_recall"]) for value in values
            ])),
            "top3_candidate_family_recall_mean": float(np.mean([
                float(value["top3_candidate_family_recall"]) for value in values
            ])),
            "validation_regret_mean": float(np.mean([
                float(value["validation_regret"]) for value in values
            ])),
            "normalized_validation_regret_mean": float(np.mean([
                float(value["normalized_validation_regret"]) for value in values
            ])),
            "removed_fraction": values[0]["removed_fraction"],
            "selected_family_count_mean": float(np.mean([
                float(value["selected_family_count"]) for value in values
            ])),
            "removed_family_fraction_mean": float(np.mean([
                float(value["removed_family_fraction"]) for value in values
            ])),
            "passes_90pct_recall_at_50pct_removal": bool(
                np.mean([float(value["best_candidate_recall"]) for value in values]) >= 0.9
                and float(values[0]["removed_fraction"]) >= 0.5
            ),
        }
        for tolerance in equivalence_tolerances:
            column = tolerance_column(tolerance)
            summary[f"{column}_mean"] = float(np.mean([
                float(value[column]) for value in values
            ]))
        summary_rows.append(summary)
    summary_path = args.out_dir / f"{manifest['encoder']}_shortlist_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_manifest = {
        "schema_version": 2,
        "created_utc": utc_now(),
        "encoder": manifest["encoder"],
        "script_sha256": sha256_file(Path(__file__)),
        "screening_manifest": str(args.manifest),
        "screening_manifest_sha256": sha256_file(args.manifest),
        "adaptation_manifest": str(adaptation_manifest_path),
        "adaptation_manifest_sha256": sha256_file(adaptation_manifest_path),
        "adaptation_inputs_sha256": {
            str(path): sha256_file(path) for path in args.adaptation_csv
        },
        "adaptation_sidecars_sha256": {
            str(adaptation_run_path(path)): sha256_file(adaptation_run_path(path))
            for path in args.adaptation_csv
        },
        "adaptation_protocol": adaptation_protocol,
        "adapter_states": adapter_states,
        "sources": manifest["sources"],
        "targets": manifest["targets"],
        "shortlist_sizes": manifest["shortlist_sizes"],
        "methods": sorted({str(row["method"]) for row in detailed}),
        "n_random": args.n_random,
        "random_seed": args.random_seed,
        "tie_tolerance": args.tie_tolerance,
        "equivalence_tolerances": equivalence_tolerances,
        "confidence_level": float(getattr(args, "confidence_level", 0.95)),
        "bootstrap_replicates": int(getattr(args, "bootstrap_replicates", 5_000)),
        "selection_metric": "mean_target_training_cv_accuracy",
        "selection_detail": detail_path.name,
        "target_test_used_for_selection": False,
        "target_test_read_during_adaptation_or_selection": False,
        "selection_frozen_before_target_test": True,
        "detail_rows": len(detailed),
        "detail_sha256": sha256_file(detail_path),
        "summary_rows": len(summary_rows),
        "summary_sha256": sha256_file(summary_path),
    }
    write_json_atomic(
        args.out_dir / f"{manifest['encoder']}_shortlist_manifest.json",
        summary_manifest,
    )
    print(json.dumps(summary_rows, indent=2), flush=True)


def heldout_manifest_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_manifest.json")


def validate_selection_freeze(path: Path) -> tuple[dict[str, object], Path, list[dict[str, str]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("selection manifest does not use the held-out isolation schema")
    if manifest.get("script_sha256") != sha256_file(Path(__file__)):
        raise ValueError("selection freeze was produced by a different script revision")
    if (
        manifest.get("target_test_used_for_selection") is not False
        or manifest.get("target_test_read_during_adaptation_or_selection") is not False
        or manifest.get("selection_frozen_before_target_test") is not True
    ):
        raise ValueError("selection manifest does not prove held-out target isolation")
    detail_name = manifest.get("selection_detail")
    if not isinstance(detail_name, str) or not detail_name:
        raise ValueError("selection manifest lacks its frozen detail file")
    detail_path = path.parent / detail_name
    if sha256_file(detail_path) != manifest.get("detail_sha256"):
        raise ValueError("frozen selection detail checksum mismatch")
    with detail_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest.get("detail_rows", -1)):
        raise ValueError("frozen selection detail row count mismatch")
    if not rows or any(
        row.get(field, "") != ""
        for row in rows
        for field in (
            "selected_test_accuracy", "oracle_selected_test_accuracy",
            "test_regret_vs_exhaustive_validation_selection",
        )
    ):
        raise ValueError("selection freeze already contains target-test metrics")
    protocol = manifest.get("adaptation_protocol")
    states = manifest.get("adapter_states")
    if not isinstance(protocol, dict) or not isinstance(states, dict):
        raise ValueError("selection manifest lacks adaptation/checkpoint provenance")
    if protocol.get("target_test_access") != "forbidden_during_adaptation_and_selection":
        raise ValueError("adaptation protocol did not forbid early target-test access")
    return manifest, detail_path, rows


def heldout_target_metadata(
    target_dir: Path,
    encoder: str,
    targets: list[str],
    cache_samples: int,
    expected_provenance: dict[str, object],
    expected_train_hashes: dict[str, str],
    expected_train_metadata_hashes: dict[str, str],
    expected_train_index_hashes: dict[str, str],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    train_metadata: dict[str, dict[str, object]] = {}
    test_metadata: dict[str, dict[str, object]] = {}
    for target in targets:
        train_key = f"{target}:train"
        train = stage2_cache_metadata(
            target_path(target_dir, encoder, target, "train", cache_samples),
            expected_encoder=encoder,
            expected_dataset=target,
            expected_split="train",
            expected_samples=cache_samples,
            label_read_field="stage2_sampling_reads_labels",
        )
        if (
            train["encoder_provenance"] != expected_provenance
            or train["computed_feature_file_sha256"] != expected_train_hashes.get(train_key)
            or train["metadata_file_sha256"] != expected_train_metadata_hashes.get(train_key)
            or train["computed_index_sha256"] != expected_train_index_hashes.get(train_key)
        ):
            raise ValueError(f"held-out target train cache changed after selection: {target}")
        test = stage2_cache_metadata(
            target_path(target_dir, encoder, target, "test", cache_samples),
            expected_encoder=encoder,
            expected_dataset=target,
            expected_split="test",
            expected_samples=cache_samples,
            label_read_field="stage2_sampling_reads_labels",
        )
        if test["encoder_provenance"] != expected_provenance:
            raise ValueError(f"held-out target test encoder differs for {target}")
        train_metadata[train_key] = train
        test_metadata[f"{target}:test"] = test
    return train_metadata, test_metadata


def load_frozen_adapter(
    checkpoint: Path,
    expected_file_sha256: str,
    expected_state_sha256: str,
    dimension: int,
    width: int,
    device: torch.device,
) -> Adapter:
    if sha256_file(checkpoint) != expected_file_sha256:
        raise ValueError(f"adapter checkpoint checksum mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"adapter checkpoint is not a state dict: {checkpoint}")
    adapter = Adapter(dimension, width)
    adapter.load_state_dict(state, strict=True)
    if module_state_sha256(adapter) != expected_state_sha256:
        raise ValueError(f"adapter parameter hash mismatch: {checkpoint}")
    adapter.eval().to(device)
    return adapter


def finite_metric(
    row: dict[str, object] | dict[str, str],
    field: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid held-out field {field}") from error
    if not np.isfinite(value):
        raise ValueError(f"non-finite held-out field {field}")
    if lower is not None and value < lower:
        raise ValueError(f"held-out field {field} is below {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"held-out field {field} is above {upper}")
    return value


def validate_heldout_baseline_resume(
    rows: list[dict[str, str]],
    expected_keys: set[tuple[str, int, str]],
    fingerprint: str,
    encoder: str,
) -> None:
    keys = {
        (row["baseline"], int(row["adapter_seed"]), row["target"])
        for row in rows
    }
    if keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("held-out baseline output has duplicate or unplanned rows")
    for row in rows:
        if row.get("run_fingerprint") != fingerprint or row.get("encoder") != encoder:
            raise ValueError("held-out baseline output has incompatible provenance")
        finite_metric(row, "test_accuracy", lower=0.0, upper=1.0)
        finite_metric(row, "evaluation_seconds", lower=0.0)
        if not row.get("device"):
            raise ValueError("held-out baseline output lacks its evaluation device")


def validate_heldout_evaluation_resume(
    rows: list[dict[str, str]],
    expected_keys: set[tuple[str, int, str]],
    fingerprint: str,
    encoder: str,
    adapter_states: dict[str, object],
    selection_dir: Path,
) -> set[tuple[str, int, str]]:
    completed = {
        (row["source"], int(row["adapter_seed"]), row["target"])
        for row in rows
    }
    if not completed.issubset(expected_keys) or len(completed) != len(rows):
        raise ValueError("held-out resume contains duplicate or unplanned evaluations")
    verified_checkpoints: set[Path] = set()
    for row in rows:
        if row.get("run_fingerprint") != fingerprint or row.get("encoder") != encoder:
            raise ValueError("held-out resume has incompatible provenance")
        finite_metric(row, "test_accuracy", lower=0.0, upper=1.0)
        finite_metric(row, "evaluation_seconds", lower=0.0)
        if not row.get("device"):
            raise ValueError("held-out resume lacks its evaluation device")
        state = adapter_states.get(f"{row['source']}:{int(row['adapter_seed'])}")
        if not isinstance(state, dict):
            raise ValueError("held-out resume references an unknown adapter state")
        expected_fields = {
            "adapter_state_path": state.get("path"),
            "adapter_state_file_sha256": state.get("file_sha256"),
            "adapter_state_sha256": state.get("state_sha256"),
        }
        if any(
            str(row.get(field, "")) != str(value)
            for field, value in expected_fields.items()
        ):
            raise ValueError("held-out resume does not match the frozen adapter state")
        checkpoint = selection_dir / str(state.get("path", ""))
        if checkpoint not in verified_checkpoints:
            if (
                not checkpoint.is_file()
                or sha256_file(checkpoint) != state.get("file_sha256")
            ):
                raise ValueError(f"frozen adapter checkpoint checksum mismatch: {checkpoint}")
            verified_checkpoints.add(checkpoint)
    return completed


def run_heldout_baselines(
    output: Path,
    encoder: str,
    targets: dict[str, dict[str, object]],
    protocol: dict[str, object],
    fingerprint: str,
    device: torch.device,
) -> Path:
    path = output.with_name(f"{encoder}_heldout_stage2_baselines.csv")
    fields = [
        "run_fingerprint", "encoder", "baseline", "adapter_seed", "target",
        "test_accuracy", "evaluation_seconds", "device",
    ]
    expected_keys = {
        ("frozen_identity", -1, target) for target in targets
    } | {
        ("random_adapter", seed, target)
        for seed in range(
            int(protocol["seed_start"]),
            int(protocol["seed_start"]) + int(protocol["n_seeds"]),
        )
        for target in targets
    }
    if path.exists():
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        validate_heldout_baseline_resume(rows, expected_keys, fingerprint, encoder)
        return path

    rows: list[dict[str, object]] = []
    for target, data in targets.items():
        train_labels = torch.from_numpy(data["train_labels"]).to(device)
        test_labels = torch.from_numpy(data["test_labels"]).to(device)
        started = time.perf_counter()
        identity_train = torch.from_numpy(data["train_features"]).to(device)
        identity_test = torch.from_numpy(data["test_features"]).to(device)
        identity_accuracy = ridge_accuracy(
            identity_train, train_labels, identity_test, test_labels,
            float(protocol["ridge"]),
        )
        torch.npu.synchronize()
        rows.append({
            "run_fingerprint": fingerprint,
            "encoder": encoder,
            "baseline": "frozen_identity",
            "adapter_seed": -1,
            "target": target,
            "test_accuracy": identity_accuracy,
            "evaluation_seconds": time.perf_counter() - started,
            "device": str(device),
        })
        for seed in range(
            int(protocol["seed_start"]),
            int(protocol["seed_start"]) + int(protocol["n_seeds"]),
        ):
            started = time.perf_counter()
            torch.manual_seed(seed)
            torch.npu.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            adapter = Adapter(data["train_features"].shape[1], int(protocol["width"]))
            adapter.eval().to(device)
            encoded_train = encode(adapter, data["train_features"], device)
            encoded_test = encode(adapter, data["test_features"], device)
            accuracy = ridge_accuracy(
                encoded_train, train_labels, encoded_test, test_labels,
                float(protocol["ridge"]),
            )
            torch.npu.synchronize()
            rows.append({
                "run_fingerprint": fingerprint,
                "encoder": encoder,
                "baseline": "random_adapter",
                "adapter_seed": seed,
                "target": target,
                "test_accuracy": accuracy,
                "evaluation_seconds": time.perf_counter() - started,
                "device": str(device),
            })
    write_csv_atomic(path, fields, rows)
    return path


def run_heldout(args: argparse.Namespace) -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("the heldout phase requires torch_npu on an Ascend host") from error

    selection, _, detail_rows = validate_selection_freeze(args.selection_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = selection["adaptation_protocol"]
    encoder = str(selection["encoder"])
    if encoder != args.encoder:
        raise ValueError(f"selection encoder {encoder} != {args.encoder}")
    targets = [str(target) for target in selection["targets"]]
    sources = set(str(source) for source in selection["sources"])
    evaluation_pairs = sorted({
        (row["selected_source"], row["target"])
        for row in detail_rows
    } | {
        (row["oracle_source"], row["target"])
        for row in detail_rows
    })
    if any(source not in sources or target not in targets for source, target in evaluation_pairs):
        raise ValueError("frozen selection references an unknown source or target")

    cache_samples = int(protocol["cache_samples"])
    train_metadata, test_metadata = heldout_target_metadata(
        args.target_dir,
        encoder,
        targets,
        cache_samples,
        protocol["encoder_provenance"],
        protocol["target_cache_sha256"],
        protocol["target_cache_metadata_sha256"],
        protocol["target_cache_index_sha256"],
    )
    configuration = {
        "schema_version": 1,
        "phase": "heldout_after_frozen_selection",
        "script_sha256": sha256_file(Path(__file__)),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "selection_detail_sha256": selection["detail_sha256"],
        "encoder": encoder,
        "encoder_provenance": protocol["encoder_provenance"],
        "targets": targets,
        "evaluation_pairs": evaluation_pairs,
        "adapter_states": selection["adapter_states"],
        "target_train_cache_sha256": {
            key: value["computed_feature_file_sha256"]
            for key, value in train_metadata.items()
        },
        "target_test_cache_sha256": {
            key: value["computed_feature_file_sha256"]
            for key, value in test_metadata.items()
        },
        "target_train_metadata_sha256": {
            key: value["metadata_file_sha256"] for key, value in train_metadata.items()
        },
        "target_test_metadata_sha256": {
            key: value["metadata_file_sha256"] for key, value in test_metadata.items()
        },
        "target_train_index_sha256": {
            key: value["computed_index_sha256"] for key, value in train_metadata.items()
        },
        "target_test_index_sha256": {
            key: value["computed_index_sha256"] for key, value in test_metadata.items()
        },
        "seed_start": protocol["seed_start"],
        "n_seeds": protocol["n_seeds"],
        "width": protocol["width"],
        "ridge": protocol["ridge"],
        "deterministic_algorithms": protocol["deterministic_algorithms"],
        "target_test_access": "after_selection_manifest_was_frozen",
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    run_path = adaptation_run_path(args.output)
    if run_path.exists():
        existing = json.loads(run_path.read_text())
        if existing.get("run_fingerprint") != fingerprint:
            raise ValueError("held-out output has incompatible provenance")
    else:
        write_json_atomic(run_path, {
            "created_utc": utc_now(),
            "run_fingerprint": fingerprint,
            "configuration": configuration,
        })

    target_data: dict[str, dict[str, object]] = {}
    for target in targets:
        train_features, train_labels = load_cache(target_path(
            args.target_dir, encoder, target, "train", cache_samples,
        ))
        test_features, test_labels = load_cache(target_path(
            args.target_dir, encoder, target, "test", cache_samples,
        ))
        train_labels, test_labels = remap_target_labels(train_labels, test_labels)
        target_data[target] = {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
        }

    seed_range = range(
        int(protocol["seed_start"]),
        int(protocol["seed_start"]) + int(protocol["n_seeds"]),
    )
    expected_keys = {
        (source, seed, target)
        for source, target in evaluation_pairs
        for seed in seed_range
    }
    raw_path = args.output.with_name(f"{args.output.stem}_evaluations.csv")
    raw_fields = [
        "run_fingerprint", "encoder", "source", "adapter_seed", "target",
        "test_accuracy", "adapter_state_path", "adapter_state_file_sha256",
        "adapter_state_sha256", "evaluation_seconds", "device",
    ]
    raw_rows: list[dict[str, object] | dict[str, str]] = []
    if raw_path.exists():
        with raw_path.open(newline="") as stream:
            raw_rows.extend(csv.DictReader(stream))
    completed = validate_heldout_evaluation_resume(
        raw_rows,
        expected_keys,
        fingerprint,
        encoder,
        selection["adapter_states"],
        args.selection_manifest.parent,
    )

    torch.use_deterministic_algorithms(
        bool(protocol["deterministic_algorithms"]), warn_only=False,
    )
    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    baseline_path = run_heldout_baselines(
        args.output, encoder, target_data, protocol, fingerprint, device,
    )
    pairs_by_source: dict[str, set[str]] = {}
    for source, target in evaluation_pairs:
        pairs_by_source.setdefault(source, set()).add(target)
    for source, source_targets in sorted(pairs_by_source.items()):
        for seed in seed_range:
            pending_targets = sorted(
                target for target in source_targets
                if (source, seed, target) not in completed
            )
            if not pending_targets:
                continue
            state_record = selection["adapter_states"].get(f"{source}:{seed}")
            if not isinstance(state_record, dict):
                raise ValueError(f"missing frozen adapter state for {source}/seed={seed}")
            checkpoint = args.selection_manifest.parent / state_record["path"]
            dimension = int(target_data[pending_targets[0]]["train_features"].shape[1])
            adapter = load_frozen_adapter(
                checkpoint,
                state_record["file_sha256"],
                state_record["state_sha256"],
                dimension,
                int(protocol["width"]),
                device,
            )
            group_rows = []
            for target in pending_targets:
                started = time.perf_counter()
                data = target_data[target]
                encoded_train = encode(adapter, data["train_features"], device)
                encoded_test = encode(adapter, data["test_features"], device)
                accuracy = ridge_accuracy(
                    encoded_train,
                    torch.from_numpy(data["train_labels"]).to(device),
                    encoded_test,
                    torch.from_numpy(data["test_labels"]).to(device),
                    float(protocol["ridge"]),
                )
                torch.npu.synchronize()
                group_rows.append({
                    "run_fingerprint": fingerprint,
                    "encoder": encoder,
                    "source": source,
                    "adapter_seed": seed,
                    "target": target,
                    "test_accuracy": accuracy,
                    "adapter_state_path": state_record["path"],
                    "adapter_state_file_sha256": state_record["file_sha256"],
                    "adapter_state_sha256": state_record["state_sha256"],
                    "evaluation_seconds": time.perf_counter() - started,
                    "device": str(device),
                })
            raw_rows.extend(group_rows)
            write_csv_atomic(raw_path, raw_fields, raw_rows)
            completed.update((source, seed, target) for target in pending_targets)
            print(
                f"heldout source={source} seed={seed} targets={len(pending_targets)}",
                flush=True,
            )
    if completed != expected_keys:
        raise ValueError("held-out evaluation did not cover every frozen selection")

    test_values: dict[tuple[str, str], list[float]] = {}
    for row in raw_rows:
        test_values.setdefault((row["source"], row["target"]), []).append(
            float(row["test_accuracy"])
        )
    enriched: list[dict[str, object]] = []
    for row in detail_rows:
        selected_values = test_values[(row["selected_source"], row["target"])]
        oracle_values = test_values[(row["oracle_source"], row["target"])]
        selected_test = float(np.mean(selected_values))
        oracle_test = float(np.mean(oracle_values))
        enriched.append({
            **row,
            "oracle_selected_test_accuracy": oracle_test,
            "selected_test_accuracy": selected_test,
            "test_regret_vs_exhaustive_validation_selection": oracle_test - selected_test,
            "heldout_adapter_seeds": len(selected_values),
        })
    write_csv_atomic(args.output, list(enriched[0]), enriched)
    heldout_manifest = {
        "created_utc": utc_now(),
        "encoder": encoder,
        "script_sha256": sha256_file(Path(__file__)),
        "selection_manifest": args.selection_manifest.name,
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "selection_detail_sha256": selection["detail_sha256"],
        "heldout_run_sidecar": run_path.name,
        "heldout_run_sidecar_sha256": sha256_file(run_path),
        "raw_evaluations": raw_path.name,
        "raw_evaluation_rows": len(raw_rows),
        "raw_evaluations_sha256": sha256_file(raw_path),
        "heldout_results": args.output.name,
        "heldout_result_rows": len(enriched),
        "heldout_results_sha256": sha256_file(args.output),
        "heldout_stage2_baselines": baseline_path.name,
        "heldout_stage2_baselines_sha256": sha256_file(baseline_path),
        "target_test_access": "after_selection_manifest_was_frozen",
        "target_test_used_for_selection": False,
        "test_evaluation_keys": len(expected_keys),
    }
    write_json_atomic(heldout_manifest_path(args.output), heldout_manifest)
    print(f"wrote {args.output} rows={len(enriched)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    screen = subparsers.add_parser("screen")
    screen.add_argument("--source-dir", type=Path, required=True)
    screen.add_argument("--out-dir", type=Path, required=True)
    screen.add_argument("--encoder", required=True)
    screen.add_argument("--cache-samples", type=int, default=5000)
    screen.add_argument("--stage1-samples", type=int, default=1000)
    screen.add_argument("--sample-seed", type=int, default=20260905)
    screen.add_argument("--top-k", type=int, default=20)
    screen.add_argument(
        "--source-weighting", choices=["equal_source", "equal_sample"],
        default="equal_source",
    )
    screen.add_argument("--allow-label-informed-cache", action="store_true")
    screen.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    screen.add_argument(
        "--exclude-source-domains", nargs="+",
        choices=sorted(set(SOURCE_DOMAINS.values())), default=[],
    )
    screen.add_argument(
        "--shortlist-sizes", nargs="+", type=int, default=SHORTLIST_SIZES,
    )

    adapt = subparsers.add_parser("adapt")
    adapt.add_argument("--manifest", type=Path, required=True)
    adapt.add_argument("--source-dir", type=Path, required=True)
    adapt.add_argument("--target-dir", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--encoder", required=True)
    adapt.add_argument("--npu", type=int, required=True)
    adapt.add_argument("--cache-samples", type=int, default=5000)
    adapt.add_argument("--adapter-samples", type=int, default=1000)
    adapt.add_argument("--allow-short-source", action="store_true")
    adapt.add_argument("--source-sample-seed", type=int, default=20260905)
    adapt.add_argument("--target-split-seed", type=int, default=20260905)
    adapt.add_argument("--target-cv-folds", type=int, default=1)
    adapt.add_argument("--target-cv-seeds", nargs="+", type=int)
    adapt.add_argument("--seed-start", type=int, default=0)
    adapt.add_argument("--n-seeds", type=int, default=5)
    adapt.add_argument("--width", type=int, default=256)
    adapt.add_argument("--steps", type=int, default=600)
    adapt.add_argument("--batch-size", type=int, default=128)
    adapt.add_argument("--ridge", type=float, default=1e-2)
    adapt.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True,
    )
    adapt.add_argument("--skip-stage2-baselines", action="store_true")
    adapt.add_argument("--shard-index", type=int, default=0)
    adapt.add_argument("--shard-count", type=int, default=1)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument(
        "--adaptation-manifest",
        type=Path,
        help="Manifest used to launch adaptation; defaults to --manifest.",
    )
    summarize.add_argument("--adaptation-csv", nargs="+", type=Path, required=True)
    summarize.add_argument("--out-dir", type=Path, required=True)
    summarize.add_argument("--n-random", type=int, default=100)
    summarize.add_argument("--random-seed", type=int, default=20260905)
    summarize.add_argument("--tie-tolerance", type=float, default=1e-12)
    summarize.add_argument(
        "--equivalence-tolerances", nargs="+", type=float,
        default=[0.0, 0.001, 0.002, 0.005],
    )
    summarize.add_argument("--confidence-level", type=float, default=0.95)
    summarize.add_argument("--bootstrap-replicates", type=int, default=5_000)

    heldout = subparsers.add_parser("heldout")
    heldout.add_argument("--selection-manifest", type=Path, required=True)
    heldout.add_argument("--target-dir", type=Path, required=True)
    heldout.add_argument("--output", type=Path, required=True)
    heldout.add_argument("--encoder", required=True)
    heldout.add_argument("--npu", type=int, required=True)

    args = parser.parse_args()
    if args.phase == "screen":
        if len(args.sources) != len(set(args.sources)):
            parser.error("screen sources must be unique")
        args.sources = [
            source for source in args.sources
            if SOURCE_DOMAINS[source] not in set(args.exclude_source_domains)
        ]
        if not args.sources:
            parser.error("source-domain exclusions removed every source")
        if any(size <= 0 or size > len(args.sources) for size in args.shortlist_sizes):
            parser.error(f"shortlist sizes must be in [1, {len(args.sources)}]")
    if args.phase == "adapt":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            parser.error("shard-index must be in [0, shard-count)")
        if args.n_seeds < 1 or args.adapter_samples < 1 or args.target_cv_folds < 1:
            parser.error("n-seeds, adapter-samples, and target-cv-folds must be positive")
        if args.target_cv_seeds is None:
            args.target_cv_seeds = [args.target_split_seed]
        if len(args.target_cv_seeds) != len(set(args.target_cv_seeds)):
            parser.error("target-cv-seeds must be unique")
    if args.phase == "summarize":
        if args.tie_tolerance < 0 or any(value < 0 for value in args.equivalence_tolerances):
            parser.error("tie/equivalence tolerances must be non-negative")
        if not 0 < args.confidence_level < 1 or args.bootstrap_replicates < 1:
            parser.error("bootstrap confidence must be in (0, 1) and replicates positive")
    return args


def main() -> None:
    args = parse_args()
    if args.phase == "screen":
        run_screen(args)
    elif args.phase == "adapt":
        run_adapt(args)
    elif args.phase == "summarize":
        run_summarize(args)
    else:
        run_heldout(args)


if __name__ == "__main__":
    main()
