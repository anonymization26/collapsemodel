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


def stage1_cache_metadata(path: Path, allow_label_informed: bool) -> dict[str, object]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        if allow_label_informed:
            return {"sampling": "unverified", "label_independent_sampling": False}
        raise ValueError(f"Stage-1 cache has no provenance metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    sampling = str(metadata.get("sampling", "unknown"))
    requested = int(metadata.get("requested_samples", -1))
    source_samples = int(metadata.get("source_samples", -1))
    uses_all_rows = (
        "requested_samples" in metadata
        and (
            requested <= 0
            or (source_samples >= 0 and requested >= source_samples)
        )
    )
    label_independent = sampling == "unlabeled_random" or uses_all_rows
    if not label_independent and not allow_label_informed:
        raise ValueError(
            f"Stage-1 cache {path} used label-informed or unknown sampling={sampling!r}; "
            "regenerate it with --sampling unlabeled_random"
        )
    return {**metadata, "label_independent_sampling": label_independent}


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


def full_rank_greedy(features: dict[str, np.ndarray], budget: int) -> list[str]:
    scatters = {
        name: value.astype(np.float64, copy=False).T @ value.astype(np.float64, copy=False)
        for name, value in features.items()
    }
    selected: list[str] = []
    current = np.zeros_like(next(iter(scatters.values())), dtype=np.float64)
    while len(selected) < budget:
        remaining = sorted(set(features) - set(selected))
        choice = max(
            remaining,
            key=lambda name: effective_rank_from_scatter(current + scatters[name]),
        )
        selected.append(choice)
        current = current + scatters[choice]
    return selected


def run_screen(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = {}
    source_rows = []
    for source in args.sources:
        path = source_path(args.source_dir, args.encoder, source, args.cache_samples)
        cache_metadata = stage1_cache_metadata(path, args.allow_label_informed_cache)
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
            "cache_sha256": sha256_file(path),
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
    select(
        "rank_l_gram",
        lambda: classic.rank_l_gram_greedy(gram_sketches, max_shortlist),
    )
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
    select(
        "exact_merged_rank_greedy",
        lambda: full_rank_greedy(features, max_shortlist),
    )

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
        "stage1_reads_labels": False,
        "stage1_requires_label_independent_cache": not args.allow_label_informed_cache,
        "top_k": args.top_k,
        "shortlist_sizes": args.shortlist_sizes,
        "summary_preprocessing_seconds": summary_seconds,
        "selections": selections,
        "selection_seconds_to_max_shortlist": runtimes,
        "source_cache_sha256": {
            str(row["source"]): str(row["cache_sha256"]) for row in source_rows
        },
    }
    path = args.out_dir / f"{args.encoder}_screening_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
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


def read_completed(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {
            (row["source"], int(row["adapter_seed"]), row["target"])
            for row in csv.DictReader(stream)
        }


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
    source_hashes = {
        source: sha256_file(source_path(
            args.source_dir, args.encoder, source, args.cache_samples,
        ))
        for source in assigned_sources
    }
    target_hashes = {
        f"{target}:{split}": sha256_file(target_path(
            args.target_dir, args.encoder, target, split, args.cache_samples,
        ))
        for target in manifest["targets"]
        for split in ("train", "test")
    }
    configuration = {
        "schema_version": 2,
        "script_sha256": sha256_file(Path(__file__)),
        "screening_manifest_sha256": sha256_file(args.manifest),
        "encoder": args.encoder,
        "assigned_sources": assigned_sources,
        "targets": manifest["targets"],
        "source_cache_sha256": source_hashes,
        "target_cache_sha256": target_hashes,
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
    encoded_test: torch.Tensor,
    data: dict[str, object],
    ridge: float,
    device: torch.device,
) -> dict[str, object]:
    train_labels_tensor = torch.from_numpy(data["train_labels"]).to(device)
    test_labels_tensor = torch.from_numpy(data["test_labels"]).to(device)
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
        "test_accuracy": ridge_accuracy(
            encoded_train, train_labels_tensor, encoded_test, test_labels_tensor, ridge,
        ),
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
        identity_test = torch.from_numpy(data["test_features"]).to(device)
        metrics = evaluate_target_representation(
            identity_train, identity_test, data, args.ridge, device,
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
            encoded_test = encode(adapter, data["test_features"], device)
            metrics = evaluate_target_representation(
                encoded_train, encoded_test, data, args.ridge, device,
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
        test_features, test_labels = load_cache(target_path(
            args.target_dir, args.encoder, target, "test", args.cache_samples,
        ))
        train_labels, test_labels = remap_target_labels(train_labels, test_labels)
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
            "test_features": test_features,
            "test_labels": test_labels,
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
        "target_cv_folds", "target_cv_partitions", "target_cv_seed", "test_accuracy",
        "source_samples", "steps", "width", "batch_size", "ridge",
        "deterministic_algorithms", "training_seconds", "evaluation_seconds", "device",
    ]
    existing_rows: list[dict[str, object] | dict[str, str]] = []
    if has_output:
        with args.output.open(newline="") as stream:
            existing_rows.extend(csv.DictReader(stream))
    completed = read_completed(args.output)
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
            adapter_rows: list[dict[str, object]] = []
            for target, data in targets.items():
                evaluation_started = time.perf_counter()
                encoded_train = encode(adapter, data["train_features"], device)
                encoded_test = encode(adapter, data["test_features"], device)
                metrics = evaluate_target_representation(
                    encoded_train, encoded_test, data, args.ridge, device,
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


def read_adaptation(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            current = list(csv.DictReader(stream))
        fingerprints = {row.get("run_fingerprint", "") for row in current}
        fingerprints.discard("")
        if fingerprints:
            if len(fingerprints) != 1 or any(not row.get("run_fingerprint") for row in current):
                raise ValueError(f"mixed adaptation provenance in {path}")
            run_path = adaptation_run_path(path)
            if not run_path.exists():
                raise ValueError(f"adaptation provenance sidecar is missing: {run_path}")
            recorded = json.loads(run_path.read_text()).get("run_fingerprint")
            if recorded != next(iter(fingerprints)):
                raise ValueError(f"adaptation provenance mismatch: {path}")
        rows.extend(current)
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
    source_values: dict[str, list[tuple[float, float, int]]],
    confidence: float,
    replicates: int,
    seed: int,
) -> set[str]:
    """Return sources not detectably worse than the observed winner across shared seeds."""

    if not 0.0 < confidence < 1.0 or replicates < 1:
        raise ValueError("bootstrap confidence and replicate count are invalid")
    winner = ranking[0]
    winner_values = {seed_id: validation for validation, _, seed_id in source_values[winner]}
    result = {winner}
    rng = np.random.default_rng(seed)
    for source in ranking[1:]:
        candidate_values = {
            seed_id: validation for validation, _, seed_id in source_values[source]
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


def run_summarize(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    rows = read_adaptation(args.adaptation_csv)
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
    grouped: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        if (
            row["encoder"] != manifest["encoder"]
            or row["source"] not in manifest_sources
            or row["target"] not in manifest_targets
        ):
            continue
        key = row["encoder"], row["target"]
        grouped.setdefault(key, {}).setdefault(row["source"], []).append((
            float(row["validation_accuracy"]), float(row["test_accuracy"]),
            int(row["adapter_seed"]),
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
                "test": float(np.mean([value[1] for value in values])),
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
        oracle_test = utility[oracle_source]["test"]
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
                test = utility[selected_source]["test"]
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
                    "oracle_selected_test_accuracy": oracle_test,
                    "selected_test_accuracy": test,
                    "test_regret_vs_exhaustive_validation_selection": oracle_test - test,
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
            "test_regret_mean": float(np.mean([
                float(value["test_regret_vs_exhaustive_validation_selection"]) for value in values
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
    print(json.dumps(summary_rows, indent=2), flush=True)


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
    else:
        run_summarize(args)


if __name__ == "__main__":
    main()
