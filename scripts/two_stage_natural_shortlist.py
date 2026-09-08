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
import torch_npu  # noqa: F401 - registers the NPU backend

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
    eigenvalues = np.maximum(np.linalg.eigvalsh(scatter.astype(np.float64, copy=False)), 0.0)
    singular = np.sqrt(eigenvalues[eigenvalues > 1e-12])
    if singular.size == 0:
        return 0.0
    probabilities = singular / singular.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def full_rank_greedy(features: dict[str, np.ndarray], budget: int) -> list[str]:
    scatters = {name: value.T @ value for name, value in features.items()}
    selected: list[str] = []
    current = np.zeros_like(next(iter(scatters.values())))
    while len(selected) < budget:
        remaining = sorted(set(features) - set(selected))
        choice = max(
            remaining,
            key=lambda name: (effective_rank_from_scatter(current + scatters[name]), name),
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
        full_features, _ = load_cache(path)
        sampled = sample_unlabeled(
            full_features, args.stage1_samples, stable_seed(source, args.sample_seed),
        )
        features[source] = sampled
        source_rows.append({
            "source": source,
            "domain": SOURCE_DOMAINS[source],
            "family": SOURCE_FAMILIES[source],
            "cache_rows": len(full_features),
            "stage1_rows": len(sampled),
            "feature_dim": sampled.shape[1],
            "cache_path": str(path),
        })

    names = sorted(features)
    stats_started = time.perf_counter()
    ranks, _, centroids, subspaces = classic.pool_statistics(features, args.top_k)
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
    select("full_merged_rank", lambda: full_rank_greedy(features, max_shortlist))

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
        *[
            {
                "method": method, "information_class": "top_k_subspace_summary",
                "bytes_per_pool": 4 * (args.top_k * dimension + 1),
                "uses_labels": False, "uses_target_data": False,
            }
            for method in [
                "facility_subspace", "kcenter_subspace", "dpp_subspace",
                "pool_vendi_subspace",
            ]
        ],
        {
            "method": "full_merged_rank", "information_class": "full_features",
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
        "top_k": args.top_k,
        "shortlist_sizes": args.shortlist_sizes,
        "summary_preprocessing_seconds": summary_seconds,
        "selections": selections,
        "selection_seconds_to_max_shortlist": runtimes,
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


def run_adapt(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if manifest["encoder"] != args.encoder:
        raise ValueError(f"manifest encoder {manifest['encoder']} != {args.encoder}")
    sources = manifest["sources"]
    assigned_sources = [
        source for index, source in enumerate(sources)
        if index % args.shard_count == args.shard_index
    ]
    targets = {}
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
        if args.target_cv_folds == 1:
            base_protocol = "stratified_holdout_80_20"
        else:
            base_protocol = f"stratified_{args.target_cv_folds}_fold"
        validation_protocol = (
            base_protocol if len(args.target_cv_seeds) == 1
            else f"repeated_{base_protocol}"
        )
        targets[target] = {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "ridge_partitions": ridge_partitions,
            "validation_protocol": validation_protocol,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output)
    fields = [
        "encoder", "source", "source_domain", "adapter_seed", "target",
        "validation_accuracy", "validation_accuracy_std", "validation_fold_accuracies",
        "validation_partition_accuracies", "validation_accuracy_std_across_partitions",
        "validation_examples", "unique_validation_examples", "target_validation_protocol",
        "target_cv_folds", "target_cv_partitions", "target_cv_seed", "test_accuracy",
        "source_samples", "steps", "width",
        "batch_size", "ridge", "training_seconds", "evaluation_seconds", "device",
    ]
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    with args.output.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
            stream.flush()
        for source in assigned_sources:
            source_features, source_labels = load_cache(source_path(
                args.source_dir, args.encoder, source, args.cache_samples,
            ))
            source_indices = stratified_sample_indices(
                source_labels,
                min(args.adapter_samples, len(source_features)),
                stable_seed(source, args.source_sample_seed),
            )
            source_features = source_features[source_indices]
            source_labels = source_labels[source_indices]
            _, source_labels = np.unique(source_labels, return_inverse=True)
            source_labels = source_labels.astype(np.int64, copy=False)
            for adapter_seed in range(args.seed_start, args.seed_start + args.n_seeds):
                pending_targets = [
                    target for target in targets
                    if (source, adapter_seed, target) not in completed
                ]
                if not pending_targets:
                    continue
                training_started = time.perf_counter()
                adapter = fit_adapter(
                    source_features, source_labels, device, adapter_seed,
                    args.width, args.steps, args.batch_size,
                )
                torch.npu.synchronize()
                training_seconds = time.perf_counter() - training_started
                for target in pending_targets:
                    evaluation_started = time.perf_counter()
                    data = targets[target]
                    encoded_train = encode(adapter, data["train_features"], device)
                    encoded_test = encode(adapter, data["test_features"], device)
                    train_labels_tensor = torch.from_numpy(data["train_labels"]).to(device)
                    test_labels_tensor = torch.from_numpy(data["test_labels"]).to(device)
                    partition_accuracies = []
                    all_fold_accuracies = []
                    validation_examples = 0
                    for _, folds in data["ridge_partitions"]:
                        fold_accuracies = []
                        fold_sizes = []
                        for train_indices_np, validation_indices_np in folds:
                            train_indices = torch.from_numpy(train_indices_np).to(device)
                            validation_indices = torch.from_numpy(validation_indices_np).to(device)
                            fold_accuracies.append(ridge_accuracy(
                                encoded_train[train_indices],
                                train_labels_tensor[train_indices],
                                encoded_train[validation_indices],
                                train_labels_tensor[validation_indices],
                                args.ridge,
                            ))
                            fold_sizes.append(len(validation_indices_np))
                        partition_accuracies.append(float(np.average(
                            fold_accuracies, weights=fold_sizes,
                        )))
                        all_fold_accuracies.extend(fold_accuracies)
                        validation_examples += sum(fold_sizes)
                    validation_accuracy = float(np.mean(partition_accuracies))
                    test_accuracy = ridge_accuracy(
                        encoded_train, train_labels_tensor,
                        encoded_test, test_labels_tensor,
                        args.ridge,
                    )
                    torch.npu.synchronize()
                    evaluation_seconds = time.perf_counter() - evaluation_started
                    writer.writerow({
                        "encoder": args.encoder,
                        "source": source,
                        "source_domain": SOURCE_DOMAINS[source],
                        "adapter_seed": adapter_seed,
                        "target": target,
                        "validation_accuracy": validation_accuracy,
                        "validation_accuracy_std": float(np.std(all_fold_accuracies)),
                        "validation_fold_accuracies": "|".join(
                            f"{value:.10f}" for value in all_fold_accuracies
                        ),
                        "validation_partition_accuracies": "|".join(
                            f"{value:.10f}" for value in partition_accuracies
                        ),
                        "validation_accuracy_std_across_partitions": float(
                            np.std(partition_accuracies)
                        ),
                        "validation_examples": validation_examples,
                        "unique_validation_examples": len(data["train_labels"]),
                        "target_validation_protocol": data["validation_protocol"],
                        "target_cv_folds": args.target_cv_folds,
                        "target_cv_partitions": len(args.target_cv_seeds),
                        "target_cv_seed": "|".join(map(str, args.target_cv_seeds)),
                        "test_accuracy": test_accuracy,
                        "source_samples": len(source_features),
                        "steps": args.steps,
                        "width": args.width,
                        "batch_size": args.batch_size,
                        "ridge": args.ridge,
                        "training_seconds": training_seconds,
                        "evaluation_seconds": evaluation_seconds,
                        "device": str(device),
                    })
                    stream.flush()
                    completed.add((source, adapter_seed, target))
                print(
                    f"source={source} seed={adapter_seed} train={training_seconds:.2f}s "
                    f"targets={len(pending_targets)}",
                    flush=True,
                )


def read_adaptation(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
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


def run_summarize(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    rows = read_adaptation(args.adaptation_csv)
    manifest_sources = set(manifest["sources"])
    source_families = manifest.get(
        "source_families",
        {source: SOURCE_FAMILIES[source] for source in manifest_sources},
    )
    grouped: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        if (
            row["encoder"] != manifest["encoder"]
            or row["source"] not in manifest_sources
        ):
            continue
        key = row["encoder"], row["target"]
        grouped.setdefault(key, {}).setdefault(row["source"], []).append((
            float(row["validation_accuracy"]), float(row["test_accuracy"]),
        ))

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
                detailed.append({
                    "encoder": encoder,
                    "target": target,
                    "method": method,
                    "replicate": replicate,
                    "shortlist_size": shortlist_size,
                    "candidate_count": len(manifest["sources"]),
                    "removed_fraction": 1.0 - shortlist_size / len(manifest["sources"]),
                    "best_candidate_recall": int(bool(oracle_sources & set(shortlist))),
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
                    "stage2_evaluations": shortlist_size,
                    "exhaustive_stage2_evaluations": len(manifest["sources"]),
                })

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
        summary_rows.append({
            "encoder": manifest["encoder"],
            "method": method,
            "shortlist_size": shortlist_size,
            "rows": len(values),
            "targets": len(set(str(value["target"]) for value in values)),
            "best_candidate_recall_mean": float(np.mean([
                float(value["best_candidate_recall"]) for value in values
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
        })
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
    adapt.add_argument("--shard-index", type=int, default=0)
    adapt.add_argument("--shard-count", type=int, default=1)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.add_argument("--adaptation-csv", nargs="+", type=Path, required=True)
    summarize.add_argument("--out-dir", type=Path, required=True)
    summarize.add_argument("--n-random", type=int, default=100)
    summarize.add_argument("--random-seed", type=int, default=20260905)
    summarize.add_argument("--tie-tolerance", type=float, default=1e-12)

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
