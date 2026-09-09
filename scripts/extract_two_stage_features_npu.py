#!/usr/bin/env python3
"""Extract auditable frozen features with each encoder's official preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from benchmark_two_stage_encoders_npu import (
    checkpoint_file_sha256,
    flatten_output,
    load_encoder,
    model_state_sha256,
)


DATASET_CHOICES = ("cifar10", "cifar100", "dtd", "eurosat", "svhn")


class ArrowImageDataset(Dataset):
    """Read a local Hugging Face Arrow image split without network access."""

    def __init__(self, arrow_path: Path, transform: object) -> None:
        from datasets import Dataset as HFDataset

        self.dataset = HFDataset.from_file(str(arrow_path))
        self.transform = transform
        self.labels = np.asarray(self.dataset["label"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deduplicate_files_by_content(paths: Sequence[Path]) -> tuple[list[Path], list[str]]:
    """Keep one deterministic path per content hash to avoid duplicate cache shards."""

    unique: list[Path] = []
    hashes: list[str] = []
    seen: set[str] = set()
    for path in sorted(paths):
        digest = sha256_file(path)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(path)
        hashes.append(digest)
    return unique, hashes


def dataset_labels(dataset: Dataset) -> np.ndarray:
    for attribute in ("targets", "labels", "_labels"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            return np.asarray(values, dtype=np.int64)
    raise TypeError(f"dataset {type(dataset).__name__} exposes no label vector")


def stratified_indices(labels: Sequence[int], samples: int, seed: int) -> np.ndarray:
    """Return a deterministic sample that is proportional whenever size permits."""

    labels_array = np.asarray(labels, dtype=np.int64)
    total = int(labels_array.size)
    if samples <= 0 or samples >= total:
        return np.arange(total, dtype=np.int64)

    classes, counts = np.unique(labels_array, return_counts=True)
    exact = samples * counts.astype(np.float64) / total
    if samples < len(classes):
        allocation = np.zeros(len(classes), dtype=np.int64)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            len(classes), samples, replace=False, p=counts.astype(float) / total,
        )
        allocation[chosen] = 1
    else:
        allocation = np.floor(exact).astype(np.int64)
        allocation = np.maximum(allocation, 1)

    while int(allocation.sum()) > samples:
        candidates = np.flatnonzero(allocation > 1)
        remove_index = candidates[np.argmin(exact[candidates] - allocation[candidates])]
        allocation[remove_index] -= 1
    while int(allocation.sum()) < samples:
        candidates = np.flatnonzero(allocation < counts)
        add_index = candidates[np.argmax(exact[candidates] - allocation[candidates])]
        allocation[add_index] += 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_value, class_samples in zip(classes, allocation):
        candidates = np.flatnonzero(labels_array == class_value)
        selected.append(rng.choice(candidates, int(class_samples), replace=False))
    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def unlabeled_indices(total: int, samples: int, seed: int) -> np.ndarray:
    """Return deterministic indices without consulting class labels."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if samples <= 0 or samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.random.default_rng(seed).choice(total, samples, replace=False).astype(
        np.int64, copy=False,
    )


def find_arrow_split(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {filename} below {root}, found {len(matches)}"
        )
    return matches[0]


def make_dataset(
    name: str,
    preprocess: object,
    data_root: Path,
    eurosat_root: Path,
    svhn_cache: Path,
) -> tuple[Dataset, str, str]:
    if name == "cifar10":
        dataset = datasets.CIFAR10(
            data_root, train=True, transform=preprocess, download=False,
        )
        return dataset, "train", str(data_root / "cifar-10-batches-py")
    if name == "cifar100":
        dataset = datasets.CIFAR100(
            data_root, train=True, transform=preprocess, download=False,
        )
        return dataset, "train", str(data_root / "cifar-100-python")
    if name == "dtd":
        dataset = datasets.DTD(
            data_root, split="train", partition=1, transform=preprocess, download=True,
        )
        return dataset, "train1", str(data_root / "dtd" / "dtd")
    if name == "eurosat":
        image_root = eurosat_root / "2750"
        dataset = datasets.ImageFolder(image_root, transform=preprocess)
        return dataset, "full", str(image_root)
    if name == "svhn":
        arrow_path = find_arrow_split(svhn_cache, "svhn-train.arrow")
        dataset = ArrowImageDataset(arrow_path, preprocess)
        return dataset, "train", str(arrow_path)
    raise ValueError(f"unsupported dataset: {name}")


def output_stem(variant: str, dataset: str, samples: int) -> str:
    sample_tag = "all" if samples <= 0 else str(samples)
    return f"{variant}__{dataset}__n{sample_tag}"


def cached_result_is_valid(
    npz_path: Path,
    metadata_path: Path,
    expected_metadata: dict[str, object],
) -> bool:
    if not npz_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        with np.load(npz_path) as payload:
            shape = list(payload["H"].shape)
            return (
                shape == metadata["feature_shape"]
                and len(payload["y"]) == shape[0]
                and len(payload["indices"]) == shape[0]
                and sha256_file(npz_path) == metadata["feature_file_sha256"]
                and all(metadata.get(key) == value for key, value in expected_metadata.items())
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def extract_one(
    model: torch.nn.Module,
    device: torch.device,
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=False,
    )
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    torch.npu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            output = flatten_output(model(images.to(device, non_blocking=True)))
            feature_batches.append(output.float().cpu().numpy())
            label_batches.append(np.asarray(labels, dtype=np.int64))
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    features = np.concatenate(feature_batches).astype(np.float32, copy=False)
    output_labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    return (
        features,
        output_labels,
        elapsed,
        torch.npu.max_memory_allocated(device) / (1024 ** 2),
        torch.npu.max_memory_reserved(device) / (1024 ** 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=list(DATASET_CHOICES))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eurosat-root", type=Path, required=True)
    parser.add_argument("--svhn-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--sample-seed", type=int, default=20260902)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__)
    encoder_script = script_path.with_name("benchmark_two_stage_encoders_npu.py")
    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    load_started = time.perf_counter()
    model, preprocess, checkpoint = load_encoder(args.variant)
    model_state_hash = model_state_sha256(model)
    checkpoint_file_hash = checkpoint_file_sha256(checkpoint)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - load_started
    script_sha256 = sha256_file(script_path)
    encoder_loader_sha256 = sha256_file(encoder_script)
    preprocess_sha256 = hashlib.sha256(repr(preprocess).encode()).hexdigest()
    print(
        f"variant={args.variant} device={device} model_load={model_load_seconds:.3f}s",
        flush=True,
    )

    for dataset_name in args.datasets:
        stem = output_stem(args.variant, dataset_name, args.samples)
        npz_path = args.output_dir / f"{stem}.npz"
        metadata_path = args.output_dir / f"{stem}.json"
        expected_metadata = {
            "variant": args.variant,
            "checkpoint": checkpoint,
            "checkpoint_file_sha256": checkpoint_file_hash,
            "model_state_sha256": model_state_hash,
            "preprocess_sha256": preprocess_sha256,
            "dataset": dataset_name,
            "requested_samples": args.samples,
            "sample_seed": args.sample_seed,
            "script_sha256": script_sha256,
            "encoder_loader_sha256": encoder_loader_sha256,
        }
        if not args.force and cached_result_is_valid(
            npz_path, metadata_path, expected_metadata,
        ):
            print(f"{dataset_name}: cached {npz_path}", flush=True)
            continue

        started_utc = datetime.now(timezone.utc).isoformat()
        dataset, split, source = make_dataset(
            dataset_name,
            preprocess,
            args.data_root,
            args.eurosat_root,
            args.svhn_cache,
        )
        labels = dataset_labels(dataset)
        indices = stratified_indices(labels, args.samples, args.sample_seed)
        features, selected_labels, elapsed, peak_allocated, peak_reserved = extract_one(
            model,
            device,
            dataset,
            indices,
            args.batch_size,
            args.workers,
        )
        if not np.array_equal(selected_labels, labels[indices]):
            raise RuntimeError(f"label/index mismatch for {dataset_name}")

        temporary_npz = npz_path.with_suffix(".tmp.npz")
        np.savez(
            temporary_npz,
            H=features,
            y=selected_labels,
            indices=indices,
        )
        temporary_npz.replace(npz_path)
        metadata = {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "variant": args.variant,
            "checkpoint": checkpoint,
            "checkpoint_file_sha256": checkpoint_file_hash,
            "model_state_sha256": model_state_hash,
            "preprocess": repr(preprocess),
            "preprocess_sha256": preprocess_sha256,
            "dataset": dataset_name,
            "split": split,
            "source": source,
            "source_samples": len(dataset),
            "requested_samples": args.samples,
            "sample_seed": args.sample_seed,
            "sampling": "deterministic_proportional_stratified",
            "saved_labels_for_audit_only": True,
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "index_dtype": str(indices.dtype),
            "inference_seconds": elapsed,
            "images_per_second": len(indices) / elapsed,
            "model_load_seconds": model_load_seconds,
            "peak_allocated_mb": peak_allocated,
            "peak_reserved_mb": peak_reserved,
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "script_sha256": script_sha256,
            "encoder_loader_sha256": encoder_loader_sha256,
            "feature_file_sha256": sha256_file(npz_path),
        }
        temporary_metadata = metadata_path.with_suffix(".tmp.json")
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_metadata.replace(metadata_path)
        print(
            f"{dataset_name}: shape={features.shape} seconds={elapsed:.3f} "
            f"rate={len(indices) / elapsed:.2f}/s sha256={metadata['feature_file_sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
