#!/usr/bin/env python3
"""Extract auditable source-pool features from locally cached Arrow data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from torch.utils.data import Dataset

from benchmark_two_stage_encoders_npu import (
    checkpoint_file_sha256,
    load_encoder,
    model_state_sha256,
)
from extract_two_stage_features_npu import (
    cached_result_is_valid,
    deduplicate_files_by_content,
    extract_one,
    sha256_file,
    stratified_indices,
    unlabeled_indices,
)


# Paths are relative to the Hugging Face datasets cache. Semeion is excluded
# because the configured upstream dataset is unavailable and no auditable local
# split exists. CIFAR-100 and CIFAR-100-Coarse intentionally share images.
SOURCE_ARROWS: dict[str, tuple[str, str, str]] = {
    "bloodmnist": (
        "albertvillanova___medmnist-v2/bloodmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "breastmnist": (
        "albertvillanova___medmnist-v2/breastmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "cifar10": ("cifar10/**/cifar10-train.arrow", "img", "label"),
    "cifar100": ("cifar100/**/cifar100-train.arrow", "img", "fine_label"),
    "cifar100_coarse": ("cifar100/**/cifar100-train.arrow", "img", "coarse_label"),
    "dermamnist": (
        "albertvillanova___medmnist-v2/dermamnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "dtd": ("tanganke___dtd/**/dtd-train.arrow", "image", "label"),
    "eurosat": (
        "tanganke___eurosat/**/eurosat-train.arrow", "image", "label",
    ),
    "fashion_mnist": (
        "fashion_mnist/**/fashion_mnist-train.arrow", "image", "label",
    ),
    "mnist": ("mnist/**/mnist-train.arrow", "image", "label"),
    "octmnist": (
        "albertvillanova___medmnist-v2/octmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "organamnist": (
        "albertvillanova___medmnist-v2/organamnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "organcmnist": (
        "albertvillanova___medmnist-v2/organcmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "organsmnist": (
        "albertvillanova___medmnist-v2/organsmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "pathmnist": (
        "albertvillanova___medmnist-v2/pathmnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "pneumoniamnist": (
        "albertvillanova___medmnist-v2/pneumoniamnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "rendered_sst2": (
        "nateraw___rendered-sst2/**/rendered-sst2-train.arrow", "image", "label",
    ),
    "retinamnist": (
        "albertvillanova___medmnist-v2/retinamnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "stl10": ("tanganke___stl10/**/stl10-train.arrow", "image", "label"),
    "svhn": ("svhn/**/svhn-train.arrow", "image", "label"),
    "tiny_imagenet": (
        "zh-plus___tiny-imagenet/**/tiny-imagenet-train.arrow", "image", "label",
    ),
    "tissuemnist": (
        "albertvillanova___medmnist-v2/tissuemnist/**/medmnist-v2-train.arrow",
        "image", "label",
    ),
    "usps": ("flwrlabs___usps/**/usps-train.arrow", "image", "label"),
}


class ArrowPoolDataset(Dataset):
    """Expose cached Arrow shards with explicit image and label columns."""

    def __init__(
        self,
        paths: list[Path],
        transform: object,
        image_key: str,
        label_key: str,
    ) -> None:
        from datasets import Dataset as HFDataset
        from datasets import concatenate_datasets

        shards = [HFDataset.from_file(str(path)) for path in paths]
        self.dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        missing = {image_key, label_key} - set(self.dataset.column_names)
        if missing:
            raise KeyError(f"missing columns {sorted(missing)} in {self.dataset.column_names}")
        labels = np.asarray(self.dataset[label_key], dtype=np.int64)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1:
            raise ValueError(f"labels for {label_key} have unsupported shape {labels.shape}")
        self.labels = labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[index]
        image = row[self.image_key].convert("RGB")
        return self.transform(image), int(row[self.label_key])


def resolve_source_files(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no Arrow files matching {pattern!r} below {root}")
    return paths


def output_stem(variant: str, dataset: str, samples: int) -> str:
    sample_tag = "all" if samples <= 0 else str(samples)
    return f"{variant}__{dataset}__train__n{sample_tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
        required=True,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(SOURCE_ARROWS), default=sorted(SOURCE_ARROWS),
    )
    parser.add_argument("--arrow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--sample-seed", type=int, default=20260905)
    parser.add_argument(
        "--sampling", choices=["unlabeled_random", "stratified"],
        default="unlabeled_random",
    )
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
        pattern, image_key, label_key = SOURCE_ARROWS[dataset_name]
        source_paths = resolve_source_files(args.arrow_root, pattern)
        source_paths, source_hashes = deduplicate_files_by_content(source_paths)
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
            "split": "train",
            "requested_samples": args.samples,
            "sample_seed": args.sample_seed,
            "sampling": args.sampling,
            "source_file_sha256": source_hashes,
            "script_sha256": script_sha256,
            "encoder_loader_sha256": encoder_loader_sha256,
        }
        if not args.force and cached_result_is_valid(
            npz_path, metadata_path, expected_metadata,
        ):
            print(f"{dataset_name}: cached {npz_path}", flush=True)
            continue

        started_utc = datetime.now(timezone.utc).isoformat()
        dataset = ArrowPoolDataset(
            source_paths, preprocess, image_key=image_key, label_key=label_key,
        )
        if args.sampling == "unlabeled_random":
            indices = unlabeled_indices(len(dataset), args.samples, args.sample_seed)
        else:
            indices = stratified_indices(dataset.labels, args.samples, args.sample_seed)
        features, labels, elapsed, peak_allocated, peak_reserved = extract_one(
            model,
            device,
            dataset,
            indices,
            args.batch_size,
            args.workers,
        )
        if not np.array_equal(labels, dataset.labels[indices]):
            raise RuntimeError(f"label/index mismatch for {dataset_name}")

        temporary_npz = npz_path.with_suffix(".tmp.npz")
        np.savez(temporary_npz, H=features, y=labels, indices=indices)
        temporary_npz.replace(npz_path)
        metadata = {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            **expected_metadata,
            "preprocess": repr(preprocess),
            "source_files": [str(path) for path in source_paths],
            "source_samples": len(dataset),
            "stage1_sampling_reads_labels": args.sampling == "stratified",
            "saved_labels_for_stage2": True,
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
            f"rate={len(indices) / elapsed:.2f}/s "
            f"sha256={metadata['feature_file_sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
