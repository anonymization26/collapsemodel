#!/usr/bin/env python3
"""Extract auditable target train/test features from cached Arrow shards."""

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

from benchmark_two_stage_encoders_npu import load_encoder
from extract_two_stage_features_npu import (
    cached_result_is_valid,
    deduplicate_files_by_content,
    extract_one,
    sha256_file,
    stratified_indices,
)


# ``test`` may map to a source named ``validation`` when that is the dataset's
# official held-out split (Food-101).
ARROW_SPLITS: dict[str, dict[str, tuple[str, str]]] = {
    "beans": {
        "train": ("beans-train.arrow", "train"),
        "test": ("beans-test.arrow", "test"),
    },
    "dtd": {
        "train": ("dtd-train.arrow", "train"),
        "test": ("dtd-test.arrow", "test"),
    },
    "eurosat": {
        "train": ("eurosat-train.arrow", "train"),
        "test": ("eurosat-test.arrow", "test"),
    },
    "flowers102": {
        "train": ("oxford_flowers102-train.arrow", "train"),
        "test": ("oxford_flowers102-test*.arrow", "test"),
    },
    "food101": {
        "train": ("food101-train*.arrow", "train"),
        "test": ("food101-validation*.arrow", "validation"),
    },
    "gtsrb": {
        "train": ("gtsrb-train.arrow", "train"),
        "test": ("gtsrb-test.arrow", "test"),
    },
    "oxford_pets": {
        "train": ("oxford-iiit-pet-train.arrow", "train"),
        "test": ("oxford-iiit-pet-test.arrow", "test"),
    },
}


class ArrowShardDataset(Dataset):
    """Expose one or more cached Hugging Face Arrow shards as a torch dataset."""

    def __init__(self, paths: list[Path], transform: object) -> None:
        from datasets import Dataset as HFDataset
        from datasets import concatenate_datasets

        shards = [HFDataset.from_file(str(path)) for path in paths]
        self.dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
        self.transform = transform
        if "label" in self.dataset.column_names:
            self.label_key = "label"
        elif "labels" in self.dataset.column_names:
            self.label_key = "labels"
        else:
            raise KeyError(f"no label column in {self.dataset.column_names}")
        self.labels = np.asarray(self.dataset[self.label_key], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row[self.label_key])


def resolve_arrow_files(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.rglob(pattern))
    if not paths:
        raise FileNotFoundError(f"no Arrow files matching {pattern!r} below {root}")
    return paths


def output_stem(variant: str, dataset: str, split: str, samples: int) -> str:
    sample_tag = "all" if samples <= 0 else str(samples)
    return f"{variant}__{dataset}__{split}__n{sample_tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"],
        required=True,
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(ARROW_SPLITS), default=sorted(ARROW_SPLITS),
    )
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--arrow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--sample-seed", type=int, default=20260905)
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
        for requested_split in args.splits:
            pattern, source_split = ARROW_SPLITS[dataset_name][requested_split]
            source_paths = resolve_arrow_files(args.arrow_root, pattern)
            source_paths, source_hashes = deduplicate_files_by_content(source_paths)
            stem = output_stem(args.variant, dataset_name, requested_split, args.samples)
            npz_path = args.output_dir / f"{stem}.npz"
            metadata_path = args.output_dir / f"{stem}.json"
            expected_metadata = {
                "variant": args.variant,
                "checkpoint": checkpoint,
                "preprocess_sha256": preprocess_sha256,
                "dataset": dataset_name,
                "split": requested_split,
                "source_split": source_split,
                "requested_samples": args.samples,
                "sample_seed": args.sample_seed,
                "script_sha256": script_sha256,
                "encoder_loader_sha256": encoder_loader_sha256,
                "source_file_sha256": source_hashes,
            }
            if not args.force and cached_result_is_valid(
                npz_path, metadata_path, expected_metadata,
            ):
                print(f"{dataset_name}/{requested_split}: cached {npz_path}", flush=True)
                continue

            started_utc = datetime.now(timezone.utc).isoformat()
            dataset = ArrowShardDataset(source_paths, preprocess)
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
                raise RuntimeError(f"label/index mismatch for {dataset_name}/{requested_split}")

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
                "sampling": "deterministic_proportional_stratified",
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
                "feature_file_sha256": sha256_file(npz_path),
            }
            temporary_metadata = metadata_path.with_suffix(".tmp.json")
            temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary_metadata.replace(metadata_path)
            print(
                f"{dataset_name}/{requested_split}: shape={features.shape} "
                f"seconds={elapsed:.3f} rate={len(indices) / elapsed:.2f}/s "
                f"sha256={metadata['feature_file_sha256']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
