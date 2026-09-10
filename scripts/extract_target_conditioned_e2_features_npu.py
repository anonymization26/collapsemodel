#!/usr/bin/env python3
"""Extract manifest-ordered frozen E2 features on an Ascend NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch_npu  # noqa: F401 - registers the NPU backend
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_two_stage_encoders_npu import (  # noqa: E402
    checkpoint_file_sha256,
    flatten_output,
    load_encoder,
    model_state_sha256,
)
from metrics.target_conditioned_e2 import (  # noqa: E402
    CACHE_SCHEMA,
    E2ArtifactError,
    read_manifest_samples,
    sha256_file,
    validate_feature_cache,
    validate_manifest_bundle,
    write_json_atomic,
)


ENCODER_CHOICES = ("resnet50", "dinov2_b14")
FEATURE_LAYERS = {
    "resnet50": "global-average-pool-before-fc",
    "dinov2_b14": "final-cls-token",
}


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        samples: List[Dict[str, object]],
        transform: object,
    ) -> None:
        self.dataset_root = dataset_root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.samples[index]
        path = self.dataset_root / str(row["relative_path"])
        with Image.open(path) as image:
            converted = image.convert("RGB")
            tensor = self.transform(converted)
        return tensor, index


def _checkpoint_identity(
    variant: str,
    checkpoint: str,
) -> Tuple[str, object]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return checkpoint_path.name, checkpoint_file_sha256(checkpoint)
    if variant == "resnet50":
        cached = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / "resnet50-11ad3fa6.pth"
        )
        if cached.is_file():
            return checkpoint, sha256_file(cached)
    return checkpoint, None


def _git_revision() -> str:
    revision_pattern = re.compile(r"[0-9a-f]{40}")
    environment_revision = os.environ.get("SOURCE_GIT_REVISION", "")
    if revision_pattern.fullmatch(environment_revision):
        return environment_revision
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision_file = ROOT / "SOURCE_REVISION"
        if revision_file.is_file():
            revision = revision_file.read_text(
                encoding="ascii"
            ).strip()
        else:
            revision = ""
    if not revision_pattern.fullmatch(revision):
        raise RuntimeError(
            "cannot establish a 40-character source Git revision"
        )
    return revision


def _device_name(device_index: int) -> str:
    try:
        return str(torch.npu.get_device_name(device_index))
    except (AttributeError, RuntimeError):
        return "unknown-ascend-npu"


def _extract(
    model: torch.nn.Module,
    device: torch.device,
    dataset: Dataset,
    batch_size: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )
    feature_batches = []
    index_batches = []
    torch.npu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, indices in loader:
            output = flatten_output(
                model(images.to(device, non_blocking=True))
            )
            if output.ndim != 2:
                raise RuntimeError(
                    f"encoder output must be two-dimensional: {output.shape}"
                )
            feature_batches.append(
                output.float().cpu().numpy()
            )
            index_batches.append(
                np.asarray(indices, dtype=np.int64)
            )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    features = np.concatenate(feature_batches).astype(
        np.float32,
        copy=False,
    )
    indices = np.concatenate(index_batches).astype(
        np.int64,
        copy=False,
    )
    expected = np.arange(len(dataset), dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError(
            "DataLoader output order differs from manifest order"
        )
    return (
        features,
        indices,
        elapsed,
        torch.npu.max_memory_allocated(device) / (1024 ** 2),
        torch.npu.max_memory_reserved(device) / (1024 ** 2),
    )


def _write_npz_atomic(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            H=features,
            y=labels,
            sample_ids=sample_ids,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=ENCODER_CHOICES,
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_report = validate_manifest_bundle(
        args.manifest_dir,
        dataset_root=args.dataset_root,
        verify_images=False,
    )
    samples = read_manifest_samples(args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "features.npz"
    metadata_path = args.output_dir / "metadata.json"
    if feature_path.exists() or metadata_path.exists():
        if feature_path.exists() and metadata_path.exists():
            try:
                report = validate_feature_cache(
                    feature_path,
                    metadata_path,
                    args.manifest_dir,
                    expected_encoder=args.variant,
                )
            except E2ArtifactError:
                if not args.force:
                    raise
            else:
                if not args.force:
                    print(
                        json.dumps(
                            {"status": "cached", **report},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    return
        elif not args.force:
            raise E2ArtifactError(
                "partial cache exists; inspect it or rerun with --force"
            )

    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    started_utc = datetime.now(timezone.utc).isoformat()
    load_started = time.perf_counter()
    model, preprocess, checkpoint = load_encoder(args.variant)
    state_hash = model_state_sha256(model)
    checkpoint_id, checkpoint_hash = _checkpoint_identity(
        args.variant,
        checkpoint,
    )
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - load_started

    dataset = ManifestImageDataset(
        args.dataset_root,
        samples,
        preprocess,
    )
    (
        features,
        _,
        inference_seconds,
        peak_allocated_mb,
        peak_reserved_mb,
    ) = _extract(
        model,
        device,
        dataset,
        args.batch_size,
        args.workers,
    )
    if not np.isfinite(features).all():
        raise RuntimeError("encoder produced non-finite features")
    labels = np.asarray(
        [int(row["class_id"]) for row in samples],
        dtype=np.int64,
    )
    sample_ids = np.asarray(
        [str(row["sample_id"]) for row in samples],
        dtype="<U64",
    )
    _write_npz_atomic(
        feature_path,
        features,
        labels,
        sample_ids,
    )

    preprocess_text = repr(preprocess)
    script_path = Path(__file__)
    encoder_loader_path = script_path.with_name(
        "benchmark_two_stage_encoders_npu.py"
    )
    manifest = json.loads(
        (args.manifest_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "dataset": manifest_report["dataset"],
        "manifest_id": manifest_report["manifest_id"],
        "manifest_file_sha256": sha256_file(
            args.manifest_dir / "manifest.json"
        ),
        "samples_file_sha256": manifest["samples_file_sha256"],
        "assignments_file_sha256": (
            manifest["assignments_file_sha256"]
        ),
        "feature_file_sha256": sha256_file(feature_path),
        "sample_count": len(samples),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "sample_id_dtype": str(sample_ids.dtype),
        "feature_postprocessing": "none_raw_pooled",
        "encoder": {
            "name": args.variant,
            "checkpoint_id": checkpoint_id,
            "checkpoint_file_sha256": checkpoint_hash,
            "model_state_sha256": state_hash,
            "feature_layer": FEATURE_LAYERS[args.variant],
            "preprocess": preprocess_text,
            "preprocess_sha256": hashlib.sha256(
                preprocess_text.encode("utf-8")
            ).hexdigest(),
        },
        "data_access": {
            "extraction_uses_labels": False,
            "labels_saved_for_evaluation_only": True,
            "ordering": "samples.csv sample_index",
            "target_test_access": (
                "feature extraction only; selection code must use "
                "assignment views"
            ),
        },
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": model_load_seconds,
            "inference_seconds": inference_seconds,
            "images_per_second": len(samples) / inference_seconds,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "device": str(device),
            "device_name": _device_name(args.npu),
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "code": {
            "git_revision": _git_revision(),
            "extractor_sha256": sha256_file(script_path),
            "encoder_loader_sha256": sha256_file(
                encoder_loader_path
            ),
        },
    }
    write_json_atomic(metadata_path, metadata)
    report = validate_feature_cache(
        feature_path,
        metadata_path,
        args.manifest_dir,
        expected_encoder=args.variant,
    )
    print(
        json.dumps(
            {
                **report,
                "status": "completed",
                "inference_seconds": inference_seconds,
                "images_per_second": len(samples) / inference_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
