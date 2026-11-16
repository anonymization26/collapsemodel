#!/usr/bin/env python3
"""Extract manifest-ordered E2b features on CUDA, CPU, or Ascend.

The historical filename and --npu option remain supported for frozen run scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
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
from metrics.target_conditioned_e2b import (  # noqa: E402
    CACHE_SCHEMA,
    E2BArtifactError,
    load_config,
    read_manifest_samples,
    sha256_file,
    validate_feature_cache,
    validate_manifest,
    write_json_atomic,
)


ENCODERS = ("clip_b32", "resnet50", "dinov2_b14")
FEATURE_LAYERS = {
    "clip_b32": "image-projection",
    "resnet50": "global-average-pool-before-fc",
    "dinov2_b14": "final-cls-token",
}


class ManifestDataset(Dataset):
    def __init__(self, root: Path, samples: list[dict[str, object]], transform: object):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.samples[index]
        path = self.root / str(row["relative_path"])
        payload = path.read_bytes()
        if len(payload) != int(row["byte_size"]):
            raise RuntimeError(f"sample size differs from manifest: {row['sample_id']}")
        if hashlib.sha256(payload).hexdigest() != row["content_sha256"]:
            raise RuntimeError(f"sample hash differs from manifest: {row['sample_id']}")
        with Image.open(io.BytesIO(payload)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def _git_revision() -> str:
    pattern = re.compile(r"[0-9a-f]{40}")
    injected = os.environ.get("SOURCE_GIT_REVISION", "")
    if pattern.fullmatch(injected):
        return injected
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision_path = ROOT / "SOURCE_REVISION"
        revision = revision_path.read_text(encoding="ascii").strip()
    if not pattern.fullmatch(revision):
        raise RuntimeError("cannot establish the extractor Git revision")
    return revision


def _checkpoint_identity(variant: str, checkpoint: str) -> tuple[str, str | None]:
    path = Path(checkpoint)
    if path.is_file():
        return path.name, checkpoint_file_sha256(checkpoint)
    if variant == "resnet50":
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"
        if cached.is_file():
            return checkpoint, sha256_file(cached)
    return checkpoint, None


def _write_npz(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, H=features, y=labels, sample_ids=sample_ids)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ENCODERS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    devices = parser.add_mutually_exclusive_group(required=True)
    devices.add_argument("--npu", type=int)
    devices.add_argument("--device", help="cuda:0, cuda:1, cuda:2, or cpu")
    parser.add_argument("--expected-metadata", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.variant != config["candidate_construction"]["anchor_encoder"] and (
        args.variant not in config["features"]["evaluators"]
    ):
        raise E2BArtifactError("encoder is not registered for E2b")
    manifest_report = validate_manifest(args.manifest_dir, args.config)
    samples = read_manifest_samples(args.manifest_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "features.npz"
    metadata_path = args.output_dir / "metadata.json"
    if feature_path.exists() or metadata_path.exists():
        if feature_path.exists() and metadata_path.exists():
            report = validate_feature_cache(
                feature_path,
                metadata_path,
                args.manifest_dir,
                args.config,
                args.variant,
            )
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if args.expected_metadata is not None and cached.get("reference_metadata_sha256") != sha256_file(args.expected_metadata):
                raise E2BArtifactError("cached features lack the requested reference identity check")
            requested_device = f"npu:{args.npu}" if args.npu is not None else str(torch.device(args.device))
            if cached.get("runtime", {}).get("device") != requested_device:
                raise E2BArtifactError("cached features were produced on a different requested device")
            print(json.dumps({"status": "cached", **report}, sort_keys=True))
            return
        raise E2BArtifactError("partial feature cache exists")

    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    if args.npu is not None:
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{args.npu}")
    else:
        device = torch.device(args.device)
        if device.type not in {"cuda", "cpu"}:
            parser.error("--device supports cuda or cpu; use --npu for Ascend")
    backend = None if device.type == "cpu" else getattr(torch, device.type)
    if backend is not None:
        backend.set_device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["XFORMERS_DISABLED"] = "1"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    started_utc = datetime.now(timezone.utc).isoformat()
    load_started = time.perf_counter()
    model, preprocess, checkpoint = load_encoder(args.variant)
    state_hash = model_state_sha256(model)
    checkpoint_id, checkpoint_hash = _checkpoint_identity(args.variant, checkpoint)
    preprocess_text = repr(preprocess)
    preprocess_hash = hashlib.sha256(preprocess_text.encode("utf-8")).hexdigest()
    expected_metadata_hash = None
    if args.expected_metadata is not None:
        expected = json.loads(args.expected_metadata.read_text(encoding="utf-8"))["encoder"]
        observed = {
            "name": args.variant,
            "checkpoint_file_sha256": checkpoint_hash,
            "model_state_sha256": state_hash,
            "preprocess_sha256": preprocess_hash,
        }
        for key, value in observed.items():
            if value is None or expected.get(key) != value:
                raise E2BArtifactError(f"encoder identity differs from reference: {key}")
        expected_metadata_hash = sha256_file(args.expected_metadata)
    model.eval().to(device)
    load_seconds = time.perf_counter() - load_started

    loader = DataLoader(
        ManifestDataset(args.dataset_root, samples, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    features = []
    observed_indices = []
    if backend is not None:
        backend.synchronize(device)
        backend.reset_peak_memory_stats(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, indices) in enumerate(loader, 1):
            output = flatten_output(model(images.to(device, non_blocking=True)))
            if output.ndim != 2:
                raise RuntimeError(f"encoder output has shape {tuple(output.shape)}")
            features.append(output.float().cpu().numpy())
            observed_indices.append(np.asarray(indices, dtype=np.int64))
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(json.dumps({"status": "extracting", "variant": args.variant,
                                  "batches": batch_index, "total_batches": len(loader),
                                  "elapsed_seconds": time.perf_counter() - inference_started}),
                      flush=True)
    if backend is not None:
        backend.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    matrix = np.concatenate(features).astype(np.float32, copy=False)
    indices = np.concatenate(observed_indices)
    if not np.array_equal(indices, np.arange(len(samples))):
        raise RuntimeError("DataLoader changed manifest order")
    if not np.isfinite(matrix).all():
        raise RuntimeError("encoder emitted nonfinite features")
    labels = np.asarray([int(row["class_id"]) for row in samples], dtype=np.int64)
    sample_ids = np.asarray([str(row["sample_id"]) for row in samples], dtype="<U64")
    _write_npz(feature_path, matrix, labels, sample_ids)

    metadata = {
        "schema_version": CACHE_SCHEMA,
        "manifest_id": manifest_report["manifest_id"],
        "config_file_sha256": sha256_file(args.config),
        "feature_file_sha256": sha256_file(feature_path),
        "feature_shape": list(matrix.shape),
        "feature_dtype": str(matrix.dtype),
        "label_dtype": str(labels.dtype),
        "sample_id_dtype": str(sample_ids.dtype),
        "encoder": {
            "name": args.variant,
            "checkpoint_id": checkpoint_id,
            "checkpoint_file_sha256": checkpoint_hash,
            "model_state_sha256": state_hash,
            "feature_layer": FEATURE_LAYERS[args.variant],
            "preprocess": preprocess_text,
            "preprocess_sha256": preprocess_hash,
        },
        "access": {
            "feature_extraction_uses_labels": False,
            "labels_saved_for_later_role_gated_stages": True,
            "ordering": "samples.csv sample_index",
        },
        "runtime": {
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "images_per_second": len(samples) / inference_seconds,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "device": str(device),
            "device_name": platform.processor() if backend is None else backend.get_device_name(device),
            "peak_allocated_mb": None if backend is None else backend.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mb": None if backend is None else backend.max_memory_reserved(device) / (1024**2),
            "cuda_version": torch.version.cuda,
            "tf32_enabled": False if device.type == "cuda" else None,
            "sdpa_backend": "math" if device.type == "cuda" else None,
            "xformers_disabled": os.environ.get("XFORMERS_DISABLED") is not None,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "code": {
            "git_revision": _git_revision(),
            "extractor_sha256": sha256_file(Path(__file__)),
            "encoder_loader_sha256": sha256_file(ROOT / "scripts/benchmark_two_stage_encoders_npu.py"),
        },
        "reference_metadata_sha256": expected_metadata_hash,
        "reconstruction_note": "New cache; cross-device bitwise equivalence is not assumed.",
    }
    write_json_atomic(metadata_path, metadata)
    report = validate_feature_cache(
        feature_path,
        metadata_path,
        args.manifest_dir,
        args.config,
        args.variant,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                **report,
                "inference_seconds": inference_seconds,
                "images_per_second": len(samples) / inference_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
