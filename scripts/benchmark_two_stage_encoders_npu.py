#!/usr/bin/env python3
"""Benchmark frozen visual encoders on an Ascend NPU without saving features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


MANUAL_WEIGHT_DIR = Path(os.environ.get("COLLAPSE_WEIGHT_DIR", "/data/Paper06/manual_weights"))
DINO_REPOSITORY = Path(os.environ.get(
    "COLLAPSE_DINO_REPOSITORY", "/root/.cache/torch/hub/facebookresearch_dinov2_main"
))
DINO_CHECKPOINT = Path(os.environ.get(
    "COLLAPSE_DINO_CHECKPOINT", "/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
))


class OpenClipImageEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images, normalize=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    """Fingerprint the final loaded model state, independent of checkpoint labels."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_file_sha256(checkpoint: str) -> str | None:
    path = Path(checkpoint)
    return sha256_file(path) if path.is_file() else None


def load_clip() -> tuple[nn.Module, object, str]:
    import open_clip

    checkpoint = MANUAL_WEIGHT_DIR / "ViT-B-32.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None,
    )
    scripted = torch.jit.load(str(checkpoint), map_location="cpu")
    incompatible = model.load_state_dict(scripted.state_dict(), strict=False)
    unexpected = [
        key for key in incompatible.unexpected_keys
        if key not in {"input_resolution", "context_length", "vocab_size"}
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"CLIP checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={unexpected}"
        )
    return OpenClipImageEncoder(model), preprocess, str(checkpoint)


def load_vit_b16() -> tuple[nn.Module, object, str]:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    model.heads = nn.Identity()
    return model, weights.transforms(), "torchvision:ViT_B_16_Weights.IMAGENET1K_V1"


def load_dinov2_b14() -> tuple[nn.Module, object, str]:
    if not DINO_REPOSITORY.exists() or not DINO_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"missing offline DINOv2 assets: {DINO_REPOSITORY}, {DINO_CHECKPOINT}"
        )
    model = torch.hub.load(
        str(DINO_REPOSITORY), "dinov2_vitb14", source="local", pretrained=False,
    )
    model.load_state_dict(torch.load(DINO_CHECKPOINT, map_location="cpu", weights_only=True))
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, preprocess, str(DINO_CHECKPOINT)


def load_encoder(variant: str) -> tuple[nn.Module, object, str]:
    if variant == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        return model, weights.transforms(), "torchvision:ResNet50_Weights.IMAGENET1K_V2"
    if variant == "clip_b32":
        return load_clip()
    if variant == "vit_b16":
        return load_vit_b16()
    return load_dinov2_b14()


def flatten_output(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"unsupported encoder output: {type(output)!r}")
    if output.ndim > 2:
        output = output.flatten(1)
    return output


def synchronize() -> None:
    torch.npu.synchronize()


def main() -> None:
    import torch_npu  # noqa: F401 - only the NPU benchmark registers this backend

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["resnet50", "vit_b16", "clip_b32", "dinov2_b14"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--npu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started_utc = datetime.now(timezone.utc).isoformat()
    device = torch.device(f"npu:{args.npu}")
    torch.npu.set_device(device)
    load_started = time.perf_counter()
    model, preprocess, checkpoint = load_encoder(args.variant)
    model_state_hash = model_state_sha256(model)
    checkpoint_file_hash = checkpoint_file_sha256(checkpoint)
    model.eval().to(device)
    model_load_seconds = time.perf_counter() - load_started

    dataset = datasets.CIFAR10(
        args.data_root, train=True, transform=preprocess, download=False,
    )
    if args.samples > len(dataset):
        raise ValueError(f"requested {args.samples} samples from a {len(dataset)}-image dataset")
    indices = np.random.default_rng(20260902).choice(
        len(dataset), args.samples, replace=False,
    ).tolist()
    subset = Subset(dataset, indices)

    def make_loader() -> DataLoader:
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            persistent_workers=False,
        )

    feature_dim = None
    warmup_loader = make_loader()
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(warmup_loader):
            output = flatten_output(model(images.to(device, non_blocking=True)))
            feature_dim = int(output.shape[1])
            if batch_index + 1 >= args.warmup_batches:
                break
    synchronize()
    torch.npu.reset_peak_memory_stats(device)

    count = 0
    batches = 0
    run_started = time.perf_counter()
    with torch.inference_mode():
        for images, _ in make_loader():
            output = flatten_output(model(images.to(device, non_blocking=True)))
            count += int(output.shape[0])
            batches += 1
            feature_dim = int(output.shape[1])
    synchronize()
    inference_seconds = time.perf_counter() - run_started
    peak_allocated = int(torch.npu.max_memory_allocated(device))
    peak_reserved = int(torch.npu.max_memory_reserved(device))

    script_path = Path(__file__)
    result = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "checkpoint": checkpoint,
        "checkpoint_file_sha256": checkpoint_file_hash,
        "model_state_sha256": model_state_hash,
        "device": str(device),
        "dataset": "CIFAR10:train",
        "samples": count,
        "batches": batches,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "warmup_batches": args.warmup_batches,
        "feature_dim": feature_dim,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "images_per_second": count / inference_seconds,
        "peak_allocated_mb": peak_allocated / (1024 ** 2),
        "peak_reserved_mb": peak_reserved / (1024 ** 2),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
