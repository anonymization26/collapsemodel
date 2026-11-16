#!/usr/bin/env python3
"""Check frozen encoder identities and one synthetic forward pass per CUDA GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from benchmark_two_stage_encoders_npu import (
    DINO_REPOSITORY, load_encoder, model_state_sha256, flatten_output,
)
from extract_target_conditioned_e2b_features_npu import _checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(torch.from_numpy(probe).numpy(), probe)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError("this preparation profile requires three available CUDA GPUs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["XFORMERS_DISABLED"] = "1"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rows = []
    for index, variant in enumerate(("resnet50", "dinov2_b14", "clip_b32")):
        reference = json.loads((args.reference_dir / variant / "metadata.json").read_text())
        started = time.perf_counter()
        model, preprocess, checkpoint = load_encoder(variant)
        _, checkpoint_hash = _checkpoint_identity(variant, checkpoint)
        observed = {"name": variant, "checkpoint_file_sha256": checkpoint_hash,
                    "model_state_sha256": model_state_sha256(model),
                    "preprocess_sha256": hashlib.sha256(repr(preprocess).encode()).hexdigest()}
        differences = {key: {"expected": reference["encoder"].get(key), "observed": value}
                       for key, value in observed.items() if reference["encoder"].get(key) != value}
        if differences:
            raise RuntimeError(f"{variant} frozen identity mismatch: {differences}")
        device = torch.device(f"cuda:{index}")
        model.eval().to(device)
        inputs = preprocess(Image.new("RGB", (256, 256), (127, 127, 127))).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = flatten_output(model(inputs)).float().cpu().numpy()
        if output.shape != (1, reference["feature_shape"][1]) or not np.isfinite(output).all():
            raise RuntimeError(f"{variant} CUDA smoke output is invalid")
        rows.append({**observed, "device": str(device), "device_name": torch.cuda.get_device_name(index),
                     "shape": list(output.shape), "elapsed_seconds": time.perf_counter() - started,
                     "architecture_repr_sha256": hashlib.sha256(repr(model).encode()).hexdigest(),
                     "synthetic_smoke_only": True})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        del model, inputs
        torch.cuda.empty_cache()
    names = ("torch", "torchvision", "numpy", "scipy", "pyarrow", "fsspec", "open_clip_torch", "timm", "Pillow")
    repository_files = {
        str(path.relative_to(DINO_REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(DINO_REPOSITORY.rglob("*.py")) if "__pycache__" not in path.parts
    }
    report = {"status": "passed", "python": platform.python_version(), "cuda": torch.version.cuda,
              "precision": {"tf32": False, "autocast": False, "xformers": False, "sdpa": "math"},
              "packages": {name: importlib.metadata.version(name) for name in names}, "encoders": rows,
              "dinov2_source_sha256": repository_files,
              "reference_scope": "Weight/state/preprocess fingerprints; not full cross-version functional equivalence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
