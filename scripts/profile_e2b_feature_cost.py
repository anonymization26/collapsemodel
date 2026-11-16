#!/usr/bin/env python3
"""Measure raw-image feature work for fixed pools, without validation/test outcomes."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time


def run(work, encoder, output):
    os.environ.update(COLLAPSE_WEIGHT_DIR=str(work / "weights"),
                      COLLAPSE_DINO_REPOSITORY=str(work / "dinov2_repository"),
                      COLLAPSE_DINO_CHECKPOINT=str(work / "weights/dinov2_vitb14_pretrain.pth"),
                      TORCH_HOME=str(work / "torch_cache"), XFORMERS_DISABLED="1")
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from benchmark_two_stage_encoders_npu import load_encoder, flatten_output, model_state_sha256
    from extract_target_conditioned_e2b_features_npu import ManifestDataset, _checkpoint_identity
    from metrics.target_conditioned_e2b import (
        jl_project, read_manifest_samples, sha256_file, stable_seed, write_json_atomic,
    )

    timing = work / "revision_results/e2b_decision_cost_v1/timing"
    if any(not (timing / e / "timing.json").is_file() for e in ("resnet50", "dinov2_b14")):
        raise RuntimeError("run feature profiling only after both isolated CPU timing jobs finish")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    frozen = work / "results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1"
    screen = json.loads((work / "stage_bundles" / encoder / "screen/manifest.json").read_text())
    source_ids = {sid for b in screen["candidates"] for sid in b["sample_ids"]}
    samples = read_manifest_samples(frozen / "data_manifest")
    samples = [r for r in samples if r["sample_id"] in source_ids
               or r["role"] in ("target_selection", "target_validation")]
    expected = json.loads((work / "reconstructed_features" / encoder / "metadata.json").read_text())["encoder"]
    tick = time.perf_counter()
    model, transform, checkpoint = load_encoder(encoder)
    _, checkpoint_hash = _checkpoint_identity(encoder, checkpoint)
    observed = {"name": encoder, "checkpoint_file_sha256": checkpoint_hash,
                "model_state_sha256": model_state_sha256(model),
                "preprocess_sha256": hashlib.sha256(repr(transform).encode()).hexdigest()}
    if any(expected.get(k) != v for k,v in observed.items()):
        raise RuntimeError("feature model identity mismatch")
    device = torch.device("cuda:0")
    model.eval().to(device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - tick
    def profile_rows(rows, domain, role, block_id=None):
        tick = time.perf_counter()
        loader = DataLoader(ManifestDataset(work / "dataset", rows, transform), batch_size=64,
                            num_workers=4, shuffle=False, pin_memory=False)
        count, batches = 0, []
        with torch.inference_mode():
            for images, indices in loader:
                features = flatten_output(model(images.to(device))).float().cpu().numpy()
                if not np.isfinite(features).all():
                    raise RuntimeError("nonfinite features")
                count += len(indices)
                batches.append(features)
        torch.cuda.synchronize()
        if count != len(rows):
            raise RuntimeError("feature coverage mismatch")
        projection_started = time.perf_counter()
        projected = jl_project(np.concatenate(batches), 128, stable_seed(20260917, encoder))
        if projected.shape != (count, 128):
            raise RuntimeError("projection coverage mismatch")
        projection_seconds = time.perf_counter() - projection_started
        record = {"domain": domain, "role": role, "count": count,
                  "seconds": time.perf_counter() - tick, "projection_seconds": projection_seconds}
        if block_id is not None:
            record["block_id"] = block_id
        print(json.dumps(record), flush=True)
        return record

    groups, source_blocks = [], []
    domains = sorted({r["domain"] for r in samples})
    by_id = {r["sample_id"]: r for r in samples}
    for domain in domains:
        blocks = []
        for candidate in sorted(screen["candidates"], key=lambda c: c["candidate_id"]):
            if candidate["domain"] != domain:
                continue
            record = profile_rows([by_id[s] for s in candidate["sample_ids"]],
                                  domain, "anchor_pool", candidate["candidate_id"])
            blocks.append(record)
            source_blocks.append(record)
        groups.append({"domain": domain, "role": "anchor_pool",
                       **{field: sum(r[field] for r in blocks)
                          for field in ("count", "seconds", "projection_seconds")}})
        for role in ("target_selection", "target_validation"):
            rows = [r for r in samples if (r["domain"], r["role"]) == (domain, role)]
            groups.append(profile_rows(rows, domain, role))
    costs = []
    for domain in domains:
        source = sum(r["seconds"] for r in groups if r["role"] == "anchor_pool" and r["domain"] != domain)
        val = next(r["seconds"] for r in groups if r["role"] == "target_validation" and r["domain"] == domain)
        selection = next(r["seconds"] for r in groups if r["role"] == "target_selection" and r["domain"] == domain)
        costs.append({"target_domain": domain, "baseline_feature_seconds": model_load_seconds + source + val,
                      "target_aware_feature_seconds": model_load_seconds + source + val + selection})
    write_json_atomic(output / "profile.json", {
        "status": "complete", "encoder": encoder, "encoder_identity": observed,
        "groups": groups, "source_blocks": source_blocks, "model_load_seconds": model_load_seconds, "per_target_accounting": costs,
        "scope": "measured-additive-raw-image-feature-accounting-not-a-replayed-end-to-end-pipeline",
        "excludes": ["download", "pool-construction", "feature-cache-serialization", "test-images", "test-audit"],
        "includes": ["image-read-hash-check", "preprocessing", "loader-workers", "GPU-forward", "CPU-feature-transfer", "normalization-and-projection"],
        "disk_cache": "operating-system-cache-not-flushed", "source_code_sha256": sha256_file(Path(__file__)),
        "environment": {"torch": torch.__version__, "numpy": np.__version__,
                        "scipy": version("scipy"), "scikit-learn": version("scikit-learn"),
                        "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                        "torch_threads": torch.get_num_threads(), "loader_workers": 4,
                        "batch_size": 64, "cpu_info": Path("/proc/cpuinfo").read_text().split("model name", 1)[1].splitlines()[0].split(":", 1)[-1].strip()},
        "input_screen_manifest_sha256": sha256_file(work / "stage_bundles" / encoder / "screen/manifest.json")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--encoder", choices=("resnet50", "dinov2_b14"), required=True)
    a = p.parse_args()
    run(a.work_root, a.encoder, a.output_root)
