#!/usr/bin/env python3
"""Verify prepared images, encoder caches, and all six isolated worker inputs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from metrics.e2b_stage_bundle import load_stage_bundle
from metrics.target_conditioned_e2b import (
    read_manifest_samples, sha256_file, validate_feature_cache, write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = ROOT / "results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1"
    config = ROOT / "code/configs/target_conditioned_e2b/domainnet_v1.json"
    manifest = frozen / "data_manifest"
    environment = json.loads((args.work_root / "environment_check.json").read_text())
    if environment.get("status") != "passed":
        raise RuntimeError("environment check has not passed")
    samples = read_manifest_samples(manifest)
    for row in samples:
        path = args.work_root / "dataset" / row["relative_path"]
        if path.stat().st_size != row["byte_size"] or sha256_file(path) != row["content_sha256"]:
            raise RuntimeError(f"prepared image differs from frozen manifest: {row['sample_id']}")
    caches, bundles = {}, {}
    for encoder in ("resnet50", "dinov2_b14", "clip_b32"):
        directory = args.work_root / "reconstructed_features" / encoder
        caches[encoder] = validate_feature_cache(directory / "features.npz", directory / "metadata.json",
                                                 manifest, config, encoder)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("reference_metadata_sha256") != sha256_file(frozen / "features" / encoder / "metadata.json"):
            raise RuntimeError("reconstructed cache lacks the reference identity check")
        runtime = metadata.get("runtime", {})
        if not runtime.get("device", "").startswith("cuda:") or runtime.get("tf32_enabled") is not False:
            raise RuntimeError("reconstructed cache is not from the registered CUDA precision profile")
        if runtime.get("sdpa_backend") != "math" or runtime.get("xformers_disabled") is not True:
            raise RuntimeError("reconstructed cache uses an unexpected attention implementation")
    for encoder in ("resnet50", "dinov2_b14"):
        for stage in ("screen", "validate", "test-audit"):
            directory = args.work_root / "stage_bundles" / encoder / stage
            loaded = load_stage_bundle(directory, config, stage, encoder)
            if loaded["metadata"]["source_feature_file_sha256"] != caches[encoder]["feature_file_sha256"]:
                raise RuntimeError("stage bundle belongs to a different reconstructed cache")
            bundles[f"{encoder}/{stage}"] = {
                "bundle_id": loaded["metadata"]["bundle_id"],
                "samples": len(loaded["samples"]), "has_labels": loaded["labels"] is not None,
            }
    code_paths = [ROOT / "code/metrics/e2b_stage_bundle.py"] + [
        ROOT / "scripts" / name for name in (
            "materialize_target_conditioned_e2b_parquet.py", "benchmark_two_stage_encoders_npu.py",
            "extract_target_conditioned_e2b_features_npu.py", "extract_target_conditioned_e2b_features.py",
            "prepare_target_conditioned_e2b_stage_bundles.py", "run_target_conditioned_e2b_shortlist.py",
            "run_target_conditioned_e2b_isolated.py", "prepare_target_conditioned_e2b_cuda.sh",
            "check_target_conditioned_e2b_cuda.py", "check_target_conditioned_e2b_preparation.py",
            "run_target_conditioned_e2b_preparation.py",
        )
    ]
    report = {"status": "ready", "sample_count": len(samples),
              "image_bytes": sum(row["byte_size"] for row in samples),
              "caches": caches, "stage_bundles": bundles,
              "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
              "real_outcome_experiments_started": False,
              "note": "CUDA reconstruction, not a claim of bitwise equality with archived NPU features."}
    write_json_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
