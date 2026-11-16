#!/usr/bin/env python3
"""Validate E2 dataset manifests or frozen-feature caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2 import (  # noqa: E402
    validate_feature_cache,
    validate_manifest_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--manifest-dir", type=Path, required=True)
    manifest_parser.add_argument("--dataset-root", type=Path)
    manifest_parser.add_argument(
        "--source-artifact",
        type=Path,
        action="append",
        default=[],
        dest="source_artifacts",
    )
    manifest_parser.add_argument("--verify-images", action="store_true")

    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--manifest-dir", type=Path, required=True)
    cache_parser.add_argument("--features", type=Path, required=True)
    cache_parser.add_argument("--metadata", type=Path, required=True)
    cache_parser.add_argument("--expected-encoder")
    args = parser.parse_args()

    if args.command == "manifest":
        report = validate_manifest_bundle(
            args.manifest_dir,
            dataset_root=args.dataset_root,
            source_artifact_paths=args.source_artifacts,
            verify_images=args.verify_images,
        )
    else:
        report = validate_feature_cache(
            args.features,
            args.metadata,
            args.manifest_dir,
            expected_encoder=args.expected_encoder,
        )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
