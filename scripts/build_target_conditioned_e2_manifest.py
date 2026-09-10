#!/usr/bin/env python3
"""Build a verified E2 dataset manifest from an extracted image tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2 import build_manifest_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--source-artifact",
        type=Path,
        action="append",
        required=True,
        dest="source_artifacts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-seed",
        default="target-conditioned-e2-v1",
    )
    parser.add_argument(
        "--blocks-per-source-domain",
        type=int,
        default=4,
    )
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest_bundle(
        dataset_root=args.dataset_root,
        receipt_path=args.receipt,
        source_artifact_paths=args.source_artifacts,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        blocks_per_source_domain=args.blocks_per_source_domain,
        verify_images=args.verify_images,
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "dataset": manifest["dataset"],
                "manifest_id": manifest["manifest_id"],
                "sample_count": manifest["sample_count"],
                "domains": manifest["domains"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
