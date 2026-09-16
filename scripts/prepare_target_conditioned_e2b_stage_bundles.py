#!/usr/bin/env python3
"""Prepare role-restricted inputs outside the experimental worker processes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics.e2b_stage_bundle import prepare_stage_bundles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-dir", "config", "candidates", "features", "metadata", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    args = parser.parse_args()
    result = prepare_stage_bundles(args.manifest_dir, args.config, args.candidates,
                                  args.features, args.metadata, args.encoder, args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
