#!/usr/bin/env python3
"""Run E2b using only role-separated inputs and frozen upstream artifacts."""

import argparse
import json
from pathlib import Path

from run_target_conditioned_e2b_shortlist import run_screen, run_validation, run_test_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "validate", "test-audit"))
    parser.add_argument("--stage-bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    args = parser.parse_args()
    if args.stage != "screen" and args.screen_dir is None:
        parser.error("--screen-dir is required for validation and audit")
    if args.stage == "test-audit" and args.validation_dir is None:
        parser.error("--validation-dir is required for test audit")
    # No full-cache, original-manifest, or anchor path is supplied to the worker.
    common = (None, args.config, None, None, None, None, None, args.encoder)
    options = {"stage_bundle": args.stage_bundle}
    if args.stage == "screen":
        result = run_screen(*common, args.output_dir, **options)
    elif args.stage == "validate":
        result = run_validation(*common, args.screen_dir, args.output_dir, **options)
    else:
        result = run_test_audit(*common, args.screen_dir, args.validation_dir,
                                args.output_dir, **options)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
