#!/usr/bin/env python3
"""Create an E2 receipt with measured source-artifact hashes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2 import (  # noqa: E402
    RECEIPT_SCHEMA,
    sha256_file,
    validate_receipt,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--official-page-url", required=True)
    parser.add_argument(
        "--download-url",
        action="append",
        required=True,
        dest="download_urls",
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        action="append",
        required=True,
        dest="source_artifacts",
    )
    parser.add_argument(
        "--expected-domain",
        action="append",
        required=True,
        dest="expected_domains",
    )
    parser.add_argument("--expected-class-count", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int)
    parser.add_argument("--usage-summary", required=True)
    parser.add_argument("--usage-terms-url", required=True)
    parser.add_argument("--citation-key", required=True)
    parser.add_argument(
        "--citation-bibtex-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    domains = sorted(args.expected_domains)
    if len(set(domains)) != len(domains):
        parser.error("--expected-domain values must be unique")
    artifacts = []
    for path in sorted(
        args.source_artifacts,
        key=lambda item: item.name,
    ):
        if not path.is_file():
            parser.error(f"source artifact is missing: {path}")
        artifacts.append(
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "dataset": args.dataset,
        "dataset_version": args.dataset_version,
        "official_page_url": args.official_page_url,
        "download_urls": args.download_urls,
        "source_artifacts": artifacts,
        "expected_domains": domains,
        "expected_class_count": args.expected_class_count,
        "expected_sample_count": args.expected_sample_count,
        "usage_terms": {
            "summary": args.usage_summary,
            "url": args.usage_terms_url,
        },
        "citation": {
            "key": args.citation_key,
            "bibtex": args.citation_bibtex_file.read_text(
                encoding="utf-8"
            ).strip(),
        },
    }
    validate_receipt(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, receipt)
    print(
        f"dataset={args.dataset} artifacts={len(artifacts)} "
        f"receipt={args.output.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
