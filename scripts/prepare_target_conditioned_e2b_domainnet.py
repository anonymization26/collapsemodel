#!/usr/bin/env python3
"""Download DomainNet and build the frozen, extracted E2b sample manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2b import (  # noqa: E402
    MANIFEST_SCHEMA,
    SAMPLE_FIELDS,
    E2BArtifactError,
    canonical_json_sha256,
    hash_fields,
    load_config,
    sample_order_key,
    selected_class_names,
    sha256_file,
    validate_manifest,
    write_csv_atomic,
    write_json_atomic,
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            url,
        ],
        check=True,
    )


def download_inputs(config_path: Path, download_dir: Path) -> None:
    config = load_config(config_path)
    dataset = config["dataset"]
    domains = [str(value) for value in dataset["domains"]]
    archive_urls = {
        str(domain): str(url) for domain, url in dataset["archive_urls"].items()
    }
    archive_suffix = str(dataset["archive_local_suffix"])
    if set(archive_urls) != set(domains):
        raise E2BArtifactError("archive_urls must contain exactly the configured domains")
    split_base = str(dataset["split_base_url"]).rstrip("/")
    for domain in domains:
        _download(archive_urls[domain], download_dir / f"{domain}{archive_suffix}")
        for split in ("train", "test"):
            filename = f"{domain}_{split}.txt"
            _download(f"{split_base}/{filename}", download_dir / filename)


def _parse_split(path: Path, domain: str, split: str) -> list[dict[str, object]]:
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            member, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)
        except ValueError as error:
            raise E2BArtifactError(
                f"invalid split row {path.name}:{line_number}"
            ) from error
        parts = Path(member).parts
        if len(parts) < 3 or parts[0] != domain:
            raise E2BArtifactError(
                f"unexpected DomainNet path {member} in {path.name}"
            )
        records.append(
            {
                "domain": domain,
                "official_split": split,
                "archive_member": member,
                "class_name": parts[1],
                "official_class_id": label,
            }
        )
    if not records:
        raise E2BArtifactError(f"empty split file: {path.name}")
    return records


def _validate_class_vocabulary(
    records: list[dict[str, object]], expected_count: int
) -> dict[str, int]:
    observed: dict[str, set[int]] = {}
    for row in records:
        observed.setdefault(str(row["class_name"]), set()).add(
            int(row["official_class_id"])
        )
    inconsistent = {name: values for name, values in observed.items() if len(values) != 1}
    if inconsistent:
        raise E2BArtifactError("DomainNet class names map to inconsistent labels")
    if len(observed) != expected_count:
        raise E2BArtifactError(
            f"expected {expected_count} DomainNet classes, found {len(observed)}"
        )
    return {name: next(iter(values)) for name, values in observed.items()}


def _valid_image_bytes(archive: zipfile.ZipFile, member: str) -> bytes | None:
    try:
        payload = archive.read(member)
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        return payload
    except (
        KeyError,
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        Image.DecompressionBombError,
    ):
        return None


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _select_role(
    *,
    records: list[dict[str, object]],
    archive: zipfile.ZipFile,
    archive_filename: str,
    extracted_root: Path,
    dataset_name: str,
    sampling_seed: str,
    domain: str,
    role: str,
    quota: int,
    used_members: set[str],
    seen_content: set[str],
    class_to_id: Mapping[str, int],
    rejected: Counter[str],
) -> list[dict[str, object]]:
    ordered = sorted(
        records,
        key=lambda row: sample_order_key(
            sampling_seed,
            role,
            domain,
            str(row["archive_member"]),
        ),
    )
    selected = []
    for row in ordered:
        member = str(row["archive_member"])
        if member in used_members:
            continue
        payload = _valid_image_bytes(archive, member)
        if payload is None:
            rejected["invalid_or_missing_image"] += 1
            used_members.add(member)
            continue
        content_hash = hashlib.sha256(payload).hexdigest()
        if content_hash in seen_content:
            rejected["duplicate_content"] += 1
            used_members.add(member)
            continue
        class_name = str(row["class_name"])
        relative = Path(
            domain,
            str(row["official_split"]),
            class_name,
            Path(member).name,
        )
        output = extracted_root / relative
        if output.exists():
            if output.read_bytes() != payload:
                raise E2BArtifactError(
                    f"existing extracted sample has different content: {relative}"
                )
        else:
            _write_bytes_atomic(output, payload)
        used_members.add(member)
        seen_content.add(content_hash)
        selected.append(
            {
                "sample_index": -1,
                "sample_id": hash_fields(
                    "domainnet-e2b-sample-v1",
                    dataset_name,
                    domain,
                    row["official_split"],
                    member,
                    content_hash,
                ),
                "domain": domain,
                "official_split": row["official_split"],
                "role": role,
                "class_name": class_name,
                "class_id": class_to_id[class_name],
                "archive_filename": archive_filename,
                "archive_member": member,
                "relative_path": relative.as_posix(),
                "content_sha256": content_hash,
                "byte_size": len(payload),
            }
        )
        if len(selected) == quota:
            return selected
    raise E2BArtifactError(
        f"insufficient valid unique images for {domain}/{role}: "
        f"needed={quota}, found={len(selected)}"
    )


def build_manifest(
    config_path: Path,
    download_dir: Path,
    extracted_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    if (output_dir / "manifest.json").is_file():
        validate_manifest(output_dir, config_path)
        return json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
    if (output_dir / "samples.csv").exists():
        raise E2BArtifactError("partial manifest table exists without manifest.json")
    config = load_config(config_path)
    dataset = config["dataset"]
    sampling = config["sampling"]
    domains = [str(value) for value in dataset["domains"]]
    all_records: list[dict[str, object]] = []
    records_by_domain_split: dict[tuple[str, str], list[dict[str, object]]] = {}
    for domain in domains:
        for split in ("train", "test"):
            path = download_dir / f"{domain}_{split}.txt"
            rows = _parse_split(path, domain, split)
            records_by_domain_split[(domain, split)] = rows
            all_records.extend(rows)

    official_mapping = _validate_class_vocabulary(
        all_records, int(dataset["expected_class_count"])
    )
    subset_config = dataset["class_subset"]
    selected_classes = selected_class_names(
        list(official_mapping),
        int(subset_config["count"]),
        str(subset_config["seed"]),
    )
    selected_set = set(selected_classes)
    class_to_id = {name: index for index, name in enumerate(selected_classes)}
    for key, rows in records_by_domain_split.items():
        records_by_domain_split[key] = [
            row for row in rows if str(row["class_name"]) in selected_set
        ]

    source_artifacts = []
    archive_urls = {
        str(domain): str(url) for domain, url in dataset["archive_urls"].items()
    }
    archive_suffix = str(dataset["archive_local_suffix"])
    if set(archive_urls) != set(domains):
        raise E2BArtifactError("archive_urls must contain exactly the configured domains")
    split_base = str(dataset["split_base_url"]).rstrip("/")
    for domain in domains:
        for filename, url in (
            (f"{domain}{archive_suffix}", archive_urls[domain]),
            (f"{domain}_train.txt", f"{split_base}/{domain}_train.txt"),
            (f"{domain}_test.txt", f"{split_base}/{domain}_test.txt"),
        ):
            path = download_dir / filename
            if not path.is_file():
                raise E2BArtifactError(f"missing source artifact {filename}")
            source_artifacts.append(
                {
                    "filename": filename,
                    "url": url,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    selected_rows: list[dict[str, object]] = []
    seen_content: set[str] = set()
    rejected: Counter[str] = Counter()
    train_roles = (
        ("target_selection", int(sampling["target_selection_per_domain"])),
        ("target_validation", int(sampling["target_validation_per_domain"])),
        ("anchor_pool", int(sampling["anchor_pool_per_domain"])),
    )
    for domain in domains:
        archive_path = download_dir / f"{domain}{archive_suffix}"
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            required = {
                str(row["archive_member"])
                for split in ("train", "test")
                for row in records_by_domain_split[(domain, split)]
            }
            missing = required - members
            if missing:
                example = min(missing)
                raise E2BArtifactError(
                    f"archive {archive_path.name} misses split member {example}"
                )
            used_members: set[str] = set()
            for role, quota in train_roles:
                selected_rows.extend(
                    _select_role(
                        records=records_by_domain_split[(domain, "train")],
                        archive=archive,
                        archive_filename=archive_path.name,
                        extracted_root=extracted_root,
                        dataset_name=str(dataset["name"]),
                        sampling_seed=str(sampling["seed"]),
                        domain=domain,
                        role=role,
                        quota=quota,
                        used_members=used_members,
                        seen_content=seen_content,
                        class_to_id=class_to_id,
                        rejected=rejected,
                    )
                )
            selected_rows.extend(
                _select_role(
                    records=records_by_domain_split[(domain, "test")],
                    archive=archive,
                    archive_filename=archive_path.name,
                    extracted_root=extracted_root,
                    dataset_name=str(dataset["name"]),
                    sampling_seed=str(sampling["seed"]),
                    domain=domain,
                    role="target_test",
                    quota=int(sampling["target_test_per_domain"]),
                    used_members=used_members,
                    seen_content=seen_content,
                    class_to_id=class_to_id,
                    rejected=rejected,
                )
            )
            print(f"prepared domain={domain}", flush=True)

    for index, row in enumerate(selected_rows):
        row["sample_index"] = index
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.csv"
    write_csv_atomic(samples_path, SAMPLE_FIELDS, selected_rows)
    role_counts = Counter(
        (str(row["domain"]), str(row["role"])) for row in selected_rows
    )
    core: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": dataset["name"],
        "dataset_version": dataset["version"],
        "config_file_sha256": sha256_file(config_path),
        "source_artifacts": source_artifacts,
        "selected_classes": selected_classes,
        "selected_classes_sha256": canonical_json_sha256(selected_classes),
        "class_to_id": class_to_id,
        "sample_count": len(selected_rows),
        "samples_file_sha256": sha256_file(samples_path),
        "role_counts": {
            domain: {
                role: role_counts[(domain, role)]
                for role in (
                    "target_selection",
                    "target_validation",
                    "anchor_pool",
                    "target_test",
                )
            }
            for domain in domains
        },
        "rejected_before_freeze": dict(sorted(rejected.items())),
        "assignment_access": {
            "uses_class_names_only_for_frozen_class_subset": True,
            "uses_labels_or_model_outputs": False,
            "official_test_is_a_distinct_source_split": True,
        },
    }
    manifest = {**core, "manifest_id": canonical_json_sha256(core)}
    write_json_atomic(output_dir / "manifest.json", manifest)
    validate_manifest(output_dir, config_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--config", type=Path, required=True)
    download.add_argument("--download-dir", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--download-dir", type=Path, required=True)
    build.add_argument("--extracted-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        download_inputs(args.config, args.download_dir)
        print(json.dumps({"status": "downloaded"}, sort_keys=True))
        return
    manifest = build_manifest(
        args.config,
        args.download_dir,
        args.extracted_root,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "manifest-frozen",
                "manifest_id": manifest["manifest_id"],
                "sample_count": manifest["sample_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
