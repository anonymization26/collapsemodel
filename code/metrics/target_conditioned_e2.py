"""Auditable dataset manifests and frozen-feature caches for E2.

Split and source-block assignments consume only dataset, domain, and image
content hashes. Class labels are retained for evaluation, but are never
arguments to the assignment rules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


RECEIPT_SCHEMA = "target-conditioned-e2-receipt-v1"
MANIFEST_SCHEMA = "target-conditioned-e2-dataset-v1"
CACHE_SCHEMA = "target-conditioned-e2-feature-cache-v1"
SPLIT_ROLES = ("target_selection", "target_calibration", "target_test")
SPLIT_BASIS_POINTS = {
    "target_selection": 2000,
    "target_calibration": 1000,
    "target_test": 7000,
}
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
SAMPLE_FIELDS = (
    "sample_index",
    "sample_id",
    "domain",
    "class_name",
    "class_id",
    "relative_path",
    "content_sha256",
    "byte_size",
)
ASSIGNMENT_FIELDS = ("target_domain", "sample_id", "role", "candidate_id")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class E2ArtifactError(ValueError):
    """Raised when an E2 artifact violates the auditable-data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_fields(*fields: object) -> str:
    digest = hashlib.sha256()
    for field in fields:
        encoded = str(field).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2ArtifactError(
            f"cannot read JSON object {path.name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise E2ArtifactError(f"{path.name} must contain a JSON object")
    return value


def _require_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise E2ArtifactError(f"{key} must be a nonempty string")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise E2ArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


def validate_receipt(receipt: Mapping[str, object]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise E2ArtifactError(f"receipt schema must be {RECEIPT_SCHEMA}")
    dataset = _require_string(receipt, "dataset")
    if Path(dataset).name != dataset:
        raise E2ArtifactError(
            "dataset must be a path-free identifier"
        )
    _require_string(receipt, "dataset_version")
    official_page = _require_string(receipt, "official_page_url")
    if not official_page.startswith("https://"):
        raise E2ArtifactError("official_page_url must use HTTPS")

    urls = receipt.get("download_urls")
    if not isinstance(urls, list) or not urls:
        raise E2ArtifactError("download_urls must be a nonempty list")
    if any(
        not isinstance(url, str) or not url.startswith("https://")
        for url in urls
    ):
        raise E2ArtifactError("all download_urls must use HTTPS")

    domains = receipt.get("expected_domains")
    if (
        not isinstance(domains, list)
        or len(domains) < 2
        or any(
            not isinstance(domain, str) or not domain for domain in domains
        )
        or len(set(domains)) != len(domains)
    ):
        raise E2ArtifactError(
            "expected_domains must contain unique nonempty names"
        )
    if domains != sorted(domains):
        raise E2ArtifactError("expected_domains must be sorted")

    class_count = receipt.get("expected_class_count")
    if not isinstance(class_count, int) or class_count <= 1:
        raise E2ArtifactError(
            "expected_class_count must be an integer greater than one"
        )
    sample_count = receipt.get("expected_sample_count")
    if sample_count is not None and (
        not isinstance(sample_count, int) or sample_count <= 0
    ):
        raise E2ArtifactError(
            "expected_sample_count must be null or a positive integer"
        )

    artifacts = receipt.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise E2ArtifactError("source_artifacts must be a nonempty list")
    names = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise E2ArtifactError(
                f"source_artifacts[{index}] must be an object"
            )
        filename = _require_string(artifact, "filename")
        if Path(filename).name != filename:
            raise E2ArtifactError(
                "source artifact filenames must not contain directories"
            )
        names.append(filename)
        _validate_sha256(
            artifact.get("sha256"),
            f"source_artifacts[{index}].sha256",
        )
        byte_size = artifact.get("byte_size")
        if not isinstance(byte_size, int) or byte_size <= 0:
            raise E2ArtifactError(
                f"source_artifacts[{index}].byte_size must be positive"
            )
    if len(set(names)) != len(names):
        raise E2ArtifactError("source artifact filenames must be unique")

    usage_terms = receipt.get("usage_terms")
    if not isinstance(usage_terms, dict) or not usage_terms:
        raise E2ArtifactError("usage_terms must be a nonempty object")
    _require_string(usage_terms, "summary")
    terms_url = _require_string(usage_terms, "url")
    if not terms_url.startswith("https://"):
        raise E2ArtifactError("usage_terms.url must use HTTPS")

    citation = receipt.get("citation")
    if not isinstance(citation, dict) or not citation:
        raise E2ArtifactError("citation must be a nonempty object")
    _require_string(citation, "key")
    _require_string(citation, "bibtex")


def verify_source_artifacts(
    receipt: Mapping[str, object],
    artifact_paths: Sequence[Path],
) -> List[Dict[str, object]]:
    artifacts = receipt.get("source_artifacts")
    if not isinstance(artifacts, list):
        raise E2ArtifactError("receipt source_artifacts is invalid")
    expected = {
        str(item["filename"]): item
        for item in artifacts
        if isinstance(item, dict)
    }
    provided = {path.name: path for path in artifact_paths}
    if len(provided) != len(artifact_paths):
        raise E2ArtifactError("source artifact basenames must be unique")
    if set(provided) != set(expected):
        raise E2ArtifactError(
            "source artifacts differ from receipt: "
            f"expected={sorted(expected)}, provided={sorted(provided)}"
        )

    verified = []
    for filename in sorted(expected):
        path = provided[filename]
        if not path.is_file():
            raise E2ArtifactError(
                f"source artifact is missing: {filename}"
            )
        byte_size = path.stat().st_size
        digest = sha256_file(path)
        expected_item = expected[filename]
        if byte_size != expected_item["byte_size"]:
            raise E2ArtifactError(
                f"source artifact size mismatch: {filename}"
            )
        if digest != expected_item["sha256"]:
            raise E2ArtifactError(
                f"source artifact hash mismatch: {filename}"
            )
        verified.append(
            {
                "filename": filename,
                "byte_size": byte_size,
                "sha256": digest,
            }
        )
    return verified


def _visible_children(path: Path) -> List[Path]:
    return sorted(
        child for child in path.iterdir() if not child.name.startswith(".")
    )


def _verify_image(path: Path) -> None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise E2ArtifactError(
            f"image decode failed: {path.name}: {error}"
        ) from error


def sample_id(
    dataset: str,
    domain: str,
    relative_path: str,
    content_sha256: str,
) -> str:
    return _hash_fields(
        "sample-v1", dataset, domain, relative_path, content_sha256
    )


def discover_samples(
    dataset_root: Path,
    receipt: Mapping[str, object],
    verify_images: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Enumerate a strict domain/class/image dataset tree."""

    validate_receipt(receipt)
    if not dataset_root.is_dir():
        raise E2ArtifactError("dataset root is not a directory")
    dataset = str(receipt["dataset"])
    expected_domains = [
        str(value) for value in receipt["expected_domains"]
    ]
    domain_dirs = [
        child for child in _visible_children(dataset_root) if child.is_dir()
    ]
    actual_domains = [child.name for child in domain_dirs]
    if actual_domains != expected_domains:
        raise E2ArtifactError(
            "domain directories differ: "
            f"expected={expected_domains}, actual={actual_domains}"
        )

    classes_by_domain: Dict[str, List[str]] = {}
    files_by_domain_class: Dict[Tuple[str, str], List[Path]] = {}
    for domain in expected_domains:
        domain_root = dataset_root / domain
        children = _visible_children(domain_root)
        outside_files = [
            child.name for child in children if not child.is_dir()
        ]
        if outside_files:
            raise E2ArtifactError(
                f"domain {domain} contains files outside class directories: "
                f"{outside_files[:5]}"
            )
        class_names = [child.name for child in children]
        if not class_names:
            raise E2ArtifactError(
                f"domain {domain} contains no class directories"
            )
        classes_by_domain[domain] = class_names
        for class_name in class_names:
            class_root = domain_root / class_name
            paths = []
            for path in sorted(class_root.rglob("*")):
                if path.name.startswith(".") or path.is_dir():
                    continue
                if path.is_symlink():
                    raise E2ArtifactError(
                        f"symbolic links are not allowed: {path.name}"
                    )
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    relative = path.relative_to(dataset_root).as_posix()
                    raise E2ArtifactError(
                        f"unsupported file below class directory: {relative}"
                    )
                paths.append(path)
            if not paths:
                raise E2ArtifactError(
                    f"empty class directory: {domain}/{class_name}"
                )
            files_by_domain_class[(domain, class_name)] = paths

    reference_classes = classes_by_domain[expected_domains[0]]
    for domain, class_names in classes_by_domain.items():
        if class_names != reference_classes:
            raise E2ArtifactError(
                f"class vocabulary for {domain} differs from "
                f"{expected_domains[0]}"
            )
    expected_class_count = int(receipt["expected_class_count"])
    if len(reference_classes) != expected_class_count:
        raise E2ArtifactError(
            f"expected {expected_class_count} classes, "
            f"found {len(reference_classes)}"
        )
    class_to_id = {
        class_name: index
        for index, class_name in enumerate(reference_classes)
    }

    samples: List[Dict[str, object]] = []
    for domain in expected_domains:
        for class_name in reference_classes:
            for path in files_by_domain_class[(domain, class_name)]:
                relative_path = path.relative_to(dataset_root).as_posix()
                if verify_images:
                    _verify_image(path)
                content_hash = sha256_file(path)
                samples.append(
                    {
                        "sample_index": len(samples),
                        "sample_id": sample_id(
                            dataset,
                            domain,
                            relative_path,
                            content_hash,
                        ),
                        "domain": domain,
                        "class_name": class_name,
                        "class_id": class_to_id[class_name],
                        "relative_path": relative_path,
                        "content_sha256": content_hash,
                        "byte_size": path.stat().st_size,
                    }
                )

    expected_sample_count = receipt.get("expected_sample_count")
    if (
        expected_sample_count is not None
        and len(samples) != expected_sample_count
    ):
        raise E2ArtifactError(
            f"expected {expected_sample_count} samples, "
            f"found {len(samples)}"
        )
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise E2ArtifactError("sample IDs are not unique")
    return samples, class_to_id


def _validate_split_basis_points(
    split_basis_points: Mapping[str, int],
) -> None:
    if set(split_basis_points) != set(SPLIT_ROLES):
        raise E2ArtifactError(
            f"split roles must be exactly {list(SPLIT_ROLES)}"
        )
    values = [split_basis_points[role] for role in SPLIT_ROLES]
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise E2ArtifactError(
            "split basis points must be positive integers"
        )
    if sum(values) != 10_000:
        raise E2ArtifactError("split basis points must sum to 10000")


def target_role(
    dataset: str,
    domain: str,
    content_sha256: str,
    split_seed: str,
    split_basis_points: Mapping[str, int] = SPLIT_BASIS_POINTS,
) -> str:
    """Assign a target role without receiving a label or model output."""

    _validate_split_basis_points(split_basis_points)
    bucket = int(
        _hash_fields(
            "target-split-v1",
            split_seed,
            dataset,
            domain,
            content_sha256,
        )[:16],
        16,
    ) % 10_000
    cumulative = 0
    for role in SPLIT_ROLES:
        cumulative += split_basis_points[role]
        if bucket < cumulative:
            return role
    raise AssertionError("unreachable split bucket")


def source_candidate_id(
    dataset: str,
    domain: str,
    content_sha256: str,
    split_seed: str,
    blocks_per_source_domain: int,
) -> str:
    """Assign a source block without receiving a label or model output."""

    if blocks_per_source_domain < 2:
        raise E2ArtifactError(
            "blocks_per_source_domain must be at least two"
        )
    block = int(
        _hash_fields(
            "source-block-v1",
            split_seed,
            dataset,
            domain,
            content_sha256,
        )[:16],
        16,
    ) % blocks_per_source_domain
    return f"{domain}__block_{block:02d}"


def build_assignments(
    samples: Sequence[Mapping[str, object]],
    dataset: str,
    target_domains: Sequence[str],
    split_seed: str,
    blocks_per_source_domain: int,
    split_basis_points: Mapping[str, int] = SPLIT_BASIS_POINTS,
) -> List[Dict[str, str]]:
    assignments = []
    for target_domain in target_domains:
        for row in samples:
            domain = str(row["domain"])
            content_hash = str(row["content_sha256"])
            if domain == target_domain:
                role = target_role(
                    dataset,
                    domain,
                    content_hash,
                    split_seed,
                    split_basis_points,
                )
                candidate_id = ""
            else:
                role = "source_candidate"
                candidate_id = source_candidate_id(
                    dataset,
                    domain,
                    content_hash,
                    split_seed,
                    blocks_per_source_domain,
                )
            assignments.append(
                {
                    "target_domain": target_domain,
                    "sample_id": str(row["sample_id"]),
                    "role": role,
                    "candidate_id": candidate_id,
                }
            )
    return assignments


def _tree_sha256(samples: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in samples:
        record = (
            f"{row['relative_path']}\0{row['content_sha256']}\0"
            f"{row['byte_size']}\n"
        )
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    temporary.replace(path)


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _nested_counts(
    rows: Iterable[Mapping[str, object]],
    keys: Sequence[str],
) -> Dict[str, object]:
    counts = Counter(
        tuple(str(row[key]) for key in keys)
        for row in rows
    )
    result: Dict[str, object] = {}
    for compound_key, count in sorted(counts.items()):
        cursor = result
        for key in compound_key[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise AssertionError("count key collision")
            cursor = child
        cursor[compound_key[-1]] = count
    return result


def build_manifest_bundle(
    dataset_root: Path,
    receipt_path: Path,
    source_artifact_paths: Sequence[Path],
    output_dir: Path,
    split_seed: str = "target-conditioned-e2-v1",
    blocks_per_source_domain: int = 4,
    verify_images: bool = False,
) -> Dict[str, object]:
    try:
        output_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        pass
    else:
        raise E2ArtifactError(
            "manifest output directory must be outside the dataset root"
        )
    receipt = _read_json(receipt_path)
    validate_receipt(receipt)
    verified_artifacts = verify_source_artifacts(
        receipt,
        source_artifact_paths,
    )
    samples, class_to_id = discover_samples(
        dataset_root,
        receipt,
        verify_images,
    )
    target_domains = [
        str(value) for value in receipt["expected_domains"]
    ]
    assignments = build_assignments(
        samples,
        str(receipt["dataset"]),
        target_domains,
        split_seed,
        blocks_per_source_domain,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_output = output_dir / "receipt.json"
    samples_output = output_dir / "samples.csv"
    assignments_output = output_dir / "assignments.csv"
    write_json_atomic(receipt_output, receipt)
    _write_csv(samples_output, SAMPLE_FIELDS, samples)
    _write_csv(assignments_output, ASSIGNMENT_FIELDS, assignments)

    core = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": receipt["dataset"],
        "dataset_version": receipt["dataset_version"],
        "receipt_sha256": sha256_file(receipt_output),
        "source_artifacts": verified_artifacts,
        "dataset_tree_sha256": _tree_sha256(samples),
        "samples_file_sha256": sha256_file(samples_output),
        "assignments_file_sha256": sha256_file(assignments_output),
        "sample_count": len(samples),
        "domains": target_domains,
        "class_to_id": class_to_id,
        "assignment_protocol": {
            "split_seed": split_seed,
            "split_basis_points": dict(SPLIT_BASIS_POINTS),
            "blocks_per_source_domain": blocks_per_source_domain,
            "uses_labels": False,
            "duplicate_content_policy": (
                "same-domain-identical-content-shares-assignment"
            ),
        },
        "sample_counts": {
            "by_domain": _nested_counts(samples, ("domain",)),
            "by_domain_and_class": _nested_counts(
                samples,
                ("domain", "class_name"),
            ),
        },
        "assignment_counts": {
            "by_target_and_role": _nested_counts(
                assignments,
                ("target_domain", "role"),
            ),
            "by_target_and_candidate": _nested_counts(
                (
                    row
                    for row in assignments
                    if row["candidate_id"]
                ),
                ("target_domain", "candidate_id"),
            ),
        },
    }
    manifest = {
        **core,
        "manifest_id": canonical_json_sha256(core),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    validate_manifest_bundle(
        output_dir,
        dataset_root=dataset_root,
        source_artifact_paths=source_artifact_paths,
        verify_images=verify_images,
    )
    return manifest


def _read_csv(
    path: Path,
    fields: Sequence[str],
) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise E2ArtifactError(
                    f"{path.name} columns differ: "
                    f"expected={list(fields)}, actual={reader.fieldnames}"
                )
            return [dict(row) for row in reader]
    except OSError as error:
        raise E2ArtifactError(
            f"cannot read {path.name}: {error}"
        ) from error


def read_manifest_samples(
    bundle_dir: Path,
) -> List[Dict[str, object]]:
    rows = _read_csv(bundle_dir / "samples.csv", SAMPLE_FIELDS)
    parsed = []
    for row in rows:
        try:
            parsed.append(
                {
                    **row,
                    "sample_index": int(row["sample_index"]),
                    "class_id": int(row["class_id"]),
                    "byte_size": int(row["byte_size"]),
                }
            )
        except ValueError as error:
            raise E2ArtifactError(
                "samples.csv contains a non-integer numeric field"
            ) from error
    return parsed


def _assert_no_absolute_strings(
    value: object,
    context: str = "metadata",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_absolute_strings(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_absolute_strings(
                child,
                f"{context}[{index}]",
            )
    elif isinstance(value, str) and value.startswith(("/", "file://")):
        raise E2ArtifactError(
            f"absolute path is forbidden in {context}"
        )


def validate_manifest_bundle(
    bundle_dir: Path,
    dataset_root: Optional[Path] = None,
    source_artifact_paths: Sequence[Path] = (),
    verify_images: bool = False,
) -> Dict[str, object]:
    manifest = _read_json(bundle_dir / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise E2ArtifactError(
            f"manifest schema must be {MANIFEST_SCHEMA}"
        )
    _assert_no_absolute_strings(manifest, "manifest")

    receipt_path = bundle_dir / "receipt.json"
    receipt = _read_json(receipt_path)
    validate_receipt(receipt)
    _assert_no_absolute_strings(receipt, "receipt")
    if manifest.get("receipt_sha256") != sha256_file(receipt_path):
        raise E2ArtifactError("receipt hash mismatch")
    if manifest.get("dataset") != receipt.get("dataset"):
        raise E2ArtifactError(
            "manifest and receipt dataset names differ"
        )
    if manifest.get("dataset_version") != receipt.get(
        "dataset_version"
    ):
        raise E2ArtifactError(
            "manifest and receipt dataset versions differ"
        )

    receipt_artifacts = receipt.get("source_artifacts")
    if not isinstance(receipt_artifacts, list):
        raise E2ArtifactError("receipt source artifacts are invalid")
    expected_artifact_records = sorted(
        (
            {
                "filename": str(item["filename"]),
                "byte_size": int(item["byte_size"]),
                "sha256": str(item["sha256"]),
            }
            for item in receipt_artifacts
            if isinstance(item, dict)
        ),
        key=lambda item: item["filename"],
    )
    if manifest.get("source_artifacts") != expected_artifact_records:
        raise E2ArtifactError(
            "manifest source artifact records differ from receipt"
        )
    if source_artifact_paths:
        verified = verify_source_artifacts(
            receipt,
            source_artifact_paths,
        )
        if expected_artifact_records != verified:
            raise E2ArtifactError(
                "source artifact verification differs from manifest"
            )

    samples_path = bundle_dir / "samples.csv"
    assignments_path = bundle_dir / "assignments.csv"
    samples = read_manifest_samples(bundle_dir)
    if not samples:
        raise E2ArtifactError("samples.csv is empty")
    if manifest.get("samples_file_sha256") != sha256_file(samples_path):
        raise E2ArtifactError("samples.csv hash mismatch")
    if manifest.get("assignments_file_sha256") != sha256_file(
        assignments_path
    ):
        raise E2ArtifactError("assignments.csv hash mismatch")
    if [
        row["sample_index"] for row in samples
    ] != list(range(len(samples))):
        raise E2ArtifactError(
            "sample indices must be contiguous and ordered"
        )
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise E2ArtifactError("sample IDs are not unique")

    dataset = str(manifest["dataset"])
    class_to_id = manifest.get("class_to_id")
    if not isinstance(class_to_id, dict):
        raise E2ArtifactError("class_to_id must be an object")
    domains = manifest.get("domains")
    if (
        not isinstance(domains, list)
        or domains != receipt.get("expected_domains")
    ):
        raise E2ArtifactError(
            "manifest domains differ from receipt"
        )
    domain_set = set(str(value) for value in domains)
    for row in samples:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise E2ArtifactError(
                "sample paths must be safe relative paths"
            )
        if str(row["domain"]) not in domain_set:
            raise E2ArtifactError("sample contains an unknown domain")
        _validate_sha256(
            row["content_sha256"],
            "content_sha256",
        )
        expected_id = sample_id(
            dataset,
            str(row["domain"]),
            str(row["relative_path"]),
            str(row["content_sha256"]),
        )
        if row["sample_id"] != expected_id:
            raise E2ArtifactError(
                "sample ID does not match its content record"
            )
        if class_to_id.get(str(row["class_name"])) != row["class_id"]:
            raise E2ArtifactError(
                "class ID does not match class_to_id"
            )
        if int(row["byte_size"]) <= 0:
            raise E2ArtifactError(
                "sample byte size must be positive"
            )
    if manifest.get("sample_count") != len(samples):
        raise E2ArtifactError("manifest sample count mismatch")
    if manifest.get("dataset_tree_sha256") != _tree_sha256(samples):
        raise E2ArtifactError("dataset tree hash mismatch")

    protocol = manifest.get("assignment_protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("uses_labels") is not False
    ):
        raise E2ArtifactError(
            "assignment protocol must explicitly forbid labels"
        )
    split_seed = protocol.get("split_seed")
    if not isinstance(split_seed, str) or not split_seed:
        raise E2ArtifactError(
            "assignment split seed must be a nonempty string"
        )
    split_basis_points = protocol.get("split_basis_points")
    if not isinstance(split_basis_points, dict) or any(
        not isinstance(value, int)
        for value in split_basis_points.values()
    ):
        raise E2ArtifactError(
            "split_basis_points must contain integer values"
        )
    split_values = {
        str(key): int(value)
        for key, value in split_basis_points.items()
    }
    expected_assignments = build_assignments(
        samples,
        dataset,
        [str(domain) for domain in domains],
        split_seed,
        int(protocol.get("blocks_per_source_domain", 0)),
        split_values,
    )
    assignments = _read_csv(
        assignments_path,
        ASSIGNMENT_FIELDS,
    )
    if assignments != expected_assignments:
        raise E2ArtifactError(
            "assignments.csv does not match the label-free protocol"
        )
    expected_sample_counts = {
        "by_domain": _nested_counts(samples, ("domain",)),
        "by_domain_and_class": _nested_counts(
            samples,
            ("domain", "class_name"),
        ),
    }
    expected_assignment_counts = {
        "by_target_and_role": _nested_counts(
            assignments,
            ("target_domain", "role"),
        ),
        "by_target_and_candidate": _nested_counts(
            (
                row
                for row in assignments
                if row["candidate_id"]
            ),
            ("target_domain", "candidate_id"),
        ),
    }
    if manifest.get("sample_counts") != expected_sample_counts:
        raise E2ArtifactError("manifest sample counts are incorrect")
    if manifest.get("assignment_counts") != expected_assignment_counts:
        raise E2ArtifactError(
            "manifest assignment counts are incorrect"
        )

    core = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_id"
    }
    if manifest.get("manifest_id") != canonical_json_sha256(core):
        raise E2ArtifactError("manifest identifier mismatch")

    if dataset_root is not None:
        discovered, _ = discover_samples(
            dataset_root,
            receipt,
            verify_images,
        )
        if discovered != samples:
            raise E2ArtifactError(
                "dataset tree differs from samples.csv"
            )
    return {
        "status": "valid",
        "dataset": dataset,
        "manifest_id": manifest["manifest_id"],
        "sample_count": len(samples),
        "assignment_count": len(assignments),
        "domains": domains,
        "target_role_counts": expected_assignment_counts[
            "by_target_and_role"
        ],
    }


def _validate_cache_metadata_paths(
    metadata: Mapping[str, object],
) -> None:
    forbidden = {
        "dataset_root",
        "source_path",
        "checkpoint_path",
        "output_path",
    }
    if forbidden.intersection(metadata):
        raise E2ArtifactError(
            "cache metadata contains a forbidden path field"
        )
    _assert_no_absolute_strings(metadata, "cache_metadata")


def validate_feature_cache(
    feature_path: Path,
    metadata_path: Path,
    manifest_dir: Path,
    expected_encoder: Optional[str] = None,
) -> Dict[str, object]:
    bundle_report = validate_manifest_bundle(manifest_dir)
    manifest = _read_json(manifest_dir / "manifest.json")
    samples = read_manifest_samples(manifest_dir)
    metadata = _read_json(metadata_path)
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise E2ArtifactError(
            f"cache schema must be {CACHE_SCHEMA}"
        )
    _validate_cache_metadata_paths(metadata)
    if metadata.get("dataset") != bundle_report["dataset"]:
        raise E2ArtifactError(
            "cache dataset name differs from manifest"
        )
    if metadata.get("manifest_id") != bundle_report["manifest_id"]:
        raise E2ArtifactError(
            "cache manifest identifier mismatch"
        )
    if metadata.get("manifest_file_sha256") != sha256_file(
        manifest_dir / "manifest.json"
    ):
        raise E2ArtifactError("cache manifest file hash mismatch")
    if metadata.get("samples_file_sha256") != manifest.get(
        "samples_file_sha256"
    ):
        raise E2ArtifactError("cache samples.csv hash mismatch")
    if metadata.get("assignments_file_sha256") != manifest.get(
        "assignments_file_sha256"
    ):
        raise E2ArtifactError(
            "cache assignments.csv hash mismatch"
        )
    if metadata.get("feature_file_sha256") != sha256_file(feature_path):
        raise E2ArtifactError("feature file hash mismatch")

    encoder = metadata.get("encoder")
    if not isinstance(encoder, dict):
        raise E2ArtifactError("encoder metadata must be an object")
    encoder_name = _require_string(encoder, "name")
    if (
        expected_encoder is not None
        and encoder_name != expected_encoder
    ):
        raise E2ArtifactError(
            "cache uses an unexpected encoder"
        )
    _require_string(encoder, "checkpoint_id")
    _require_string(encoder, "feature_layer")
    _require_string(encoder, "preprocess")
    _validate_sha256(
        encoder.get("model_state_sha256"),
        "model_state_sha256",
    )
    checkpoint_hash = encoder.get("checkpoint_file_sha256")
    if checkpoint_hash is not None:
        _validate_sha256(
            checkpoint_hash,
            "checkpoint_file_sha256",
        )
    _validate_sha256(
        encoder.get("preprocess_sha256"),
        "preprocess_sha256",
    )

    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            if set(payload.files) != {"H", "y", "sample_ids"}:
                raise E2ArtifactError(
                    "feature cache must contain H, y, and sample_ids"
                )
            features = np.asarray(payload["H"])
            labels = np.asarray(payload["y"])
            sample_ids = np.asarray(payload["sample_ids"])
    except (OSError, ValueError) as error:
        raise E2ArtifactError(
            f"cannot read feature cache: {error}"
        ) from error

    if features.ndim != 2 or features.dtype != np.float32:
        raise E2ArtifactError(
            "H must be a two-dimensional float32 array"
        )
    if labels.ndim != 1 or labels.dtype != np.int64:
        raise E2ArtifactError(
            "y must be a one-dimensional int64 array"
        )
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise E2ArtifactError(
            "sample_ids must be a one-dimensional fixed-width string array"
        )
    if not np.isfinite(features).all():
        raise E2ArtifactError("H contains non-finite values")
    expected_ids = np.asarray(
        [str(row["sample_id"]) for row in samples]
    )
    expected_labels = np.asarray(
        [int(row["class_id"]) for row in samples],
        dtype=np.int64,
    )
    if not np.array_equal(sample_ids.astype(str), expected_ids):
        raise E2ArtifactError(
            "cache sample IDs or sample ordering differ from manifest"
        )
    if not np.array_equal(labels, expected_labels):
        raise E2ArtifactError(
            "cache labels differ from manifest"
        )
    if features.shape[0] != len(samples):
        raise E2ArtifactError(
            "cache feature count differs from manifest"
        )

    expected_metadata = {
        "sample_count": len(samples),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "sample_id_dtype": str(sample_ids.dtype),
        "feature_postprocessing": "none_raw_pooled",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise E2ArtifactError(
                f"cache metadata field {key} is incorrect"
            )
    return {
        "status": "valid",
        "dataset": bundle_report["dataset"],
        "manifest_id": bundle_report["manifest_id"],
        "encoder": encoder_name,
        "feature_shape": list(features.shape),
        "feature_file_sha256": metadata["feature_file_sha256"],
    }
