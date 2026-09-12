"""Core contracts and numerical helpers for the independent E2b run."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CONFIG_SCHEMA = "target-conditioned-e2b-config-v1"
MANIFEST_SCHEMA = "target-conditioned-e2b-domainnet-manifest-v1"
CACHE_SCHEMA = "target-conditioned-e2b-feature-cache-v1"
CANDIDATE_SCHEMA = "target-conditioned-e2b-candidates-v1"
SCREEN_SCHEMA = "target-conditioned-e2b-screen-v1"
VALIDATION_SCHEMA = "target-conditioned-e2b-validation-v1"
TEST_AUDIT_SCHEMA = "target-conditioned-e2b-test-audit-v1"

SAMPLE_FIELDS = (
    "sample_index",
    "sample_id",
    "domain",
    "official_split",
    "role",
    "class_name",
    "class_id",
    "archive_filename",
    "archive_member",
    "relative_path",
    "content_sha256",
    "byte_size",
)


class E2BArtifactError(ValueError):
    """Raised when an E2b artifact violates its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_fields(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def stable_seed(*values: object) -> int:
    return int(hash_fields(*values)[:16], 16) % (2**32)


def hash_ids(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise E2BArtifactError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise E2BArtifactError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def read_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise E2BArtifactError(
                    f"{path.name} columns differ from the frozen schema"
                )
            return [dict(row) for row in reader]
    except OSError as error:
        raise E2BArtifactError(f"cannot read {path}: {error}") from error


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise E2BArtifactError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise E2BArtifactError(f"{name} must be a nonempty list")
    return value


def load_config(path: Path) -> dict[str, object]:
    config = read_json(path)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise E2BArtifactError(f"config schema must be {CONFIG_SCHEMA}")
    if config.get("status") != (
        "frozen-before-domainnet-feature-extraction-and-outcome-evaluation"
    ):
        raise E2BArtifactError("E2b config is not frozen")

    dataset = _mapping(config.get("dataset"), "dataset")
    domains = [str(value) for value in _list(dataset.get("domains"), "domains")]
    if domains != sorted(set(domains)) or len(domains) != 6:
        raise E2BArtifactError("E2b requires six sorted unique domains")
    subset = _mapping(dataset.get("class_subset"), "class_subset")
    if int(subset.get("count", 0)) < 2:
        raise E2BArtifactError("class subset must contain at least two classes")

    sampling = _mapping(config.get("sampling"), "sampling")
    for key in (
        "target_selection_per_domain",
        "target_validation_per_domain",
        "anchor_pool_per_domain",
        "target_test_per_domain",
    ):
        if int(sampling.get(key, 0)) < 1:
            raise E2BArtifactError(f"sampling.{key} must be positive")
    if int(sampling["anchor_pool_per_domain"]) % 3:
        raise E2BArtifactError("anchor pool must divide evenly into three strata")

    construction = _mapping(
        config.get("candidate_construction"), "candidate_construction"
    )
    strata = [str(value) for value in _list(construction.get("strata"), "strata")]
    if len(strata) != int(construction.get("blocks_per_source_domain", -1)):
        raise E2BArtifactError("strata and source block counts differ")
    block_size = int(construction.get("candidate_samples_per_stratum", 0))
    if block_size < 1 or block_size > int(sampling["anchor_pool_per_domain"]) // 3:
        raise E2BArtifactError("candidate block size is incompatible with strata")

    combinations = _mapping(config.get("combinations"), "combinations")
    budget = int(combinations.get("budget_blocks", 0))
    candidate_count = int(construction["candidate_blocks_per_target"])
    expected = math.comb(candidate_count, budget)
    if expected != int(combinations.get("expected_combinations_per_target", -1)):
        raise E2BArtifactError("frozen combination count is incorrect")
    if int(combinations.get("fixed_training_sample_count", -1)) != budget * block_size:
        raise E2BArtifactError("fixed training sample count is incorrect")

    screen = _mapping(config.get("screen"), "screen")
    methods = _mapping(screen.get("methods"), "screen.methods")
    if "target_a" not in methods or "random" not in methods:
        raise E2BArtifactError("screen methods omit required controls")
    sizes = [int(value) for value in _list(screen.get("shortlist_sizes"), "shortlist")]
    if sizes != sorted(set(sizes)) or sizes[-1] > expected:
        raise E2BArtifactError("shortlist sizes are invalid")
    if int(screen.get("primary_shortlist_size", -1)) not in sizes:
        raise E2BArtifactError("primary shortlist size is not registered")
    if (expected - int(screen["primary_shortlist_size"])) / expected < 0.5:
        raise E2BArtifactError("primary shortlist does not remove at least half")

    validation = _mapping(config.get("validation"), "validation")
    if float(validation.get("ridge_regularization", 0.0)) <= 0.0:
        raise E2BArtifactError("ridge regularization must be positive")
    audit = _mapping(config.get("test_audit"), "test_audit")
    if audit.get("evaluate_all_combinations") is not True:
        raise E2BArtifactError("test audit must exhaust all combinations")
    if int(audit.get("top_q", 0)) < 1:
        raise E2BArtifactError("test top-q must be positive")
    return config


def selected_class_names(
    class_names: Sequence[str],
    count: int,
    seed: str,
) -> list[str]:
    names = sorted(set(str(value) for value in class_names))
    if len(names) < count:
        raise E2BArtifactError("class vocabulary is smaller than the frozen subset")
    ranked = sorted(names, key=lambda name: (hash_fields(seed, name), name))
    return sorted(ranked[:count])


def sample_order_key(seed: str, role: str, domain: str, path: str) -> tuple[str, str]:
    return hash_fields(seed, role, domain, path), path


def read_manifest_samples(bundle_dir: Path) -> list[dict[str, object]]:
    rows = read_csv(bundle_dir / "samples.csv", SAMPLE_FIELDS)
    parsed: list[dict[str, object]] = []
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
            raise E2BArtifactError("manifest has a non-integer field") from error
    return parsed


def validate_manifest(bundle_dir: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    manifest = read_json(bundle_dir / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise E2BArtifactError(f"manifest schema must be {MANIFEST_SCHEMA}")
    if manifest.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("manifest uses a different frozen config")
    samples_path = bundle_dir / "samples.csv"
    if manifest.get("samples_file_sha256") != sha256_file(samples_path):
        raise E2BArtifactError("manifest samples hash mismatch")
    samples = read_manifest_samples(bundle_dir)
    if [row["sample_index"] for row in samples] != list(range(len(samples))):
        raise E2BArtifactError("sample indices are not contiguous")
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise E2BArtifactError("sample IDs are not unique")
    if len({str(row["content_sha256"]) for row in samples}) != len(samples):
        raise E2BArtifactError("selected sample content is not unique")

    sampling = _mapping(config["sampling"], "sampling")
    expected_roles = {
        "target_selection": int(sampling["target_selection_per_domain"]),
        "target_validation": int(sampling["target_validation_per_domain"]),
        "anchor_pool": int(sampling["anchor_pool_per_domain"]),
        "target_test": int(sampling["target_test_per_domain"]),
    }
    domains = [str(value) for value in _mapping(config["dataset"], "dataset")["domains"]]
    counts = Counter((str(row["domain"]), str(row["role"])) for row in samples)
    for domain in domains:
        for role, count in expected_roles.items():
            if counts[(domain, role)] != count:
                raise E2BArtifactError(f"quota mismatch for {domain}/{role}")
    if manifest.get("sample_count") != len(samples):
        raise E2BArtifactError("manifest sample count mismatch")
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_id") != canonical_json_sha256(core):
        raise E2BArtifactError("manifest identifier mismatch")
    return {
        "status": "valid",
        "manifest_id": manifest["manifest_id"],
        "sample_count": len(samples),
        "domains": domains,
        "class_count": len(manifest["class_to_id"]),
    }


def validate_feature_cache_unlabeled(
    feature_path: Path,
    metadata_path: Path,
    manifest_dir: Path,
    config_path: Path,
    expected_encoder: str | None = None,
) -> dict[str, object]:
    report = validate_manifest(manifest_dir, config_path)
    metadata = read_json(metadata_path)
    if metadata.get("schema_version") != CACHE_SCHEMA:
        raise E2BArtifactError(f"cache schema must be {CACHE_SCHEMA}")
    if metadata.get("manifest_id") != report["manifest_id"]:
        raise E2BArtifactError("feature cache manifest mismatch")
    if metadata.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("feature cache config mismatch")
    if metadata.get("feature_file_sha256") != sha256_file(feature_path):
        raise E2BArtifactError("feature cache file hash mismatch")
    encoder = _mapping(metadata.get("encoder"), "encoder")
    encoder_name = str(encoder.get("name", ""))
    if not encoder_name:
        raise E2BArtifactError("feature cache encoder is empty")
    if expected_encoder is not None and encoder_name != expected_encoder:
        raise E2BArtifactError("feature cache uses an unexpected encoder")
    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            if set(payload.files) != {"H", "y", "sample_ids"}:
                raise E2BArtifactError("feature cache keys are incorrect")
            features = np.asarray(payload["H"])
            sample_ids = np.asarray(payload["sample_ids"])
    except (OSError, ValueError) as error:
        raise E2BArtifactError(f"cannot load feature cache: {error}") from error
    samples = read_manifest_samples(manifest_dir)
    expected_ids = np.asarray([str(row["sample_id"]) for row in samples])
    if features.ndim != 2 or features.dtype != np.float32:
        raise E2BArtifactError("feature H must be a float32 matrix")
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise E2BArtifactError("feature sample IDs have the wrong type or shape")
    if not np.array_equal(sample_ids.astype(str), expected_ids):
        raise E2BArtifactError("feature sample IDs differ from the manifest")
    if features.shape[0] != len(samples) or not np.isfinite(features).all():
        raise E2BArtifactError("feature matrix differs from the manifest or is nonfinite")
    if metadata.get("feature_shape") != list(features.shape):
        raise E2BArtifactError("feature metadata shape is incorrect")
    return {
        "status": "valid",
        "manifest_id": report["manifest_id"],
        "encoder": encoder_name,
        "feature_shape": list(features.shape),
        "feature_file_sha256": metadata["feature_file_sha256"],
    }


def validate_feature_cache(
    feature_path: Path,
    metadata_path: Path,
    manifest_dir: Path,
    config_path: Path,
    expected_encoder: str | None = None,
) -> dict[str, object]:
    report = validate_feature_cache_unlabeled(
        feature_path,
        metadata_path,
        manifest_dir,
        config_path,
        expected_encoder,
    )
    try:
        with np.load(feature_path, allow_pickle=False) as payload:
            labels = np.asarray(payload["y"])
    except (OSError, ValueError) as error:
        raise E2BArtifactError(f"cannot load feature labels: {error}") from error
    samples = read_manifest_samples(manifest_dir)
    expected_labels = np.asarray(
        [int(row["class_id"]) for row in samples], dtype=np.int64
    )
    if labels.dtype != np.int64 or labels.shape != (len(samples),):
        raise E2BArtifactError("feature labels have the wrong type or shape")
    if not np.array_equal(labels, expected_labels):
        raise E2BArtifactError("feature labels differ from the manifest")
    return report


def validate_candidates(
    candidate_path: Path,
    manifest_dir: Path,
    config_path: Path,
    anchor_feature_path: Path,
    anchor_metadata_path: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    manifest_report = validate_manifest(manifest_dir, config_path)
    anchor_name = str(config["candidate_construction"]["anchor_encoder"])
    anchor_report = validate_feature_cache_unlabeled(
        anchor_feature_path,
        anchor_metadata_path,
        manifest_dir,
        config_path,
        anchor_name,
    )
    artifact = read_json(candidate_path)
    if artifact.get("schema_version") != CANDIDATE_SCHEMA:
        raise E2BArtifactError(f"candidate schema must be {CANDIDATE_SCHEMA}")
    if artifact.get("manifest_id") != manifest_report["manifest_id"]:
        raise E2BArtifactError("candidate manifest mismatch")
    if artifact.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("candidate config mismatch")
    if artifact.get("anchor_feature_file_sha256") != anchor_report[
        "feature_file_sha256"
    ]:
        raise E2BArtifactError("candidate anchor feature mismatch")
    if artifact.get("uses_labels_or_target_data") is not False:
        raise E2BArtifactError("candidate artifact does not forbid labels")
    core = {key: value for key, value in artifact.items() if key != "candidate_set_id"}
    if artifact.get("candidate_set_id") != canonical_json_sha256(core):
        raise E2BArtifactError("candidate artifact identifier mismatch")

    construction = _mapping(config["candidate_construction"], "construction")
    domains = [str(value) for value in _mapping(config["dataset"], "dataset")["domains"]]
    strata = [str(value) for value in construction["strata"]]
    expected_size = int(construction["candidate_samples_per_stratum"])
    samples = read_manifest_samples(manifest_dir)
    role_by_id = {str(row["sample_id"]): str(row["role"]) for row in samples}
    domain_by_id = {str(row["sample_id"]): str(row["domain"]) for row in samples}
    candidates = _list(artifact.get("candidates"), "candidates")
    expected_ids = {
        f"{domain}__clip_pc1_{stratum}"
        for domain in domains
        for stratum in strata
    }
    actual_ids: set[str] = set()
    all_samples: set[str] = set()
    for value in candidates:
        row = _mapping(value, "candidate")
        candidate_id = str(row.get("candidate_id", ""))
        domain = str(row.get("domain", ""))
        stratum = str(row.get("stratum", ""))
        ids = [str(item) for item in _list(row.get("sample_ids"), "sample_ids")]
        if candidate_id != f"{domain}__clip_pc1_{stratum}":
            raise E2BArtifactError("candidate ID is not canonical")
        if len(ids) != expected_size or len(set(ids)) != expected_size:
            raise E2BArtifactError("candidate block has the wrong size")
        if row.get("sample_ids_sha256") != hash_ids(ids):
            raise E2BArtifactError("candidate sample hash mismatch")
        if any(role_by_id.get(item) != "anchor_pool" for item in ids):
            raise E2BArtifactError("candidate contains a non-anchor sample")
        if any(domain_by_id.get(item) != domain for item in ids):
            raise E2BArtifactError("candidate crosses source domains")
        if all_samples.intersection(ids):
            raise E2BArtifactError("candidate blocks overlap")
        all_samples.update(ids)
        actual_ids.add(candidate_id)
    if actual_ids != expected_ids or len(candidates) != len(expected_ids):
        raise E2BArtifactError("candidate inventory is incomplete")
    return {
        "status": "valid",
        "candidate_set_id": artifact["candidate_set_id"],
        "candidate_count": len(candidates),
        "candidate_sample_count": len(all_samples),
    }


def normalize_rows(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise E2BArtifactError("features must be a finite nonempty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(epsilon))


def jl_project(
    features: np.ndarray,
    output_dimension: int,
    seed: int,
) -> np.ndarray:
    matrix = normalize_rows(features)
    if output_dimension < 1:
        raise E2BArtifactError("projection dimension must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.integers(
        0,
        2,
        size=(matrix.shape[1], output_dimension),
        dtype=np.int8,
    )
    projection = (2.0 * signs.astype(np.float32) - 1.0) / math.sqrt(
        output_dimension
    )
    return normalize_rows(matrix @ projection)


def all_combinations(names: Sequence[str], budget: int) -> list[tuple[str, ...]]:
    values = sorted(set(str(value) for value in names))
    return list(itertools.combinations(values, budget))


def combination_name(value: Sequence[str]) -> str:
    names = tuple(sorted(str(item) for item in value))
    if not names or len(names) != len(set(names)):
        raise E2BArtifactError("combination is empty or contains duplicates")
    return "|".join(names)


def parse_combination(value: str) -> tuple[str, ...]:
    return tuple(combination_name(value.split("|")).split("|"))


def effective_rank_from_scatter(scatter: np.ndarray, sample_count: int) -> float:
    matrix = np.asarray(scatter, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))
    if singular.size == 0 or singular.max(initial=0.0) == 0.0:
        return 0.0
    tolerance = singular.max() * math.sqrt(
        np.finfo(np.float64).eps
        * max(sample_count, matrix.shape[0], 1)
        * 8.0
    )
    retained = singular[singular > tolerance]
    if not len(retained):
        return 0.0
    probabilities = retained / retained.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def ridge_metrics(
    block_features: Mapping[str, np.ndarray],
    block_labels: Mapping[str, np.ndarray],
    combination: Sequence[str],
    target_features: np.ndarray,
    target_labels: np.ndarray,
    class_count: int,
    regularization: float,
    ece_bins: int,
) -> dict[str, float]:
    dimension = next(iter(block_features.values())).shape[1]
    gram = regularization * np.eye(dimension, dtype=np.float64)
    rhs = np.zeros((dimension, class_count), dtype=np.float64)
    for name in combination:
        features = np.asarray(block_features[name], dtype=np.float64)
        labels = np.asarray(block_labels[name], dtype=np.int64)
        gram += features.T @ features
        rhs += features.T @ np.eye(class_count, dtype=np.float64)[labels]
    weights = np.linalg.solve((gram + gram.T) / 2.0, rhs)
    scores = np.asarray(target_features, dtype=np.float64) @ weights
    truth = np.asarray(target_labels, dtype=np.int64)
    one_hot = np.eye(class_count, dtype=np.float64)[truth]
    squared_loss = float(np.mean(np.sum((scores - one_hot) ** 2, axis=1)))
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    nll = float(
        -np.mean(
            np.log(
                np.maximum(
                    probabilities[np.arange(len(truth)), truth],
                    1e-15,
                )
            )
        )
    )
    accuracy = float(np.mean(predicted == truth))
    f1 = []
    for label in range(class_count):
        true_positive = int(np.sum((predicted == label) & (truth == label)))
        false_positive = int(np.sum((predicted == label) & (truth != label)))
        false_negative = int(np.sum((predicted != label) & (truth == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    confidence = probabilities.max(axis=1)
    correct = (predicted == truth).astype(np.float64)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        mask = (confidence >= lower) & (
            confidence <= upper if bin_index + 1 == ece_bins else confidence < upper
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {
        "brier_score": brier,
        "squared_loss": squared_loss,
        "nll": nll,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)),
        "ece": ece,
    }
