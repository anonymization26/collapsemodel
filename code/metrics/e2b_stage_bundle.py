"""Role-separated E2b inputs, produced by an offline data-preparation step.

Preparation may read the full cache. Experimental workers receive only their
stage directory, whose manifest omits class names, paths, and forbidden labels.
This is a file-access contract, not an operating-system sandbox.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .target_conditioned_e2b import (
    CANDIDATE_SCHEMA, E2BArtifactError, canonical_json_sha256, hash_ids,
    load_config, read_json, read_manifest_samples, sha256_file,
    validate_feature_cache, validate_manifest, write_json_atomic,
)


SCHEMA = "target-conditioned-e2b-stage-bundle-v1"
TARGET_ROLE = {
    "screen": "target_selection",
    "validate": "target_validation",
    "test-audit": "target_test",
}
ROW_FIELDS = {"sample_id", "domain", "role"}
MANIFEST_FIELDS = {
    "schema_version", "stage", "encoder", "manifest_id", "config_file_sha256",
    "candidate_set_id", "source_feature_file_sha256", "source_metadata_sha256",
    "class_count", "feature_file_sha256", "feature_shape", "samples", "candidates",
    "candidate_policy", "preparation_has_full_data_access", "bundle_id",
}


def _candidates(path: Path, samples: list[dict], config: dict, manifest_id: str,
                config_hash: str) -> dict:
    artifact = read_json(path)
    core = {k: v for k, v in artifact.items() if k != "candidate_set_id"}
    if (artifact.get("schema_version") != CANDIDATE_SCHEMA
            or artifact.get("candidate_set_id") != canonical_json_sha256(core)
            or artifact.get("manifest_id") != manifest_id
            or artifact.get("config_file_sha256") != config_hash
            or artifact.get("uses_labels_or_target_data") is not False):
        raise E2BArtifactError("frozen candidate provenance mismatch")
    by_id = {str(row["sample_id"]): row for row in samples}
    expected = {
        f"{domain}__clip_pc1_{stratum}"
        for domain in config["dataset"]["domains"]
        for stratum in config["candidate_construction"]["strata"]
    }
    names, used = set(), set()
    count = int(config["candidate_construction"]["candidate_samples_per_stratum"])
    for block in artifact["candidates"]:
        ids = block["sample_ids"]
        name = block["candidate_id"]
        if (name in names or name not in expected or len(ids) != count
                or len(set(ids)) != count or used.intersection(ids)
                or block["sample_ids_sha256"] != hash_ids(ids)):
            raise E2BArtifactError("frozen candidate membership is invalid")
        for sid in ids:
            row = by_id.get(sid, {})
            if row.get("role") != "anchor_pool" or row.get("domain") != block["domain"]:
                raise E2BArtifactError("candidate uses an unexpected sample role")
        names.add(name)
        used.update(ids)
    if names != expected:
        raise E2BArtifactError("candidate coverage is incomplete")
    return artifact


def prepare_stage_bundles(manifest_dir: Path, config_path: Path,
                          candidate_path: Path, feature_path: Path,
                          metadata_path: Path, encoder: str,
                          output_dir: Path) -> dict:
    """Preserve frozen candidate IDs even when reconstructing on a new backend."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2BArtifactError("refusing to overwrite prepared stage bundles")
    config = load_config(config_path)
    report = validate_manifest(manifest_dir, config_path)
    feature_report = validate_feature_cache(
        feature_path, metadata_path, manifest_dir, config_path, encoder
    )
    samples = read_manifest_samples(manifest_dir)
    config_hash = sha256_file(config_path)
    artifact = _candidates(candidate_path, samples, config, report["manifest_id"], config_hash)
    blocks = [
        {key: block[key] for key in ("candidate_id", "domain", "sample_ids")}
        for block in artifact["candidates"]
    ]
    source_ids = {sid for block in blocks for sid in block["sample_ids"]}
    with np.load(feature_path, allow_pickle=False) as payload:
        features, labels = payload["H"], payload["y"]
    receipt = {}
    for stage, target_role in TARGET_ROLE.items():
        indices = [i for i, row in enumerate(samples)
                   if row["sample_id"] in source_ids or row["role"] == target_role]
        selected = [samples[i] for i in indices]
        directory = output_dir / stage
        directory.mkdir(parents=True)
        arrays = {
            "H": features[indices],
            "sample_ids": np.asarray([row["sample_id"] for row in selected], dtype="<U64"),
        }
        if stage != "screen":
            arrays["y"] = labels[indices]
        path = directory / "features.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        core = {
            "schema_version": SCHEMA, "stage": stage, "encoder": encoder,
            "manifest_id": report["manifest_id"], "config_file_sha256": config_hash,
            "candidate_set_id": artifact["candidate_set_id"],
            "source_feature_file_sha256": feature_report["feature_file_sha256"],
            "source_metadata_sha256": sha256_file(metadata_path),
            "class_count": report["class_count"],
            "feature_file_sha256": sha256_file(path),
            "feature_shape": list(arrays["H"].shape),
            "samples": [{key: row[key] for key in ROW_FIELDS} for row in selected],
            "candidates": blocks,
            "candidate_policy": "reuse-frozen-membership-without-recomputing-CLIP-strata",
            "preparation_has_full_data_access": True,
        }
        metadata = {**core, "bundle_id": canonical_json_sha256(core)}
        write_json_atomic(directory / "manifest.json", metadata)
        load_stage_bundle(directory, config_path, stage, encoder)
        receipt[stage] = metadata["bundle_id"]
    return receipt


def load_stage_bundle(directory: Path, config_path: Path, stage: str,
                      encoder: str) -> dict:
    """Read exactly one sanitized manifest and its role-restricted NPZ."""
    if stage not in TARGET_ROLE:
        raise E2BArtifactError("unknown experiment stage")
    config = load_config(config_path)
    metadata = read_json(directory / "manifest.json")
    core = {k: v for k, v in metadata.items() if k != "bundle_id"}
    if (set(metadata) != MANIFEST_FIELDS or metadata.get("schema_version") != SCHEMA
            or metadata.get("bundle_id") != canonical_json_sha256(core)
            or metadata.get("stage") != stage or metadata.get("encoder") != encoder
            or metadata.get("config_file_sha256") != sha256_file(config_path)):
        raise E2BArtifactError("stage bundle identity mismatch")
    if metadata["class_count"] != int(config["dataset"]["class_subset"]["count"]):
        raise E2BArtifactError("stage class count mismatch")
    samples = metadata["samples"]
    allowed = {"anchor_pool", TARGET_ROLE[stage]}
    if any(set(row) != ROW_FIELDS or row["role"] not in allowed for row in samples):
        raise E2BArtifactError("stage manifest exposes forbidden metadata or sample roles")
    ids = [row["sample_id"] for row in samples]
    if len(set(ids)) != len(ids):
        raise E2BArtifactError("stage sample IDs are not unique")
    domains = set(config["dataset"]["domains"])
    if any(row["domain"] not in domains for row in samples):
        raise E2BArtifactError("unexpected domain in stage bundle")
    quota_key = TARGET_ROLE[stage] + "_per_domain"
    quota = int(config["sampling"][quota_key])
    for domain in domains:
        if sum(row["domain"] == domain and row["role"] == TARGET_ROLE[stage]
               for row in samples) != quota:
            raise E2BArtifactError("stage target quota mismatch")
    by_id = {row["sample_id"]: row for row in samples}
    used, block_names = set(), set()
    expected_names = {
        f"{domain}__clip_pc1_{stratum}"
        for domain in domains for stratum in config["candidate_construction"]["strata"]
    }
    block_size = int(config["candidate_construction"]["candidate_samples_per_stratum"])
    for block in metadata["candidates"]:
        block_ids = block["sample_ids"]
        if (set(block) != {"candidate_id", "domain", "sample_ids"}
                or block["candidate_id"] not in expected_names
                or block["candidate_id"] in block_names
                or len(block_ids) != block_size or len(set(block_ids)) != block_size
                or used.intersection(block_ids)):
            raise E2BArtifactError("stage candidate metadata mismatch")
        for sid in block_ids:
            row = by_id.get(sid, {})
            if row.get("role") != "anchor_pool" or row.get("domain") != block["domain"]:
                raise E2BArtifactError("stage candidate sample role mismatch")
        used.update(block_ids)
        block_names.add(block["candidate_id"])
    if block_names != expected_names or used != {
        row["sample_id"] for row in samples if row["role"] == "anchor_pool"
    }:
        raise E2BArtifactError("stage source coverage mismatch")
    path = directory / "features.npz"
    if sha256_file(path) != metadata["feature_file_sha256"]:
        raise E2BArtifactError("stage feature hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        keys = {"H", "sample_ids"} | ({"y"} if stage != "screen" else set())
        if set(payload.files) != keys:
            raise E2BArtifactError("stage cache exposes forbidden or missing arrays")
        features, observed_ids = payload["H"], payload["sample_ids"]
        labels = payload["y"] if stage != "screen" else None
    if (features.dtype != np.float32 or features.ndim != 2
            or features.shape[0] != len(samples) or not np.isfinite(features).all()
            or list(features.shape) != metadata["feature_shape"]
            or not np.array_equal(observed_ids, np.asarray(ids))):
        raise E2BArtifactError("stage features or sample order are invalid")
    if labels is not None and (labels.dtype != np.int64 or labels.shape != (len(ids),)
            or np.any(labels < 0) or np.any(labels >= int(metadata["class_count"]))):
        raise E2BArtifactError("stage labels are invalid")
    return {"metadata": metadata, "features": features, "labels": labels,
            "sample_ids": observed_ids, "samples": samples,
            "candidate_ids": {b["candidate_id"]: b["sample_ids"] for b in metadata["candidates"]},
            "candidate_domains": {b["candidate_id"]: b["domain"] for b in metadata["candidates"]}}
