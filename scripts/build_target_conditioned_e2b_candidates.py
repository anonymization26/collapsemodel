#!/usr/bin/env python3
"""Build equal-sized heterogeneous E2b blocks from label-free CLIP features."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2b import (  # noqa: E402
    CANDIDATE_SCHEMA,
    E2BArtifactError,
    canonical_json_sha256,
    hash_fields,
    hash_ids,
    load_config,
    normalize_rows,
    read_manifest_samples,
    sha256_file,
    validate_candidates,
    validate_feature_cache_unlabeled,
    validate_manifest,
    write_json_atomic,
)


def build_candidates(
    manifest_dir: Path,
    config_path: Path,
    anchor_feature_path: Path,
    anchor_metadata_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        validate_candidates(
            output_path,
            manifest_dir,
            config_path,
            anchor_feature_path,
            anchor_metadata_path,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    manifest_report = validate_manifest(manifest_dir, config_path)
    construction = config["candidate_construction"]
    anchor_name = str(construction["anchor_encoder"])
    anchor_report = validate_feature_cache_unlabeled(
        anchor_feature_path,
        anchor_metadata_path,
        manifest_dir,
        config_path,
        anchor_name,
    )
    samples = read_manifest_samples(manifest_dir)
    try:
        with np.load(anchor_feature_path, allow_pickle=False) as payload:
            features = np.asarray(payload["H"])
            sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    except (OSError, ValueError) as error:
        raise E2BArtifactError(f"cannot load anchor features: {error}") from error
    if len(features) != len(samples):
        raise E2BArtifactError("anchor features differ from the manifest")

    domains = [str(value) for value in config["dataset"]["domains"]]
    strata = [str(value) for value in construction["strata"]]
    block_size = int(construction["candidate_samples_per_stratum"])
    pick_seed = str(construction["candidate_pick_seed"])
    candidates = []
    diagnostics = []
    for domain in domains:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(samples)
                if row["domain"] == domain and row["role"] == "anchor_pool"
            ],
            dtype=np.int64,
        )
        domain_features = normalize_rows(features[indices]).astype(np.float64)
        centered = domain_features - domain_features.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
        axis = eigenvectors[:, -1]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0.0:
            axis = -axis
        projections = centered @ axis
        ids = sample_ids[indices]
        order = np.lexsort((ids, projections))
        groups = np.array_split(order, len(strata))
        if len({len(group) for group in groups}) != 1:
            raise E2BArtifactError("anchor strata are not equal")
        domain_candidate_means = []
        for stratum, group in zip(strata, groups):
            ranked = sorted(
                group.tolist(),
                key=lambda local_index: (
                    hash_fields(
                        pick_seed,
                        domain,
                        stratum,
                        ids[local_index],
                    ),
                    ids[local_index],
                ),
            )
            chosen = ranked[:block_size]
            chosen_ids = [str(ids[index]) for index in chosen]
            candidate_id = f"{domain}__clip_pc1_{stratum}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "domain": domain,
                    "stratum": stratum,
                    "sample_count": len(chosen_ids),
                    "sample_ids": chosen_ids,
                    "sample_ids_sha256": hash_ids(chosen_ids),
                    "projection_min": float(np.min(projections[group])),
                    "projection_max": float(np.max(projections[group])),
                    "projection_mean_selected": float(np.mean(projections[chosen])),
                }
            )
            domain_candidate_means.append(
                normalize_rows(domain_features[chosen]).mean(axis=0)
            )
        pairwise = []
        for left in range(len(domain_candidate_means)):
            for right in range(left + 1, len(domain_candidate_means)):
                pairwise.append(
                    float(
                        np.linalg.norm(
                            domain_candidate_means[left] - domain_candidate_means[right]
                        )
                    )
                )
        diagnostics.append(
            {
                "domain": domain,
                "anchor_count": len(indices),
                "leading_eigenvalue": float(eigenvalues[-1]),
                "leading_variance_fraction": float(
                    eigenvalues[-1] / max(float(eigenvalues.sum()), 1e-15)
                ),
                "mean_pairwise_candidate_centroid_distance": float(np.mean(pairwise)),
                "minimum_pairwise_candidate_centroid_distance": float(np.min(pairwise)),
            }
        )
    core: dict[str, object] = {
        "schema_version": CANDIDATE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_id": manifest_report["manifest_id"],
        "config_file_sha256": sha256_file(config_path),
        "anchor_encoder": anchor_name,
        "anchor_feature_file_sha256": anchor_report["feature_file_sha256"],
        "construction": construction,
        "uses_labels_or_target_data": False,
        "candidates": candidates,
        "heterogeneity_diagnostics": diagnostics,
    }
    artifact = {**core, "candidate_set_id": canonical_json_sha256(core)}
    write_json_atomic(output_path, artifact)
    validate_candidates(
        output_path,
        manifest_dir,
        config_path,
        anchor_feature_path,
        anchor_metadata_path,
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--anchor-features", type=Path, required=True)
    parser.add_argument("--anchor-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_candidates(
        args.manifest_dir,
        args.config,
        args.anchor_features,
        args.anchor_metadata,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": "candidates-frozen",
                "candidate_set_id": artifact["candidate_set_id"],
                "candidate_count": len(artifact["candidates"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
