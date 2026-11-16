import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2 import (  # noqa: E402
    CACHE_SCHEMA,
    E2ArtifactError,
    build_assignments,
    build_manifest_bundle,
    canonical_json_sha256,
    read_manifest_samples,
    sha256_file,
    validate_feature_cache,
    validate_manifest_bundle,
)


class TargetConditionedE2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset_a"
        self._make_dataset(self.dataset_root)
        self.source_artifact = self.root / "fixture-release.zip"
        self.source_artifact.write_bytes(b"auditable fixture release")
        self.receipt_path = self.root / "receipt.json"
        self.receipt = {
            "schema_version": "target-conditioned-e2-receipt-v1",
            "dataset": "fixture_domains",
            "dataset_version": "fixture-v1",
            "official_page_url": "https://example.org/fixture",
            "download_urls": [
                "https://example.org/fixture-release.zip"
            ],
            "source_artifacts": [
                {
                    "filename": self.source_artifact.name,
                    "sha256": sha256_file(self.source_artifact),
                    "byte_size": self.source_artifact.stat().st_size,
                }
            ],
            "expected_domains": ["domain_a", "domain_b"],
            "expected_class_count": 2,
            "expected_sample_count": 40,
            "usage_terms": {
                "summary": "Synthetic fixture for tests.",
                "url": "https://example.org/fixture-terms",
            },
            "citation": {
                "key": "fixture2026",
                "bibtex": "@misc{fixture2026, title={Fixture}}",
            },
        }
        self.receipt_path.write_text(
            json.dumps(self.receipt),
            encoding="utf-8",
        )
        self.bundle = self.root / "bundle"
        build_manifest_bundle(
            self.dataset_root,
            self.receipt_path,
            [self.source_artifact],
            self.bundle,
            verify_images=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for domain_index, domain in enumerate(
            ("domain_a", "domain_b")
        ):
            for class_index, class_name in enumerate(
                ("class_a", "class_b")
            ):
                class_root = root / domain / class_name
                class_root.mkdir(parents=True)
                for sample_index in range(10):
                    value = (
                        domain_index * 100
                        + class_index * 30
                        + sample_index
                    )
                    pixels = np.full(
                        (6, 6, 3),
                        value,
                        dtype=np.uint8,
                    )
                    Image.fromarray(pixels).save(
                        class_root / f"sample_{sample_index:02d}.png"
                    )

    def _cache_metadata(
        self,
        feature_path: Path,
        features: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
    ):
        manifest = json.loads(
            (self.bundle / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return {
            "schema_version": CACHE_SCHEMA,
            "dataset": manifest["dataset"],
            "manifest_id": manifest["manifest_id"],
            "manifest_file_sha256": sha256_file(
                self.bundle / "manifest.json"
            ),
            "samples_file_sha256": manifest[
                "samples_file_sha256"
            ],
            "assignments_file_sha256": manifest[
                "assignments_file_sha256"
            ],
            "duplicates_file_sha256": manifest[
                "duplicates_file_sha256"
            ],
            "feature_file_sha256": sha256_file(feature_path),
            "sample_count": len(features),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "label_dtype": str(labels.dtype),
            "sample_id_dtype": str(sample_ids.dtype),
            "feature_postprocessing": "none_raw_pooled",
            "encoder": {
                "name": "fixture_encoder",
                "checkpoint_id": "fixture-weights-v1",
                "checkpoint_file_sha256": None,
                "model_state_sha256": "1" * 64,
                "feature_layer": "fixture-pool",
                "preprocess": "fixture-transform",
                "preprocess_sha256": "2" * 64,
            },
        }

    def _write_cache(
        self,
        directory: Path,
        features: np.ndarray,
        sample_ids=None,
    ):
        directory.mkdir()
        samples = read_manifest_samples(self.bundle)
        labels = np.asarray(
            [row["class_id"] for row in samples],
            dtype=np.int64,
        )
        if sample_ids is None:
            sample_ids = np.asarray(
                [row["sample_id"] for row in samples],
                dtype="<U64",
            )
        feature_path = directory / "features.npz"
        np.savez(
            feature_path,
            H=features,
            y=labels,
            sample_ids=sample_ids,
        )
        metadata = self._cache_metadata(
            feature_path,
            features,
            labels,
            sample_ids,
        )
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        return feature_path, metadata_path

    def test_manifest_is_location_independent_and_has_no_paths(self):
        copied_root = self.root / "dataset_b"
        shutil.copytree(self.dataset_root, copied_root)
        copied_bundle = self.root / "bundle_b"
        copied_manifest = build_manifest_bundle(
            copied_root,
            self.receipt_path,
            [self.source_artifact],
            copied_bundle,
            verify_images=True,
        )
        original_manifest = json.loads(
            (self.bundle / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(original_manifest, copied_manifest)
        self.assertEqual(
            (self.bundle / "samples.csv").read_bytes(),
            (copied_bundle / "samples.csv").read_bytes(),
        )
        self.assertEqual(
            (self.bundle / "assignments.csv").read_bytes(),
            (copied_bundle / "assignments.csv").read_bytes(),
        )
        self.assertEqual(
            (self.bundle / "content_duplicates.csv").read_bytes(),
            (
                copied_bundle
                / "content_duplicates.csv"
            ).read_bytes(),
        )
        serialized = json.dumps(original_manifest)
        self.assertNotIn(str(self.root), serialized)
        report = validate_manifest_bundle(
            copied_bundle,
            dataset_root=copied_root,
            source_artifact_paths=[self.source_artifact],
            verify_images=True,
        )
        self.assertEqual(report["sample_count"], 40)
        self.assertEqual(report["assignment_count"], 80)
        self.assertEqual(
            report["content_duplicate_audit"][
                "duplicate_group_count"
            ],
            0,
        )

    def test_assignments_ignore_class_labels_and_keep_duplicates_together(self):
        content_hash = "a" * 64
        samples = [
            {
                "sample_id": "first",
                "domain": "domain_a",
                "class_name": "class_a",
                "class_id": 0,
                "content_sha256": content_hash,
            },
            {
                "sample_id": "second",
                "domain": "domain_a",
                "class_name": "renamed_class",
                "class_id": 99,
                "content_sha256": content_hash,
            },
        ]
        target_rows = build_assignments(
            samples,
            "fixture",
            ["domain_a"],
            "seed",
            4,
        )
        self.assertEqual(target_rows[0]["role"], target_rows[1]["role"])
        source_rows = build_assignments(
            samples,
            "fixture",
            ["domain_b"],
            "seed",
            4,
        )
        self.assertEqual(
            source_rows[0]["candidate_id"],
            source_rows[1]["candidate_id"],
        )

    def test_source_copy_of_target_content_is_excluded(self):
        content_hash = "b" * 64
        samples = [
            {
                "sample_id": "target-copy",
                "domain": "domain_a",
                "class_name": "class_a",
                "class_id": 0,
                "content_sha256": content_hash,
            },
            {
                "sample_id": "source-copy",
                "domain": "domain_b",
                "class_name": "renamed_class",
                "class_id": 99,
                "content_sha256": content_hash,
            },
        ]
        rows = build_assignments(
            samples,
            "fixture",
            ["domain_a"],
            "seed",
            4,
        )
        self.assertTrue(rows[0]["role"].startswith("target_"))
        self.assertEqual(
            rows[1]["role"],
            "excluded_target_duplicate",
        )
        self.assertEqual(rows[1]["candidate_id"], "")

    def test_changed_dataset_or_source_artifact_is_rejected(self):
        changed_image = (
            self.dataset_root
            / "domain_a"
            / "class_a"
            / "sample_00.png"
        )
        Image.fromarray(
            np.full((6, 6, 3), 255, dtype=np.uint8)
        ).save(changed_image)
        with self.assertRaisesRegex(
            E2ArtifactError,
            "dataset tree differs",
        ):
            validate_manifest_bundle(
                self.bundle,
                dataset_root=self.dataset_root,
            )

        self.source_artifact.write_bytes(b"tampered release")
        with self.assertRaisesRegex(
            E2ArtifactError,
            "size mismatch|hash mismatch",
        ):
            validate_manifest_bundle(
                self.bundle,
                source_artifact_paths=[self.source_artifact],
            )

    def test_rehashed_assignment_tampering_is_still_rejected(self):
        assignment_path = self.bundle / "assignments.csv"
        with assignment_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as stream:
            rows = list(csv.DictReader(stream))
        target_row = next(
            row
            for row in rows
            if row["role"] == "target_selection"
        )
        target_row["role"] = "target_test"
        with assignment_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        manifest_path = self.bundle / "manifest.json"
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        manifest["assignments_file_sha256"] = sha256_file(
            assignment_path
        )
        manifest["assignment_counts"]["by_target_and_role"] = {
            target: {
                role: sum(
                    row["target_domain"] == target
                    and row["role"] == role
                    for row in rows
                )
                for role in (
                    "source_candidate",
                    "target_calibration",
                    "target_selection",
                    "target_test",
                )
                if any(
                    row["target_domain"] == target
                    and row["role"] == role
                    for row in rows
                )
            }
            for target in ("domain_a", "domain_b")
        }
        core = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_id"
        }
        manifest["manifest_id"] = canonical_json_sha256(core)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            E2ArtifactError,
            "label-free protocol",
        ):
            validate_manifest_bundle(self.bundle)

    def test_feature_cache_validation_and_tamper_detection(self):
        samples = read_manifest_samples(self.bundle)
        features = np.arange(
            len(samples) * 3,
            dtype=np.float32,
        ).reshape(len(samples), 3)
        feature_path, metadata_path = self._write_cache(
            self.root / "valid_cache",
            features,
        )
        report = validate_feature_cache(
            feature_path,
            metadata_path,
            self.bundle,
            expected_encoder="fixture_encoder",
        )
        self.assertEqual(report["feature_shape"], [40, 3])

        reversed_ids = np.asarray(
            [row["sample_id"] for row in reversed(samples)],
            dtype="<U64",
        )
        bad_id_path, bad_id_metadata = self._write_cache(
            self.root / "bad_ids",
            features,
            sample_ids=reversed_ids,
        )
        with self.assertRaisesRegex(
            E2ArtifactError,
            "ordering",
        ):
            validate_feature_cache(
                bad_id_path,
                bad_id_metadata,
                self.bundle,
            )

        nonfinite = features.copy()
        nonfinite[0, 0] = np.nan
        bad_value_path, bad_value_metadata = self._write_cache(
            self.root / "bad_values",
            nonfinite,
        )
        with self.assertRaisesRegex(
            E2ArtifactError,
            "non-finite",
        ):
            validate_feature_cache(
                bad_value_path,
                bad_value_metadata,
                self.bundle,
            )

    def test_absolute_cache_path_is_rejected(self):
        samples = read_manifest_samples(self.bundle)
        features = np.ones((len(samples), 2), dtype=np.float32)
        feature_path, metadata_path = self._write_cache(
            self.root / "path_cache",
            features,
        )
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        metadata["dataset_root"] = "/private/dataset"
        metadata_path.write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            E2ArtifactError,
            "forbidden path",
        ):
            validate_feature_cache(
                feature_path,
                metadata_path,
                self.bundle,
            )


if __name__ == "__main__":
    unittest.main()
