import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.target_conditioned_e2b import (  # noqa: E402
    CACHE_SCHEMA,
    MANIFEST_SCHEMA,
    SAMPLE_FIELDS,
    E2BArtifactError,
    canonical_json_sha256,
    hash_fields,
    jl_project,
    load_config,
    sha256_file,
    validate_feature_cache,
    validate_feature_cache_unlabeled,
    write_csv_atomic,
    write_json_atomic,
)
from build_target_conditioned_e2b_candidates import build_candidates  # noqa: E402
from materialize_target_conditioned_e2b_parquet import (  # noqa: E402
    _merge_ranges,
    _runs,
    _source_url,
    _split_ranges,
)
from run_target_conditioned_e2b_shortlist import (  # noqa: E402
    AUDIT_FIELDS,
    run_screen,
    run_test_audit,
    run_validation,
)
from summarize_target_conditioned_e2b import summarize  # noqa: E402


class TargetConditionedE2BTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        original = json.loads(
            (
                ROOT
                / "code/configs/target_conditioned_e2b/domainnet_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.config = copy.deepcopy(original)
        self.config["dataset"]["expected_class_count"] = 2
        self.config["dataset"]["class_subset"]["count"] = 2
        self.config["sampling"].update(
            {
                "target_selection_per_domain": 2,
                "target_validation_per_domain": 1,
                "anchor_pool_per_domain": 6,
                "target_test_per_domain": 2,
            }
        )
        self.config["candidate_construction"]["candidate_samples_per_stratum"] = 1
        self.config["combinations"]["samples_per_block"] = 1
        self.config["combinations"]["fixed_training_sample_count"] = 3
        self.config["features"]["projection"]["output_dimension"] = 4
        self.config["screen"]["subspace_rank"] = 2
        self.config["screen"]["random_repeats"] = 1
        self.config_path = self.root / "config.json"
        write_json_atomic(self.config_path, self.config)
        self.manifest_dir = self.root / "manifest"
        self._write_manifest()
        self.anchor_dir = self.root / "anchor"
        self.eval_dir = self.root / "eval"
        self._write_cache(self.anchor_dir, "clip_b32", invalid_labels=True)
        self._write_cache(self.eval_dir, "resnet50", invalid_labels=False)
        self.candidate_path = self.root / "candidates.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_manifest(self):
        domains = self.config["dataset"]["domains"]
        role_counts = {
            "target_selection": 2,
            "target_validation": 1,
            "anchor_pool": 6,
            "target_test": 2,
        }
        rows = []
        for domain_index, domain in enumerate(domains):
            for role, count in role_counts.items():
                split = "test" if role == "target_test" else "train"
                for offset in range(count):
                    token = f"{domain}-{role}-{offset}"
                    class_id = (domain_index + offset) % 2
                    rows.append(
                        {
                            "sample_index": len(rows),
                            "sample_id": hash_fields("sample", token),
                            "domain": domain,
                            "official_split": split,
                            "role": role,
                            "class_name": f"class_{class_id}",
                            "class_id": class_id,
                            "archive_filename": f"{domain}.zip",
                            "archive_member": f"{domain}/class_{class_id}/{token}.png",
                            "relative_path": f"{domain}/{split}/class_{class_id}/{token}.png",
                            "content_sha256": hash_fields("content", token),
                            "byte_size": 10,
                        }
                    )
        self.manifest_dir.mkdir()
        samples_path = self.manifest_dir / "samples.csv"
        write_csv_atomic(samples_path, SAMPLE_FIELDS, rows)
        core = {
            "schema_version": MANIFEST_SCHEMA,
            "dataset": "domainnet",
            "dataset_version": "fixture",
            "config_file_sha256": sha256_file(self.config_path),
            "source_artifacts": [],
            "selected_classes": ["class_0", "class_1"],
            "selected_classes_sha256": canonical_json_sha256(
                ["class_0", "class_1"]
            ),
            "class_to_id": {"class_0": 0, "class_1": 1},
            "sample_count": len(rows),
            "samples_file_sha256": sha256_file(samples_path),
            "role_counts": {
                domain: dict(role_counts) for domain in domains
            },
            "rejected_before_freeze": {},
            "assignment_access": {
                "uses_class_names_only_for_frozen_class_subset": True,
                "uses_labels_or_model_outputs": False,
                "official_test_is_a_distinct_source_split": True,
            },
        }
        write_json_atomic(
            self.manifest_dir / "manifest.json",
            {**core, "manifest_id": canonical_json_sha256(core)},
        )

    def _write_cache(self, directory: Path, encoder: str, invalid_labels: bool):
        directory.mkdir()
        with (self.manifest_dir / "samples.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        rng = np.random.default_rng(42 if encoder == "clip_b32" else 43)
        features = rng.normal(size=(len(rows), 8)).astype(np.float32)
        labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
        if invalid_labels:
            labels[:] = 99
        sample_ids = np.asarray([row["sample_id"] for row in rows], dtype="<U64")
        feature_path = directory / "features.npz"
        np.savez(feature_path, H=features, y=labels, sample_ids=sample_ids)
        metadata = {
            "schema_version": CACHE_SCHEMA,
            "manifest_id": json.loads(
                (self.manifest_dir / "manifest.json").read_text(encoding="utf-8")
            )["manifest_id"],
            "config_file_sha256": sha256_file(self.config_path),
            "feature_file_sha256": sha256_file(feature_path),
            "feature_shape": list(features.shape),
            "encoder": {"name": encoder},
        }
        write_json_atomic(directory / "metadata.json", metadata)

    def test_frozen_config_and_projection_are_deterministic(self):
        config = load_config(
            ROOT / "code/configs/target_conditioned_e2b/domainnet_v1.json"
        )
        self.assertEqual(config["combinations"]["expected_combinations_per_target"], 455)
        matrix = np.arange(40, dtype=np.float32).reshape(5, 8) + 1
        first = jl_project(matrix, 4, 7)
        second = jl_project(matrix, 4, 7)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.allclose(np.linalg.norm(first, axis=1), 1.0))

    def test_server_wrapper_explicitly_loads_ascend_environment(self):
        wrapper = (ROOT / "scripts/run_target_conditioned_e2b_server.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "source \"$ASCEND_ENV\"",
            wrapper,
        )
        self.assertLess(
            wrapper.index("source \"$ASCEND_ENV\""),
            wrapper.index('WORK_ROOT="${E2B_WORK_ROOT'),
        )

    def test_parquet_source_is_revision_and_content_pinned(self):
        source = json.loads(
            (
                ROOT
                / "code/configs/target_conditioned_e2b/domainnet_hf_parquet_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(len(source["files"]), 4)
        for item in source["files"]:
            self.assertGreater(item["byte_size"], 0)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn(
            source["revision"],
            _source_url(
                "https://example.invalid",
                source["repository"],
                source["revision"],
                source["files"][0]["path"],
            ),
        )

    def test_sparse_parquet_range_planning_is_deterministic(self):
        self.assertEqual(_runs([7, 4, 5, 10, 4]), [(4, 5), (7, 7), (10, 10)])
        self.assertEqual(
            _merge_ranges([(20, 29), (0, 9), (10, 19), (40, 49)]),
            [(0, 29), (40, 49)],
        )
        self.assertEqual(
            _split_ranges([(0, 9), (20, 24)], 4),
            [(0, 3), (4, 7), (8, 9), (20, 23), (24, 24)],
        )

    def test_unlabeled_validation_does_not_read_label_payload(self):
        report = validate_feature_cache_unlabeled(
            self.anchor_dir / "features.npz",
            self.anchor_dir / "metadata.json",
            self.manifest_dir,
            self.config_path,
            "clip_b32",
        )
        self.assertEqual(report["encoder"], "clip_b32")
        with self.assertRaises(E2BArtifactError):
            validate_feature_cache(
                self.anchor_dir / "features.npz",
                self.anchor_dir / "metadata.json",
                self.manifest_dir,
                self.config_path,
                "clip_b32",
            )

    def test_end_to_end_shortlist_uses_set_recall(self):
        build_candidates(
            self.manifest_dir,
            self.config_path,
            self.anchor_dir / "features.npz",
            self.anchor_dir / "metadata.json",
            self.candidate_path,
        )
        screen_dir = self.root / "screen"
        validation_dir = self.root / "validation"
        audit_dir = self.root / "audit"
        common = (
            self.manifest_dir,
            self.config_path,
            self.candidate_path,
            self.anchor_dir / "features.npz",
            self.anchor_dir / "metadata.json",
            self.eval_dir / "features.npz",
            self.eval_dir / "metadata.json",
            "resnet50",
        )
        screen = run_screen(*common, screen_dir)
        self.assertFalse(screen["access"]["source_labels"])
        validation = run_validation(*common, screen_dir, validation_dir)
        self.assertTrue(validation["selection_frozen_before_target_test"])
        for task in validation["tasks"]:
            self.assertFalse(task["cross_method_result_cache"])
            self.assertEqual(task["physical_primary_combination_evaluations"], 227)
            self.assertTrue(
                all(
                    row["physical_combination_evaluations"] == 227
                    for row in task["evaluations_by_method_repeat"]
                )
            )
        run_test_audit(*common, screen_dir, validation_dir, audit_dir)

        with (audit_dir / "shortlist_audit.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(tuple(reader.fieldnames or ()), AUDIT_FIELDS)
            rows = list(reader)
        primary = next(
            row
            for row in rows
            if row["target_domain"] == "clipart"
            and row["method"] == "target_a"
            and int(row["shortlist_size"]) == 10
        )
        recall = float(primary["true_top_q_recall"])
        self.assertGreaterEqual(recall, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertAlmostEqual(
            float(primary["shortlist_reduction"]), (455 - 10) / 455
        )

        dino_dir = self.root / "audit_dino"
        shutil.copytree(audit_dir, dino_dir)
        audit_path = dino_dir / "shortlist_audit.csv"
        with audit_path.open("r", encoding="utf-8", newline="") as stream:
            dino_rows = list(csv.DictReader(stream))
        for row in dino_rows:
            row["encoder"] = "dinov2_b14"
        write_csv_atomic(audit_path, AUDIT_FIELDS, dino_rows)
        dino_manifest = json.loads(
            (dino_dir / "manifest.json").read_text(encoding="utf-8")
        )
        dino_manifest["encoder"] = "dinov2_b14"
        dino_manifest["audit_file_sha256"] = sha256_file(audit_path)
        dino_core = {
            key: value
            for key, value in dino_manifest.items()
            if key != "test_audit_id"
        }
        dino_manifest["test_audit_id"] = canonical_json_sha256(dino_core)
        write_json_atomic(dino_dir / "manifest.json", dino_manifest)
        summary = summarize(
            self.config_path,
            [audit_dir, dino_dir],
            self.root / "summary",
        )
        self.assertEqual(summary["primary_result"]["unit_count"], 6)
        self.assertIsInstance(summary["h3_passed"], bool)


if __name__ == "__main__":
    unittest.main()
