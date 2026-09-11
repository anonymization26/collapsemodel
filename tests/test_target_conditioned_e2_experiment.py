import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.target_conditioned import target_a_objective  # noqa: E402
from metrics.target_conditioned_e2 import (  # noqa: E402
    CACHE_SCHEMA,
    E2ArtifactError,
    build_manifest_bundle,
    canonical_json_sha256,
    read_manifest_samples,
    sha256_file,
)
import run_target_conditioned_e2 as experiment  # noqa: E402
import run_target_conditioned_e2_oracle as oracle  # noqa: E402
import summarize_target_conditioned_e2 as summary  # noqa: E402
import summarize_target_conditioned_e2_oracle as oracle_summary  # noqa: E402


class TargetConditionedE2ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        self._make_dataset(self.dataset_root)
        self.source_artifact = self.root / "fixture-release.zip"
        self.source_artifact.write_bytes(b"auditable fixture release")
        self.receipt_path = self.root / "receipt.json"
        self.receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": "target-conditioned-e2-receipt-v1",
                    "dataset": "fixture_domains",
                    "dataset_version": "fixture-v1",
                    "official_page_url": "https://example.org/fixture",
                    "download_urls": ["https://example.org/fixture-release.zip"],
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
            ),
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
        self.feature_path, self.metadata_path = self._write_cache()
        self.config_path = self._write_config()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _make_dataset(root: Path) -> None:
        for domain_index, domain in enumerate(("domain_a", "domain_b")):
            for class_index, class_name in enumerate(("class_a", "class_b")):
                class_root = root / domain / class_name
                class_root.mkdir(parents=True)
                for sample_index in range(10):
                    value = domain_index * 100 + class_index * 30 + sample_index
                    pixels = np.full((6, 6, 3), value, dtype=np.uint8)
                    Image.fromarray(pixels).save(
                        class_root / f"sample_{sample_index:02d}.png"
                    )

    def _write_cache(self) -> tuple[Path, Path]:
        samples = read_manifest_samples(self.bundle)
        rng = np.random.default_rng(23)
        features = rng.normal(size=(len(samples), 5)).astype(np.float32)
        labels = np.asarray([row["class_id"] for row in samples], dtype=np.int64)
        sample_ids = np.asarray([row["sample_id"] for row in samples], dtype="<U64")
        cache_dir = self.root / "cache"
        cache_dir.mkdir()
        feature_path = cache_dir / "features.npz"
        np.savez(feature_path, H=features, y=labels, sample_ids=sample_ids)
        manifest = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))
        metadata = {
            "schema_version": CACHE_SCHEMA,
            "dataset": manifest["dataset"],
            "manifest_id": manifest["manifest_id"],
            "manifest_file_sha256": sha256_file(self.bundle / "manifest.json"),
            "samples_file_sha256": manifest["samples_file_sha256"],
            "assignments_file_sha256": manifest["assignments_file_sha256"],
            "duplicates_file_sha256": manifest["duplicates_file_sha256"],
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
            "code": {
                "git_revision": "fixture-revision",
                "extractor_sha256": "3" * 64,
                "encoder_loader_sha256": "4" * 64,
            },
        }
        metadata_path = cache_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return feature_path, metadata_path

    def _write_config(self) -> Path:
        config = {
            "schema_version": experiment.CONFIG_SCHEMA,
            "status": "frozen-before-method-evaluation",
            "datasets": ["fixture_domains"],
            "encoders": ["fixture_encoder"],
            "feature_transform": {
                "name": "row-l2-normalize",
                "epsilon": 1e-8,
                "intercept": False,
                "centering": "none",
            },
            "selection": {
                "budgets": [1, 2],
                "primary_budget": 1,
                "prior_precision": 1.0,
                "noise_variance": 1.0,
                "subspace_rank": 2,
                "rank_l": 2,
                "random_repeats": 3,
                "random_seed": 7,
                "candidate_cost": "one-per-manifest-block",
                "uses_source_labels": False,
                "uses_target_labels": False,
                "uses_target_calibration": False,
                "uses_target_test": False,
            },
            "evaluation": {
                "ridge_regularization": 1.0,
                "primary_metric": "brier_score",
                "supporting_metrics": [
                    "squared_loss",
                    "nll",
                    "accuracy",
                    "macro_f1",
                    "ece",
                ],
                "ece_bins": 5,
                "target_test_opened_after_selection_freeze": True,
            },
            "primary_analysis": {
                "unit": "dataset-target-domain",
                "encoder_handling": "average-within-unit",
                "bootstrap_repeats": 100,
                "confidence_level": 0.95,
                "noninferiority_relative_tolerance": 0.005,
                "minimum_mean_relative_improvement": 0.02,
                "minimum_noninferior_unit_fraction": 0.7,
            },
            "method_groups": {
                "target_conditioned": [
                    "target_a",
                    "target_energy",
                    "second_moment_mmd",
                ],
                "target_blind": [
                    "bayesian_d",
                    "collapse_4s",
                    "domain_balance",
                    "dpp_subspace",
                    "effective_rank",
                    "facility_subspace",
                    "kcenter_subspace",
                    "random",
                    "size",
                    "spectrum_rank_l",
                ],
            },
            "claim_boundary": "fixture",
        }
        path = self.root / "experiment.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_optimized_target_a_value_matches_canonical_formula(self):
        rng = np.random.default_rng(19)
        source = rng.normal(size=(12, 5))
        target = rng.normal(size=(7, 5))
        gram = source.T @ source
        prior = 0.7 * np.eye(5)
        expected = target_a_objective(
            target.T @ target / len(target),
            prior,
            gram,
            noise_variance=1.3,
        )
        actual = experiment.target_a_value(prior + gram / 1.3, target)
        self.assertAlmostEqual(actual, expected, places=11)

    def test_joint_ridge_and_hutchinson_solve_is_exact_for_hadamard_probes(self):
        rng = np.random.default_rng(41)
        train = rng.normal(size=(9, 5))
        labels = np.asarray([0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int64)
        test = rng.normal(size=(6, 5))
        target = rng.normal(size=(4, 5))
        hadamard = np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ]
        )
        probe_rhs = target.T @ hadamard
        expected_scores, expected_solver = experiment.fit_ridge_scores(
            train, labels, test, 2, 1.0
        )
        scores, solver, estimate, first, second, half_difference = (
            oracle.fit_ridge_scores_and_target_a_proxy(
                train,
                labels,
                test,
                2,
                1.0,
                probe_rhs,
                len(target),
                1.0,
            )
        )
        exact = experiment.target_a_value(
            np.eye(train.shape[1]) + train.T @ train,
            target,
        )
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-12, atol=1e-12)
        self.assertEqual(solver, expected_solver)
        self.assertAlmostEqual(estimate, exact, places=11)
        self.assertAlmostEqual((first + second) / 2.0, estimate, places=13)
        self.assertGreaterEqual(half_difference, 0.0)

    def test_selection_evaluation_and_summary_are_auditable(self):
        output_dir = self.root / "evaluation"
        selection_path = output_dir / "selection.json"
        selection = experiment.run_selection(
            self.bundle,
            self.feature_path,
            self.metadata_path,
            self.config_path,
            selection_path,
        )
        with self.assertRaisesRegex(E2ArtifactError, "refusing to overwrite"):
            experiment.run_selection(
                self.bundle,
                self.feature_path,
                self.metadata_path,
                self.config_path,
                selection_path,
            )
        self.assertEqual(len(selection["tasks"]), 2)
        for task in selection["tasks"]:
            access = task["algorithmic_access"]
            self.assertFalse(access["labels_accessed"])
            self.assertFalse(access["target_calibration_accessed"])
            self.assertFalse(access["target_test_accessed"])
            self.assertEqual(len(task["selections"]), 30)

        evaluation = experiment.run_evaluation(
            self.bundle,
            self.feature_path,
            self.metadata_path,
            self.config_path,
            selection_path,
            output_dir,
        )
        with self.assertRaisesRegex(E2ArtifactError, "refusing to overwrite"):
            experiment.run_evaluation(
                self.bundle,
                self.feature_path,
                self.metadata_path,
                self.config_path,
                selection_path,
                output_dir,
            )
        self.assertEqual(evaluation["row_count"], 60)
        self.assertFalse(
            evaluation["evaluation_access"]["target_calibration_used_by_metrics"]
        )
        result = summary.summarize([output_dir], self.root / "summary")
        self.assertEqual(result["run_count"], 1)
        self.assertEqual(result["task_unit_count"], 2)

        tampered = json.loads(selection_path.read_text(encoding="utf-8"))
        task = tampered["tasks"][0]
        record = task["selections"][0]
        alternative = next(
            value["candidate_id"]
            for value in task["candidate_inventory"]
            if value["candidate_id"] not in record["selected"]
        )
        record["selected"] = [alternative]
        record["selected_order"] = [alternative]
        core = {key: value for key, value in tampered.items() if key != "selection_id"}
        tampered["selection_id"] = canonical_json_sha256(core)
        selection_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(
            E2ArtifactError,
            "selection sample count|selection sample hash|selection domain counts",
        ):
            experiment.run_evaluation(
                self.bundle,
                self.feature_path,
                self.metadata_path,
                self.config_path,
                selection_path,
                self.root / "tampered_evaluation",
            )

    def test_exhaustive_oracle_reproduces_parent_and_summarizes(self):
        parent_dir = self.root / "parent"
        selection_path = parent_dir / "selection.json"
        experiment.run_selection(
            self.bundle,
            self.feature_path,
            self.metadata_path,
            self.config_path,
            selection_path,
        )
        parent_manifest = experiment.run_evaluation(
            self.bundle,
            self.feature_path,
            self.metadata_path,
            self.config_path,
            selection_path,
            parent_dir,
        )
        oracle_config = {
            "schema_version": oracle.CONFIG_SCHEMA,
            "status": "frozen-before-exploratory-oracle-evaluation",
            "parent_experiment_config_file_sha256": sha256_file(self.config_path),
            "parent_evaluation_runner_revision": parent_manifest["runner_revision"],
            "datasets": ["fixture_domains"],
            "encoders": ["fixture_encoder"],
            "analysis": {
                "budgets": [1, 2],
                "primary_budget": 1,
                "primary_metric": "brier_score",
                "top_q": 3,
                "normalized_regret_denominator": "absolute-oracle-risk",
                "combination_tie_break": "metric-then-lexicographic",
                "target_a_proxy": "common-rademacher-hutchinson",
                "hutchinson_probes": 4,
                "hutchinson_seed": 101,
                "probe_halves_reported": True,
                "expected_candidate_count_per_task": 4,
            },
            "access": {
                "target_test_used_to_define_oracle": True,
                "parent_selection_remains_immutable": True,
                "changes_parent_confirmatory_gates": False,
            },
            "claim_boundary": "post-hoc fixture diagnostic only",
        }
        oracle_config_path = self.root / "oracle.json"
        oracle_config_path.write_text(json.dumps(oracle_config), encoding="utf-8")
        oracle_dir = self.root / "oracle"
        with mock.patch.object(
            experiment, "git_revision", return_value="new-oracle-revision"
        ):
            oracle_manifest = oracle.run_oracle(
                self.bundle,
                self.feature_path,
                self.metadata_path,
                self.config_path,
                oracle_config_path,
                parent_dir,
                oracle_dir,
            )
        self.assertEqual(oracle_manifest["runner_revision"], "new-oracle-revision")
        self.assertEqual(oracle_manifest["combination_row_count"], 20)
        self.assertEqual(oracle_manifest["selection_diagnostic_row_count"], 60)
        self.assertEqual(oracle_manifest["task_summary_row_count"], 4)
        self.assertFalse(
            oracle_manifest["evaluation_access"]["changes_parent_confirmatory_gates"]
        )
        with self.assertRaisesRegex(E2ArtifactError, "refusing to overwrite"):
            oracle.run_oracle(
                self.bundle,
                self.feature_path,
                self.metadata_path,
                self.config_path,
                oracle_config_path,
                parent_dir,
                oracle_dir,
            )

        aggregate = oracle_summary.summarize(
            [oracle_dir], self.root / "oracle_summary"
        )
        self.assertEqual(aggregate["run_count"], 1)
        self.assertEqual(aggregate["combination_row_count"], 20)
        self.assertEqual(aggregate["task_unit_count"], 2)
        self.assertEqual(aggregate["primary"]["budget"], 1)
        self.assertIn(
            aggregate["primary"]["strongest_parent_target_blind"],
            {
                "bayesian_d",
                "collapse_4s",
                "domain_balance",
                "dpp_subspace",
                "effective_rank",
                "facility_subspace",
                "kcenter_subspace",
                "random",
                "size",
                "spectrum_rank_l",
            },
        )
        with (oracle_dir / "combinations.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(E2ArtifactError, "hash mismatch"):
            oracle_summary.validate_run(oracle_dir)


if __name__ == "__main__":
    unittest.main()
