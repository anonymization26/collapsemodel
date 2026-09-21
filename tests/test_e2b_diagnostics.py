import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_e2b_stage_bundle as fixtures
from metrics.target_conditioned_e2b import E2BArtifactError, write_json_atomic
from run_target_conditioned_e2b_diagnostics import (
    METRICS, bootstrap_topq, fit_sample_losses, run_diagnostics, sample_losses,
)
from run_target_conditioned_e2b_shortlist import (
    _evaluate_combinations, _prediction_metrics, run_screen, run_validation, run_test_audit,
)


class SampleDiagnosticsTests(unittest.TestCase):
    def test_per_sample_means_match_reference_including_extreme_scores(self):
        rng = np.random.default_rng(51)
        for scale in (0.0, 1.0, 1000.0):
            scores = rng.normal(size=(17, 5)) * scale
            labels = np.asarray([0, 0, 0, 4, 2, 1, 1, 0, 2, 4, 0, 2, 0, 1, 0, 1, 0])
            expected = _prediction_metrics(scores, labels, 5, 15)
            for metric, values in sample_losses(scores, labels).items():
                target = 1.0 - expected["accuracy"] if metric == "error_rate" else expected[metric]
                self.assertAlmostEqual(float(values.mean()), target, places=12)

    def test_ridge_reconstruction_matches_existing_implementation(self):
        rng = np.random.default_rng(74)
        blocks = {name: rng.normal(size=(5, 4)) for name in ("a", "b", "c")}
        labels = {name: rng.integers(0, 3, size=5) for name in blocks}
        target, truth = rng.normal(size=(8, 4)), rng.integers(0, 3, size=8)
        combinations = ["a|b", "a|c", "b|c"]
        expected = _evaluate_combinations(combinations, blocks, labels, target, truth, 3, 1, 15,
                                          progress_label="fixture")
        losses = fit_sample_losses(combinations, blocks, labels, target, truth, 3, 1)
        for index, row in enumerate(expected):
            for metric in METRICS:
                value = 1.0 - row["accuracy"] if metric == "error_rate" else row[metric]
                self.assertAlmostEqual(float(losses[metric][index].mean()), value, places=12)

    def test_bootstrap_is_paired_reproducible_and_tie_deterministic(self):
        losses = np.asarray([[0., 1., 4.], [4., 1., 0.], [2., 2., 2.], [0., 1., 4.]])
        masks = np.asarray([[True] * 4, [True, False, False, True]])
        first = bootstrap_topq(losses, masks, 2, 37, 7, 517)
        second = bootstrap_topq(losses, masks, 2, 37, 7, 517)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        np.testing.assert_array_equal(first["recalls"][:, 0], np.ones(37))
        self.assertAlmostEqual(float(first["frequency"].sum()), 2)
        # A direct sample-weight calculation must select exactly the same indices.
        counts = np.random.default_rng(517).multinomial(3, np.full(3, 1 / 3), size=37)
        direct = np.argsort(counts @ losses.T / 3, axis=1, kind="stable")[:, :2]
        np.testing.assert_array_equal(first["indices"], direct)
        np.testing.assert_array_equal(first["recalls"], masks[:, direct].mean(axis=2).T)
        tied = bootstrap_topq(np.ones((4, 3)), masks, 2, 9, 4, 44)
        np.testing.assert_array_equal(tied["indices"], np.tile([0, 1], (9, 1)))
        np.testing.assert_array_equal(tied["overlaps"], np.ones(9))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(E2BArtifactError):
            sample_losses(np.asarray([[np.nan, 0.]]), np.asarray([0]))
        with self.assertRaises(E2BArtifactError):
            bootstrap_topq(np.ones((3, 2)), np.ones((1, 3)), 4, 10, 5, 1)


class DiagnosticsPipelineTests(unittest.TestCase):
    def setUp(self):
        self.bundle_fixture = fixtures.StageBundleTests()
        self.bundle_fixture.setUp()
        self.f = self.bundle_fixture.fixture
        self.settings_path = self.f.root / "diagnostics.json"
        settings = json.loads((ROOT / "code/configs/target_conditioned_e2b/diagnostics_v1.json").read_text())
        settings.update(bootstrap_repeats=12, bootstrap_batch_size=5)
        write_json_atomic(self.settings_path, settings)
        self.screen, self.validation, self.audit = [self.f.root / s for s in ("screen", "validation", "audit")]

    def tearDown(self):
        self.bundle_fixture.tearDown()

    def run_diagnostic(self, output):
        return run_diagnostics(self.f.config_path, self.settings_path,
                               self.bundle_fixture.bundles / "test-audit", "resnet50",
                               self.screen, self.validation, self.audit, output)

    def test_missing_freeze_is_rejected_before_loading_features(self):
        with patch("run_target_conditioned_e2b_diagnostics._stage_inputs",
                   side_effect=AssertionError("test features opened too soon")):
            with self.assertRaises(E2BArtifactError):
                self.run_diagnostic(self.f.root / "diagnostics")

    def test_end_to_end_mean_checks_and_tampering_rejection(self):
        f = self.bundle_fixture
        run_screen(*f.isolated, self.screen, stage_bundle=f.bundles / "screen")
        run_validation(*f.isolated, self.screen, self.validation, stage_bundle=f.bundles / "validate")
        run_test_audit(*f.isolated, self.screen, self.validation, self.audit, stage_bundle=f.bundles / "test-audit")
        result = self.run_diagnostic(self.f.root / "diagnostics")
        self.assertEqual(len(result["domains"]), 6)
        self.assertTrue(all(d["loss_mean_max_abs_error"] < 1e-10 for d in result["domains"]))
        self.assertFalse(result["test_outcomes_change_selection"])
        with self.assertRaises(E2BArtifactError):
            self.run_diagnostic(self.f.root / "diagnostics")
        # Corrupt the sealed CSV while leaving the manifest unchanged.
        with (self.audit / "combinations.csv").open("a") as stream:
            stream.write("\n")
        with patch("run_target_conditioned_e2b_diagnostics._stage_inputs",
                   side_effect=AssertionError("test features opened too soon")):
            with self.assertRaises(E2BArtifactError):
                self.run_diagnostic(self.f.root / "tampered")


if __name__ == "__main__":
    unittest.main()
