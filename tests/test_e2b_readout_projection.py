import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_e2b_readout_projection as sweep
import test_e2b_stage_bundle as fixture_module
from metrics.target_conditioned_e2b import E2BArtifactError, sha256_file
from run_target_conditioned_e2b_shortlist import _prediction_metrics


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.spec = sweep.load_spec(ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json")
        self.x = np.random.default_rng(17).normal(size=(90, 8))

    def test_logistic_probabilities_match_sklearn_multiclass_and_binary(self):
        for classes in (2, 3):
            y = np.arange(len(self.x)) % classes
            w, active, _ = sweep.fit_readout(self.x, y, classes, "logistic", self.spec)
            model = LogisticRegression(**self.spec["logistic"]).fit(self.x, y)
            np.testing.assert_allclose(softmax(sweep.model_scores(self.x, w, active), axis=1),
                                       model.predict_proba(self.x), rtol=1e-10, atol=1e-12)

    def test_missing_classes_and_single_class_are_not_relabelled(self):
        for y in (np.where(np.arange(len(self.x)) % 2, 3, 1), np.full(len(self.x), 3)):
            w, active, _ = sweep.fit_readout(self.x, y, 5, "logistic", self.spec)
            p = softmax(sweep.model_scores(self.x, w, active), axis=1)
            np.testing.assert_array_equal(p[:, [0, 2, 4]], 0)
            np.testing.assert_allclose(p.sum(axis=1), 1)
            self.assertTrue(set(np.argmax(p, axis=1)) <= set(y))

    def test_ridge_and_losses_match_original_definition(self):
        y = np.arange(len(self.x)) % 3
        w, active, _ = sweep.fit_readout(self.x, y, 3, "ridge", self.spec)
        expected = np.linalg.solve(self.x.T @ self.x + np.eye(8), self.x.T @ np.eye(3)[y])
        np.testing.assert_allclose(w, expected)
        scores = sweep.model_scores(self.x, w, active)
        old = _prediction_metrics(scores, y, 3, 15)
        new = sweep.prediction_losses(scores, y)
        self.assertAlmostEqual(old["brier_score"], new["brier_score"], places=12)
        self.assertAlmostEqual(old["nll"], new["nll"], places=12)
        self.assertAlmostEqual(1 - old["accuracy"], new["error_rate"], places=12)

    def test_nonconverged_logistic_is_not_silently_accepted(self):
        spec = copy.deepcopy(self.spec)
        spec["logistic"]["max_iter"] = 1
        with self.assertRaises(sweep.ConvergenceWarning):
            sweep.fit_readout(self.x, np.arange(len(self.x)) % 3, 3, "logistic", spec)

    def test_selection_is_metric_specific_and_lexicographic(self):
        losses = {"b": {"brier_score": 1., "error_rate": .1},
                  "a": {"brier_score": 1., "error_rate": .2}}
        choices = sweep.selections_for({"target_a": ["b", "a"]}, losses, [2],
                                       ["brier_score", "error_rate"])
        self.assertEqual([r["combination"] for r in choices], ["a", "b"])
        rows = sweep.audit_selection({"target_a": ["b", "a"]}, losses, choices, 1)
        self.assertIsNone(rows[0]["span_normalized_regret"])
        self.assertEqual(rows[0]["top_q_boundary_ties"], 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.StageBundleTests()
        self.fixture.setUp()
        self.f = self.fixture.fixture
        self.protocol = ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json"
        self.root = self.f.root / "supplement"

    def tearDown(self):
        self.fixture.tearDown()

    def stage(self, name):
        return sweep.run(name, self.fixture.bundles / name, self.f.config_path,
                         self.protocol, "resnet50", 20260911, self.root)

    def test_end_to_end_freezes_models_without_mutating_base(self):
        before = sha256_file(self.f.config_path)
        for stage in ("screen", "validate", "test-audit"):
            result = self.stage(stage)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["access"]["target_test"], stage == "test-audit")
        self.assertEqual(before, sha256_file(self.f.config_path))
        rows = json.loads((self.root / "test-audit/audit.json").read_text())["rows"]
        self.assertEqual({r["readout"] for r in rows}, {"ridge", "logistic"})
        self.assertTrue(all(0 <= r["recall_at_top_q"] <= 1 for r in rows))
        with self.assertRaises(FileExistsError):
            self.stage("screen")

    def test_missing_freeze_prevents_test_payload_access(self):
        with patch.object(sweep, "load_stage_bundle", side_effect=AssertionError("early test access")):
            with self.assertRaises(E2BArtifactError):
                self.stage("test-audit")

    def test_tampering_prevents_test_payload_access(self):
        self.stage("screen")
        self.stage("validate")
        path = self.root / "validate/selections.json"
        path.write_text("{}\n")
        with patch.object(sweep, "load_stage_bundle", side_effect=AssertionError("early test access")):
            with self.assertRaises(E2BArtifactError):
                self.stage("test-audit")

    def test_unregistered_seed_is_rejected(self):
        with self.assertRaises(E2BArtifactError):
            sweep.run("screen", self.fixture.bundles / "screen", self.f.config_path,
                      self.protocol, "resnet50", 999, self.root)


if __name__ == "__main__":
    unittest.main()
