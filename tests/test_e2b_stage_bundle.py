import csv
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import test_target_conditioned_e2b as fixtures
from build_target_conditioned_e2b_candidates import build_candidates
from metrics.e2b_stage_bundle import load_stage_bundle, prepare_stage_bundles
from metrics.target_conditioned_e2b import (
    E2BArtifactError, canonical_json_sha256, sha256_file, write_json_atomic,
)
from run_target_conditioned_e2b_shortlist import run_screen, run_validation, run_test_audit


class StageBundleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TargetConditionedE2BTests()
        self.fixture.setUp()
        f = self.fixture
        build_candidates(f.manifest_dir, f.config_path, f.anchor_dir / "features.npz",
                         f.anchor_dir / "metadata.json", f.candidate_path)
        self.bundles = f.root / "bundles"
        self.prepare(self.bundles)
        self.legacy = (f.manifest_dir, f.config_path, f.candidate_path,
                       f.anchor_dir / "features.npz", f.anchor_dir / "metadata.json",
                       f.eval_dir / "features.npz", f.eval_dir / "metadata.json", "resnet50")
        self.isolated = (None, f.config_path, None, None, None, None, None, "resnet50")

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self, path):
        f = self.fixture
        return prepare_stage_bundles(f.manifest_dir, f.config_path, f.candidate_path,
                                     f.eval_dir / "features.npz", f.eval_dir / "metadata.json",
                                     "resnet50", path)

    def reseal(self, stage, mutate):
        path = self.bundles / stage / "manifest.json"
        value = json.loads(path.read_text())
        mutate(value)
        core = {k: v for k, v in value.items() if k != "bundle_id"}
        write_json_atomic(path, {**core, "bundle_id": canonical_json_sha256(core)})

    def test_payload_permissions_and_wrong_stage_rejection(self):
        for stage in ("screen", "validate", "test-audit"):
            loaded = load_stage_bundle(self.bundles / stage, self.fixture.config_path,
                                       stage, "resnet50")
            self.assertEqual(loaded["labels"] is None, stage == "screen")
            self.assertTrue(all(set(row) == {"sample_id", "role", "domain"}
                                for row in loaded["samples"]))
        with self.assertRaises(E2BArtifactError):
            load_stage_bundle(self.bundles / "test-audit", self.fixture.config_path,
                              "validate", "resnet50")
        with self.assertRaises(E2BArtifactError):
            self.prepare(self.bundles)

    def test_forbidden_label_array_is_rejected_even_with_updated_hash(self):
        path = self.bundles / "screen" / "features.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["y"] = np.zeros(len(arrays["H"]), dtype=np.int64)
        np.savez(path, **arrays)
        self.reseal("screen", lambda meta: meta.update(feature_file_sha256=sha256_file(path)))
        with self.assertRaises(E2BArtifactError):
            load_stage_bundle(path.parent, self.fixture.config_path, "screen", "resnet50")

    def test_forbidden_sample_role_is_rejected(self):
        def mutate(meta):
            next(row for row in meta["samples"] if row["role"] == "target_validation")["role"] = "target_test"
        self.reseal("validate", mutate)
        with self.assertRaises(E2BArtifactError):
            load_stage_bundle(self.bundles / "validate", self.fixture.config_path,
                              "validate", "resnet50")

    def test_label_metadata_is_rejected_even_with_updated_identity(self):
        self.reseal("screen", lambda meta: meta.update(class_labels=[0, 1]))
        with self.assertRaises(E2BArtifactError):
            load_stage_bundle(self.bundles / "screen", self.fixture.config_path,
                              "screen", "resnet50")

    def test_forbidden_feature_perturbations_do_not_change_screen_payload(self):
        f = self.fixture
        with np.load(f.eval_dir / "features.npz", allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        with (f.manifest_dir / "samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows):
            if row["role"] in {"target_validation", "target_test"}:
                arrays["H"][i] += 1000
        np.savez(f.eval_dir / "features.npz", **arrays)
        metadata = json.loads((f.eval_dir / "metadata.json").read_text())
        metadata["feature_file_sha256"] = sha256_file(f.eval_dir / "features.npz")
        write_json_atomic(f.eval_dir / "metadata.json", metadata)
        other = f.root / "perturbed"
        self.prepare(other)
        with np.load(self.bundles / "screen" / "features.npz") as a, np.load(other / "screen" / "features.npz") as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])

    def test_isolated_pipeline_matches_legacy_without_original_inputs(self):
        f = self.fixture
        legacy_screen, legacy_validation, legacy_audit = [f.root / ("legacy_" + s) for s in ("screen", "validation", "audit")]
        run_screen(*self.legacy, legacy_screen)
        run_validation(*self.legacy, legacy_screen, legacy_validation)
        run_test_audit(*self.legacy, legacy_screen, legacy_validation, legacy_audit)
        # Workers must succeed when no original manifest, full cache, or anchor exists.
        for directory in (f.manifest_dir, f.anchor_dir, f.eval_dir):
            shutil.rmtree(directory)
        f.candidate_path.unlink()
        screen, validation, audit = [f.root / s for s in ("screen", "validation", "audit")]
        with patch("run_target_conditioned_e2b_shortlist._load_features", side_effect=AssertionError("full cache read")):
            run_screen(*self.isolated, screen, stage_bundle=self.bundles / "screen")
            run_validation(*self.isolated, screen, validation, stage_bundle=self.bundles / "validate")
            run_test_audit(*self.isolated, screen, validation, audit, stage_bundle=self.bundles / "test-audit")
        for old, new, filename, ignore in (
            (legacy_screen, screen, "rankings.csv", set()),
            (legacy_validation, validation, "selections.csv", set()),
            (legacy_audit, audit, "combinations.csv", {"evaluation_seconds"}),
        ):
            with (old / filename).open() as a, (new / filename).open() as b:
                left, right = list(csv.DictReader(a)), list(csv.DictReader(b))
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                for key in x:
                    if key in ignore:
                        continue
                    try:
                        self.assertAlmostEqual(float(x[key]), float(y[key]), places=10)
                    except ValueError:
                        self.assertEqual(x[key], y[key])

    def test_missing_freeze_rejected_before_test_features_are_loaded(self):
        with patch("run_target_conditioned_e2b_shortlist.load_stage_bundle",
                   side_effect=AssertionError("test features opened too soon")):
            with self.assertRaises(E2BArtifactError):
                run_test_audit(*self.isolated, self.fixture.root / "missing_screen",
                               self.fixture.root / "missing_validation", self.fixture.root / "audit",
                               stage_bundle=self.bundles / "test-audit")


if __name__ == "__main__":
    unittest.main()
