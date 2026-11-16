import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import run_e2b_decision_challenge as challenge
import test_e2b_stage_bundle as fixtures
from metrics.target_conditioned_e2b import E2BArtifactError, sha256_file


class ChallengePipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.StageBundleTests()
        self.fixture.setUp()
        self.f = self.fixture.fixture
        self.base_protocol = ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json"
        self.protocol = ROOT / "code/configs/target_conditioned_e2b/decision_cost_v1.json"
        self.output = self.f.root / "challenge"

    def tearDown(self):
        self.fixture.tearDown()

    def run_stage(self, stage, construction="hash_partition"):
        with threadpool_limits(limits=1):
            return challenge.run(stage, self.fixture.bundles / stage, self.f.config_path,
                                 self.base_protocol, self.protocol, "resnet50", construction,
                                 self.output)

    def test_three_stage_audit_freezes_and_preserves_membership(self):
        config_hash = sha256_file(self.f.config_path)
        manifests = [self.run_stage(stage) for stage in ("screen", "validate", "test-audit")]
        self.assertEqual(sha256_file(self.f.config_path), config_hash)
        self.assertEqual(len({m["lineage"]["candidate_membership_id"] for m in manifests}), 1)
        test = json.loads((self.output / "test-audit/audit.json").read_text())["rows"]
        self.assertEqual({r["readout"] for r in test}, {"ridge", "logistic"})
        self.assertEqual(sum(r["group"] == "exhaustive" for r in test), 36)
        self.assertTrue(all(r["shortlist_size"] == 455 for r in test if r["group"] == "exhaustive"))
        self.assertEqual(manifests[-1]["upstream"], {m["stage"]: m["artifact_id"] for m in manifests[:-1]})

    def test_changed_screen_is_rejected_before_validation_data_access(self):
        self.run_stage("screen")
        path = self.output / "screen/rankings.json"
        path.write_text("{}\n")
        with patch.object(challenge, "load_stage_bundle", side_effect=AssertionError("data opened early")):
            with self.assertRaises(E2BArtifactError):
                self.run_stage("validate")

    def test_construction_switch_is_rejected_before_validation_data_access(self):
        self.run_stage("screen", "original")
        with patch.object(challenge, "load_stage_bundle", side_effect=AssertionError("data opened early")):
            with self.assertRaises(E2BArtifactError):
                self.run_stage("validate", "hash_partition")


if __name__ == "__main__":
    unittest.main()
