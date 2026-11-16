import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_target_conditioned_e1 as merge  # noqa: E402


class MergeTargetConditionedTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "dimensions": [16],
            "candidate_counts": [8],
            "budgets": [1],
            "target_samples": [32],
            "target_rank_fractions": [0.25],
            "source_samples": 64,
            "mini": False,
        }

    def test_merge_configs_combines_disjoint_seeds(self):
        merged = merge.merge_configs([
            {"seeds": [2], **self.base},
            {"seeds": [1], **self.base},
        ])
        self.assertEqual(merged["seeds"], [1, 2])

    def test_merge_configs_rejects_overlap_and_mismatch(self):
        with self.assertRaisesRegex(ValueError, "more than one"):
            merge.merge_configs([
                {"seeds": [1], **self.base},
                {"seeds": [1], **self.base},
            ])
        with self.assertRaisesRegex(ValueError, "differ"):
            merge.merge_configs([
                {"seeds": [1], **self.base},
                {"seeds": [2], **{**self.base, "dimensions": [32]}},
            ])

    def test_require_unique_rejects_duplicate_rows(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge.require_unique(
                [{"seed": "1"}, {"seed": "1"}],
                ("seed",),
                "test",
            )

    def test_require_common_revision_rejects_mixed_sources(self):
        manifests = [
            {"provenance": {"git_revision": "a"}},
            {"provenance": {"git_revision": "b"}},
        ]
        with self.assertRaisesRegex(ValueError, "mix git revisions"):
            merge.require_common_revision(manifests, "test")

    def test_controlled_target_sample_audit_accepts_fixed_controls(self):
        shared = []
        conditional = []
        for target_samples in (16, 32):
            base = {
                "seed": "1",
                "dimension": "16",
                "candidate_count": "8",
                "budget": "1",
                "target_samples": str(target_samples),
                "target_rank": "4",
            }
            shared.append({
                **base,
                "method": "effective_rank",
                "random_repeat": "",
                "selected": "pool_00",
                "target_risk": "1.0",
                "oracle_risk": "0.5",
            })
            shared.extend([
                {
                    **base,
                    "method": "target_a_estimated",
                    "random_repeat": "",
                    "selected": "pool_00",
                    "normalized_regret": "0.0",
                    "combination_rank": "1",
                },
                {
                    **base,
                    "method": "target_a_true_ct_diagnostic",
                    "random_repeat": "",
                    "selected": "pool_00",
                    "target_risk": "1.0",
                    "oracle_risk": "0.5",
                    "normalized_regret": "0.0",
                    "combination_rank": "1",
                },
            ])
            conditional.append({
                **base,
                "shift": "0.5",
                "method": "effective_rank",
                "selected": "pool_00",
                "target_mse": "2.0",
                "conditional_oracle_mse": "1.0",
            })
        audit = merge.controlled_target_sample_audit(
            shared,
            conditional,
            [16, 32],
        )
        self.assertEqual(audit["status"], "passed")
        sensitivity = merge.target_sample_sensitivity(shared)
        self.assertEqual(
            sensitivity["16"][
                "selection_match_rate_vs_true_target_moment_greedy"
            ],
            1.0,
        )

    def test_controlled_target_sample_audit_rejects_changed_control(self):
        rows = []
        for target_samples, selected in ((16, "pool_00"), (32, "pool_01")):
            rows.append({
                "seed": "1",
                "dimension": "16",
                "candidate_count": "8",
                "budget": "1",
                "target_samples": str(target_samples),
                "target_rank": "4",
                "method": "effective_rank",
                "random_repeat": "",
                "selected": selected,
                "target_risk": "1.0",
                "oracle_risk": "0.5",
            })
        with self.assertRaisesRegex(ValueError, "invariance failed"):
            merge.require_sweep_invariance(
                rows,
                {"effective_rank"},
                (
                    "seed",
                    "dimension",
                    "candidate_count",
                    "budget",
                    "target_rank",
                    "method",
                    "random_repeat",
                ),
                ("selected", "target_risk", "oracle_risk"),
                [16, 32],
                "test",
            )


if __name__ == "__main__":
    unittest.main()
