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


if __name__ == "__main__":
    unittest.main()
