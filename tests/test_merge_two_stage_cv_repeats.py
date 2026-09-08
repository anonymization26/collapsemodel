import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_two_stage_cv_repeats as merge  # noqa: E402


def row(seed, validation, test=0.8):
    return {
        "encoder": "encoder",
        "source": "source",
        "adapter_seed": "0",
        "target": "target",
        "target_cv_folds": "2",
        "target_cv_seed": str(seed),
        "validation_accuracy": str(validation),
        "validation_accuracy_std": "0.1",
        "validation_fold_accuracies": f"{validation - 0.1}|{validation + 0.1}",
        "target_validation_protocol": "stratified_2_fold",
        "test_accuracy": str(test),
        "training_seconds": "1.0",
        "evaluation_seconds": "2.0",
    }


class MergeCvRepeatsTests(unittest.TestCase):
    def test_merge_averages_partitions_but_preserves_adapter_key(self):
        merged = merge.merge_rows([row(1, 0.7), row(2, 0.9)])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["validation_accuracy"], 0.8)
        self.assertEqual(merged[0]["cv_partition_count"], 2)
        self.assertEqual(merged[0]["target_cv_seed"], "1|2")
        self.assertEqual(
            merged[0]["target_validation_protocol"],
            "repeated_stratified_2_fold",
        )

    def test_merge_rejects_test_metric_changes(self):
        with self.assertRaises(ValueError):
            merge.merge_rows([row(1, 0.7, 0.8), row(2, 0.9, 0.7)])

    def test_merge_rejects_duplicate_partition(self):
        with self.assertRaises(ValueError):
            merge.merge_rows([row(1, 0.7), row(1, 0.9)])


if __name__ == "__main__":
    unittest.main()
