import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_two_stage_split_sensitivity as sensitivity  # noqa: E402


class SplitSensitivitySummaryTests(unittest.TestCase):
    def test_seed_directory_parser(self):
        seed, directory = sensitivity.parse_seed_directory("17=/tmp/results")
        self.assertEqual(seed, "17")
        self.assertEqual(directory, Path("/tmp/results"))
        with self.assertRaises(ValueError):
            sensitivity.parse_seed_directory("missing-separator")

    def test_average_ranks_assigns_midranks_to_ties(self):
        np.testing.assert_array_equal(
            sensitivity.average_ranks(np.asarray([3.0, 1.0, 1.0, 2.0])),
            np.asarray([4.0, 1.5, 1.5, 3.0]),
        )

    def test_analysis_detects_unstable_oracle(self):
        utility = {
            "1": {
                ("encoder", "target", "a"): 0.9,
                ("encoder", "target", "b"): 0.8,
                ("encoder", "target", "c"): 0.7,
            },
            "2": {
                ("encoder", "target", "a"): 0.7,
                ("encoder", "target", "b"): 0.8,
                ("encoder", "target", "c"): 0.9,
            },
        }
        rows, summary = sensitivity.analyze_utility(utility, tie_tolerance=1e-12)
        self.assertEqual(rows[0]["distinct_primary_oracles"], 2)
        self.assertFalse(rows[0]["has_shared_tie_optimal_source"])
        self.assertEqual(summary["stable_primary_oracle_pairs"], 0)
        self.assertEqual(summary["shared_tie_optimal_source_pairs"], 0)
        self.assertAlmostEqual(summary["pairwise_spearman_mean"], -1.0)

    def test_family_stability_is_distinct_from_exact_source_stability(self):
        utility = {
            "1": {
                ("encoder", "target", "cifar100"): 0.9,
                ("encoder", "target", "cifar100_coarse"): 0.8,
                ("encoder", "target", "other"): 0.7,
            },
            "2": {
                ("encoder", "target", "cifar100"): 0.8,
                ("encoder", "target", "cifar100_coarse"): 0.9,
                ("encoder", "target", "other"): 0.7,
            },
        }
        rows, summary = sensitivity.analyze_utility(
            utility,
            tie_tolerance=1e-12,
            source_families=sensitivity.SOURCE_FAMILY_ALIASES,
        )
        self.assertEqual(rows[0]["distinct_primary_oracles"], 2)
        self.assertEqual(rows[0]["distinct_primary_oracle_families"], 1)
        self.assertEqual(summary["stable_primary_oracle_pairs"], 0)
        self.assertEqual(summary["stable_primary_oracle_family_pairs"], 1)

    def test_joint_cv_reader_averages_adapter_seeds_per_partition(self):
        fields = [
            "encoder", "target", "source", "target_cv_seed",
            "target_cv_partitions", "validation_partition_accuracies",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder" / "adaptation_results.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        "encoder": "e", "target": "t", "source": "s",
                        "target_cv_seed": "11|13", "target_cv_partitions": "2",
                        "validation_partition_accuracies": "0.8|0.6",
                    },
                    {
                        "encoder": "e", "target": "t", "source": "s",
                        "target_cv_seed": "11|13", "target_cv_partitions": "2",
                        "validation_partition_accuracies": "1.0|0.8",
                    },
                ])
            utility, paths = sensitivity.read_joint_cv_utility(Path(directory))
        self.assertEqual(paths, [path])
        self.assertAlmostEqual(utility["11"][("e", "t", "s")], 0.9)
        self.assertAlmostEqual(utility["13"][("e", "t", "s")], 0.7)


if __name__ == "__main__":
    unittest.main()
