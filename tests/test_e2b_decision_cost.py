import sys
import subprocess
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_e2b_domain_ties import shuffle_composition_ties
from summarize_e2b_decision_cost import aggregate_task_times, feature_accounting, random_union_feature_seconds


class CostAndTieTests(unittest.TestCase):
    def test_cost_summary_is_a_standalone_entry_point(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_e2b_decision_cost.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ties_do_not_reorder_different_domain_compositions(self):
        domains = {"a0": "a", "a1": "a", "a2": "a", "b0": "b", "b1": "b"}
        ranking = ["a0|a1|a2", "a0|a1|b0", "a0|a1|b1", "a0|a2|b0"]
        output = shuffle_composition_ties(ranking, domains, 17)
        self.assertEqual(output[0], ranking[0])
        self.assertEqual(set(output), set(ranking))
        self.assertEqual(output, shuffle_composition_ties(ranking, domains, 17))

    def test_noncontiguous_domain_scores_are_rejected(self):
        domains = {"a0": "a", "a1": "a", "a2": "a", "b0": "b"}
        with self.assertRaises(ValueError):
            shuffle_composition_ties(["a0|a1|b0", "a0|a1|a2", "a0|a2|b0"], domains, 7)

    def test_timing_repeats_are_collapsed_before_task_aggregation(self):
        records = [{"encoder": "resnet50", "target_domain": "real", "readout": "ridge",
                    "method": "target_a", "shortlist_size": 50, "repeat": i,
                    "score_seconds": 1, "decision_seconds": value,
                    "test_brier_score": (0.05, 0.10, 0.40)[i]}
                   for i, value in enumerate((2, 4, 99))]
        result = aggregate_task_times(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["median_resident_seconds"], 4)
        self.assertEqual(result[0]["resident_seconds"], 35)
        self.assertAlmostEqual(result[0]["test_brier_score"], 0.55 / 3)
        with self.assertRaises(ValueError):
            aggregate_task_times(records[:2])
        with self.assertRaises(ValueError):
            aggregate_task_times(records + [records[0]])

    def test_random_feature_cost_charges_union_once_not_all_source_blocks(self):
        profile = {"source_blocks": [{"block_id": f"b{i}", "count": 512, "seconds": 2}
                                     for i in range(18)]}
        seconds, count = random_union_feature_seconds(profile,
            [["b0|b1|b2", "b0|b1|b3"]] * 3, 2)
        self.assertEqual(count, 4)
        self.assertEqual(seconds, 8)

    def test_feature_accounting_excludes_target_source_and_charges_selection_only_to_screen(self):
        groups = [{"domain": str(d), "role": role, "count": count, "seconds": seconds}
                  for d in range(6)
                  for role, count, seconds in (("anchor_pool", 1536, 2),
                                               ("target_selection", 1536, 3),
                                               ("target_validation", 1024, 4))]
        profile = {"model_load_seconds": 1, "groups": groups,
                   "per_target_accounting": [{"target_domain": str(d),
                        "baseline_feature_seconds": 15, "target_aware_feature_seconds": 18}
                       for d in range(6)]}
        expected = feature_accounting(profile)
        self.assertEqual(expected["0"]["baseline_feature_seconds"], 1 + 5 * 2 + 4)
        self.assertEqual(expected["0"]["target_aware_feature_seconds"], 18)
        profile["per_target_accounting"][0]["baseline_feature_seconds"] = 17
        with self.assertRaises(ValueError):
            feature_accounting(profile)
        profile["per_target_accounting"][0]["baseline_feature_seconds"] = 15
        profile["groups"].append(groups[0])
        with self.assertRaises(ValueError):
            feature_accounting(profile)


if __name__ == "__main__":
    unittest.main()
