import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_target_conditioned_h5 as experiment  # noqa: E402


class TargetConditionedH5Tests(unittest.TestCase):
    def test_packed_gram_byte_count(self):
        self.assertEqual(experiment.packed_gram_bytes(3, 4), 3 * 10 * 8)

    def test_gate_uses_packed_bytes_and_requires_full_grid(self):
        base = {
            "seed": 1,
            "dimension": 8,
            "candidate_count": 4,
            "budget": 1,
            "target_samples": 16,
            "target_rank": 2,
            "target_moment_relative_error": 0.1,
            "strategy_id": "fixed_l1",
            "mode": "fixed_rank",
            "rank_cap": 1,
            "rank_cap_fraction": 0.125,
            "rank_schedule": "1",
            "fallback_to_full": 0,
            "selected": "a",
            "full_selected": "a",
            "selection_match": 1,
            "proxy_risk": 1.0,
            "full_proxy_risk": 1.0,
            "relative_proxy_risk_vs_full_greedy": 0.0,
            "true_target_risk": 1.0,
            "full_true_target_risk": 1.0,
            "relative_true_risk_vs_full_greedy": 0.0,
            "transmitted_bytes": 10,
            "dense_full_gram_bytes": 100,
            "packed_full_gram_bytes": 50,
            "byte_ratio_vs_dense": 0.1,
            "byte_ratio_vs_packed": 0.2,
            "selection_seconds": 0.01,
            "decomposition_seconds": 0.01,
            "certified_steps": 1,
            "certificate_rate": 1.0,
            "certified_comparisons": 2,
            "total_comparisons": 3,
            "pair_certificate_rate": 2 / 3,
            "covered_intervals": 4,
            "total_intervals": 4,
            "interval_coverage_rate": 1.0,
            "mean_selected_interval_width": 0.1,
        }
        summary = experiment.summarize([base], configuration_count=1)
        gate = summary["h5_gate"]
        self.assertEqual(gate["byte_denominator"], "packed symmetric float64 Gram")
        self.assertEqual(gate["status"], "passed")

        incomplete = experiment.summarize([base], configuration_count=2)
        self.assertEqual(incomplete["h5_gate"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
