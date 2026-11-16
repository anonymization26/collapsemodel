import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_e2b_readout_projection import aggregate, averages, collapse_random_repeats, MEASURES


def row(domain, value, group="target_a", seed=20260917, encoder="resnet50"):
    return {"encoder": encoder, "projection_seed": seed, "target_domain": domain,
            "readout": "ridge", "metric": "brier_score", "shortlist_size": 227,
            "group": group, **{k: value for k in MEASURES}}


class SummaryTests(unittest.TestCase):
    def test_random_repeats_collapse_before_domain_averaging(self):
        rows = [row("a", .2, "random:0"), row("a", .6, "random:1"), row("b", .8, "random:0")]
        units = collapse_random_repeats(rows)
        self.assertEqual(len(units), 2)
        result = aggregate(units, {20260917})[0]
        self.assertAlmostEqual(result["recall_at_top_q"], .6)
        self.assertEqual(result["target_domain_count"], 2)

    def test_domains_not_number_of_seed_repetitions_determine_weight(self):
        rows = [row("a", .2), row("a", .2, seed=20260918), row("b", .8)]
        result = aggregate(collapse_random_repeats(rows), {20260917, 20260918})[0]
        self.assertAlmostEqual(result["recall_at_top_q"], .5)
        self.assertEqual(result["repeated_task_count"], 3)

    def test_base_seed_is_not_pooled_with_new_seeds(self):
        rows = [row("a", .1, seed=20260911), row("a", .9)]
        units = collapse_random_repeats(rows)
        self.assertAlmostEqual(aggregate(units, {20260911})[0]["recall_at_top_q"], .1)
        self.assertAlmostEqual(aggregate(units, {20260917})[0]["recall_at_top_q"], .9)

    def test_undefined_denominators_are_not_zero_regret(self):
        r = row("a", .1)
        r["relative_regret"] = None
        result = averages([r])
        self.assertIsNone(result["relative_regret"])
        self.assertEqual(result["relative_regret_defined_count"], 0)

    def test_partial_summary_lists_observed_not_only_requested_seeds(self):
        result = aggregate(collapse_random_repeats([row("a", .5)]), {20260917, 20260918})[0]
        self.assertEqual(result["projection_seeds"], [20260917])
        self.assertEqual(result["requested_projection_seeds"], [20260917, 20260918])


if __name__ == "__main__":
    unittest.main()
