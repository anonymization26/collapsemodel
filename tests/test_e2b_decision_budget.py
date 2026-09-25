import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_e2b_decision_budget import aggregate_rows, paired_comparisons, select_reference


class BudgetTests(unittest.TestCase):
    def row(self, domain="a", seed=1, method="target_a", budget=227, gap=0.):
        return dict(encoder="resnet50", projection_seed=seed, target_domain=domain, readout="ridge",
                    metric="brier_score", shortlist_size=budget, method=method, test_loss=1 + gap,
                    oracle_loss=.9, full_validation_test_loss=1., gap_to_full=gap,
                    absolute_omission=0., recall_at_top_q=.8)

    def test_full_reference_is_validation_choice_and_lexicographic(self):
        validation = {"a": {"brier_score": 1.}, "b": {"brier_score": .8}}
        test = {"a": {"brier_score": .1}, "b": {"brier_score": .9}}
        self.assertEqual(select_reference(validation, test, "brier_score"), ("b", .9))
        validation["a"]["brier_score"] = .8
        self.assertEqual(select_reference(validation, test, "brier_score"), ("a", .1))

    def test_failures_are_counted_before_random_and_domain_averaging(self):
        rows = [self.row(method="random", gap=.02), self.row(method="random", gap=0.),
                self.row(domain="b", method="random", gap=.02)]
        _, _, failures = aggregate_rows(rows, {"brier_score": [.015]})
        self.assertAlmostEqual(failures[0]["failure_rate"], .75)

    def test_pairing_uses_all_repeats_even_at_primary_budget(self):
        units = []
        for seed, gap in ((1, .1), (2, .3)):
            units += [self.row(seed=seed, gap=gap), self.row(seed=seed, method="random"),
                      self.row(seed=seed, method="exhaustive", budget=455)]
        domains, _ = paired_comparisons(units)
        self.assertEqual(len(domains), 2)
        self.assertTrue(all(abs(r["paired_loss_difference"] - .2) < 1e-12 for r in domains))

    def test_negative_gap_is_not_clipped(self):
        _, curves, failures = aggregate_rows([self.row(gap=-.02)], {"brier_score": [.01]})
        self.assertEqual(curves[0]["gap_to_full"], -.02)
        self.assertEqual(failures[0]["failure_rate"], 0.)


if __name__ == "__main__":
    unittest.main()
