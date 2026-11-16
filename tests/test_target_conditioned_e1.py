import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_target_conditioned_e1 as experiment  # noqa: E402
from metrics.target_conditioned import aggregate_gram, target_a_objective  # noqa: E402


class TargetConditionedE1Tests(unittest.TestCase):
    def setUp(self):
        self.problem = experiment.make_synthetic_problem(
            seed=7,
            dimension=12,
            candidate_count=6,
            source_samples=20,
            target_samples=30,
            target_rank=3,
        )

    def test_problem_has_psd_compatible_statistics(self):
        self.assertEqual(len(self.problem.blocks), 6)
        self.assertEqual(self.problem.target_moment.shape, (12, 12))
        self.assertEqual(self.problem.estimated_target_moment.shape, (12, 12))
        for name, gram in self.problem.blocks.items():
            self.assertEqual(self.problem.features[name].shape, (20, 12))
            self.assertGreaterEqual(float(np.linalg.eigvalsh(gram)[0]), -1e-10)

    def test_problem_seed_is_shared_across_controlled_target_factors(self):
        common = (17, 32, 8)
        self.assertEqual(
            experiment.synthetic_problem_seed(*common),
            experiment.synthetic_problem_seed(*common),
        )
        self.assertNotEqual(
            experiment.synthetic_problem_seed(*common),
            experiment.synthetic_problem_seed(18, 32, 8),
        )

    def test_target_sample_sweep_keeps_source_candidates_fixed(self):
        common = {
            "seed": experiment.synthetic_problem_seed(17, 12, 6),
            "dimension": 12,
            "candidate_count": 6,
            "source_samples": 20,
            "target_rank": 3,
        }
        small = experiment.make_synthetic_problem(target_samples=16, **common)
        large = experiment.make_synthetic_problem(target_samples=64, **common)
        np.testing.assert_allclose(small.target_moment, large.target_moment)
        self.assertFalse(np.allclose(
            small.estimated_target_moment,
            large.estimated_target_moment,
        ))
        for name in small.features:
            np.testing.assert_allclose(small.features[name], large.features[name])
            np.testing.assert_allclose(
                small.source_directions[name], large.source_directions[name]
            )

    def test_enumerated_oracle_is_sorted_and_exact(self):
        ordered = experiment.enumerate_combination_risks(
            self.problem, k=2, max_combinations=100
        )
        self.assertEqual(len(ordered), 15)
        self.assertEqual([value for _, value in ordered], sorted(
            value for _, value in ordered
        ))
        selected, objective = ordered[0]
        direct = target_a_objective(
            self.problem.target_moment,
            self.problem.prior_precision,
            aggregate_gram(self.problem.blocks, selected),
        )
        self.assertAlmostEqual(objective, direct, places=12)

    def test_classic_baselines_return_valid_unique_selections(self):
        context = experiment.build_classic_baseline_context(
            self.problem, subspace_rank=4
        )
        selections = experiment.classic_baseline_selections(
            self.problem, context, k=2
        )
        self.assertEqual(set(selections), {
            "collapse_4s",
            "spectrum_rank_l",
            "facility_subspace",
            "kcenter_subspace",
            "dpp_subspace",
        })
        for selected in selections.values():
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(set(selected)), 2)
            self.assertTrue(set(selected).issubset(self.problem.blocks))

        for selected in [
            experiment.target_energy_greedy(self.problem, 2),
            experiment.second_moment_mmd_greedy(self.problem, 2),
        ]:
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(set(selected)), 2)

    def test_shared_records_group_random_repetitions(self):
        rows = experiment.shared_model_records(
            problem=self.problem,
            seed=11,
            dimension=12,
            candidate_count=6,
            budget=2,
            target_samples=30,
            target_rank=3,
            sketch_ranks=[2, 3],
            random_repeats=4,
            max_combinations=100,
        )
        random_rows = [row for row in rows if row["method"] == "random"]
        self.assertEqual(len(random_rows), 4)
        self.assertEqual(
            sorted(int(row["random_repeat"]) for row in random_rows),
            [0, 1, 2, 3],
        )
        summary = experiment.summarize(rows, [])
        self.assertEqual(summary["shared_model"]["random"]["n"], 4)

        csv_like_rows = []
        for row in rows:
            csv_row = dict(row)
            csv_row.setdefault("certificate_rate", "")
            csv_row.setdefault("byte_ratio", "")
            csv_like_rows.append(csv_row)
        csv_summary = experiment.summarize(csv_like_rows, [])
        self.assertEqual(csv_summary["shared_model"]["random"]["n"], 4)
        self.assertIn("h2_same_information", csv_summary["gates"])

    def test_conditional_shift_preserves_selection_but_changes_utility(self):
        rows = experiment.conditional_shift_records(
            problem=self.problem,
            seed=13,
            dimension=12,
            candidate_count=6,
            budget=2,
            target_samples=30,
            target_rank=3,
            shift_levels=[0.0, 1.0],
            target_test_samples=80,
        )
        by_key = {(row["shift"], row["method"]): row for row in rows}
        self.assertEqual(
            by_key[(0.0, "target_a_estimated")]["selected"],
            by_key[(1.0, "target_a_estimated")]["selected"],
        )
        self.assertNotEqual(
            by_key[(0.0, "target_a_estimated")]["target_mse"],
            by_key[(1.0, "target_a_estimated")]["target_mse"],
        )


if __name__ == "__main__":
    unittest.main()
