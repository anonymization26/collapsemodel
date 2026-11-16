import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned import (  # noqa: E402
    adaptive_sketch_target_a,
    aggregate_gram,
    block_marginal_gain,
    candidate_sketch_intervals,
    certified_minimum,
    combine_ridge_statistics,
    continuous_target_a,
    decompose_psd,
    d_opt_information_gain,
    exhaustive_target_a,
    gram_from_features,
    greedy_d_optimal,
    greedy_target_a,
    make_feature_sketch,
    make_psd_sketch,
    posterior_covariance,
    ridge_squared_risk,
    ridge_statistics,
    sample_marginal_gain,
    second_moment,
    sketch_from_decomposition,
    sketch_target_a_interval,
    target_a_objective,
    target_a_risk,
)


class TargetConditionedTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260910)
        self.dimension = 5
        prior_factor = self.rng.normal(size=(self.dimension, self.dimension))
        self.prior = prior_factor.T @ prior_factor + np.eye(self.dimension)
        target_factor = self.rng.normal(size=(self.dimension, 3))
        self.target = target_factor @ target_factor.T / 3.0

    def test_weighted_gram_and_second_moment(self):
        features = self.rng.normal(size=(7, self.dimension))
        weights = np.linspace(0.5, 1.5, len(features))
        expected = sum(
            weight * np.outer(row, row)
            for row, weight in zip(features, weights)
        )
        np.testing.assert_allclose(
            gram_from_features(features, weights), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            second_moment(features, weights), expected / weights.sum(), atol=1e-12
        )

    def test_target_risk_and_weighted_sample_marginal_are_exact(self):
        base_features = self.rng.normal(size=(8, self.dimension))
        base_gram = gram_from_features(base_features)
        posterior = posterior_covariance(self.prior, base_gram, noise_variance=0.7)
        before = target_a_risk(self.target, posterior)
        feature = self.rng.normal(size=self.dimension)
        weight = 2.3
        after = target_a_objective(
            self.target,
            self.prior,
            base_gram + weight * np.outer(feature, feature),
            noise_variance=0.7,
        )
        gain = sample_marginal_gain(
            posterior,
            self.target,
            feature,
            noise_variance=0.7,
            weight=weight,
        )
        self.assertAlmostEqual(gain, before - after, places=11)

    def test_woodbury_block_marginal_is_exact(self):
        base = self.rng.normal(size=(6, self.dimension))
        block = self.rng.normal(size=(3, self.dimension))
        base_gram = gram_from_features(base)
        posterior = posterior_covariance(self.prior, base_gram, noise_variance=1.4)
        before = target_a_risk(self.target, posterior)
        after = target_a_objective(
            self.target,
            self.prior,
            base_gram + gram_from_features(block),
            noise_variance=1.4,
        )
        gain = block_marginal_gain(
            posterior, self.target, block, noise_variance=1.4
        )
        self.assertAlmostEqual(gain, before - after, places=11)

    def test_continuous_a_opt_gradient_hessian_and_convexity(self):
        rows = [self.rng.normal(size=(3, self.dimension)) for _ in range(4)]
        blocks = [gram_from_features(row) for row in rows]
        weights = np.array([0.2, 0.4, 0.7, 0.9])
        objective, gradient, hessian = continuous_target_a(
            weights, blocks, self.target, self.prior, noise_variance=0.8
        )
        self.assertGreater(objective, 0.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(hessian)[0]), -1e-10)

        epsilon = 1e-5
        numerical_gradient = np.empty_like(gradient)
        for index in range(len(weights)):
            direction = np.zeros_like(weights)
            direction[index] = epsilon
            plus = continuous_target_a(
                weights + direction,
                blocks,
                self.target,
                self.prior,
                noise_variance=0.8,
            )[0]
            minus = continuous_target_a(
                weights - direction,
                blocks,
                self.target,
                self.prior,
                noise_variance=0.8,
            )[0]
            numerical_gradient[index] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(gradient, numerical_gradient, rtol=2e-6, atol=2e-7)

    def test_d_opt_information_gain_is_monotone_submodular(self):
        features = self.rng.normal(size=(4, self.dimension))
        blocks = {str(i): np.outer(row, row) for i, row in enumerate(features)}
        names = tuple(sorted(blocks))

        def value(selected):
            return d_opt_information_gain(
                self.prior, aggregate_gram(blocks, selected), noise_variance=0.9
            )

        subsets = [
            frozenset(combination)
            for size in range(len(names) + 1)
            for combination in itertools.combinations(names, size)
        ]
        for smaller in subsets:
            for larger in subsets:
                if not smaller.issubset(larger):
                    continue
                self.assertLessEqual(value(smaller), value(larger) + 1e-11)
                for candidate in set(names) - set(larger):
                    gain_small = value(smaller | {candidate}) - value(smaller)
                    gain_large = value(larger | {candidate}) - value(larger)
                    self.assertGreaterEqual(gain_small + 1e-10, gain_large)

    def test_a_opt_benefit_is_not_assumed_submodular(self):
        features = np.array([
            [-1.379107303258097, 13.610895726498347],
            [-0.5438133406059672, 0.6519343443096093],
            [2.008858610711189, 11.588225405885595],
        ])
        blocks = [np.outer(row, row) for row in features]
        prior = np.eye(2)
        target = np.diag([10.0, 0.1])

        def risk(indices):
            gram = sum((blocks[index] for index in indices), start=np.zeros((2, 2)))
            return target_a_objective(target, prior, gram)

        gain_empty = risk(()) - risk((0,))
        gain_after_block_two = risk((2,)) - risk((2, 0))
        self.assertGreater(gain_after_block_two, gain_empty + 1.0)

    def test_greedy_and_exhaustive_have_deterministic_valid_outputs(self):
        rows = {
            name: self.rng.normal(size=(4, self.dimension))
            for name in ["c", "a", "d", "b"]
        }
        blocks = {name: gram_from_features(value) for name, value in rows.items()}
        greedy = greedy_target_a(blocks, self.target, self.prior, k=2)
        exhaustive = exhaustive_target_a(blocks, self.target, self.prior, k=2)
        d_opt = greedy_d_optimal(blocks, self.prior, k=2)
        self.assertEqual(len(greedy.selected), 2)
        self.assertEqual(len(set(greedy.selected)), 2)
        self.assertEqual(len(d_opt.selected), 2)
        self.assertLessEqual(exhaustive.objective, greedy.objective + 1e-12)
        self.assertLess(greedy.steps[0].objective_after, greedy.steps[0].objective_before)

        budgeted = greedy_target_a(
            blocks,
            self.target,
            self.prior,
            k=4,
            costs={"a": 1.0, "b": 1.0, "c": 2.0, "d": 2.0},
            budget=2.0,
        )
        self.assertLessEqual(budgeted.total_cost, 2.0)

    def test_sketch_interval_contains_exact_risk(self):
        blocks = {}
        sketches = {}
        for name in ["a", "b", "c"]:
            features = self.rng.normal(size=(7, self.dimension))
            blocks[name] = gram_from_features(features)
            sketches[name] = make_psd_sketch(blocks[name], rank=2)
        selected = ["a", "c"]
        interval = sketch_target_a_interval(
            [sketches[name] for name in selected], self.target, self.prior
        )
        exact = target_a_objective(
            self.target, self.prior, aggregate_gram(blocks, selected)
        )
        self.assertLessEqual(interval.lower, exact + 1e-11)
        self.assertGreaterEqual(interval.upper, exact - 1e-11)

    def test_reusable_decomposition_matches_direct_sketch(self):
        features = self.rng.normal(size=(9, self.dimension))
        gram = gram_from_features(features)
        direct = make_psd_sketch(gram, rank=3)
        reused = sketch_from_decomposition(decompose_psd(gram), rank=3)
        np.testing.assert_allclose(reused.approximation, direct.approximation)
        self.assertAlmostEqual(
            reused.tail_operator_bound, direct.tail_operator_bound, places=12
        )
        self.assertAlmostEqual(reused.tail_trace, direct.tail_trace, places=12)

    def test_full_rank_feature_sketch_is_exact_and_can_certify(self):
        features = self.rng.normal(size=(9, self.dimension))
        sketch = make_feature_sketch(features, rank=self.dimension)
        np.testing.assert_allclose(
            sketch.approximation, gram_from_features(features), rtol=1e-11, atol=1e-11
        )
        self.assertAlmostEqual(sketch.tail_operator_bound, 0.0)

        prior = np.eye(2)
        target = np.diag([10.0, 1.0])
        sketches = {
            "aligned": make_psd_sketch(np.diag([10.0, 0.0]), rank=2),
            "off_target": make_psd_sketch(np.diag([0.0, 1.0]), rank=2),
        }
        intervals = candidate_sketch_intervals(
            [], list(sketches), sketches, target, prior
        )
        self.assertEqual(certified_minimum(intervals), "aligned")

    def test_adaptive_sketch_matches_unique_full_gram_greedy(self):
        blocks = {
            "aligned": np.diag([12.0, 0.1, 0.0]),
            "mixed": np.diag([3.0, 3.0, 0.1]),
            "off_target": np.diag([0.1, 0.1, 12.0]),
        }
        target = np.diag([8.0, 2.0, 0.1])
        prior = np.eye(3)
        expected = greedy_target_a(blocks, target, prior, k=2)
        actual = adaptive_sketch_target_a(
            blocks,
            target,
            prior,
            k=2,
            rank_schedule=[1, 2],
            fallback_to_full=True,
            decompositions={
                name: decompose_psd(gram) for name, gram in blocks.items()
            },
        )
        self.assertEqual(actual.selected, expected.selected)
        self.assertAlmostEqual(actual.objective, expected.objective, places=12)
        self.assertTrue(all(step.certified for step in actual.steps))
        self.assertGreater(actual.full_gram_bytes, 0)

        with self.assertRaisesRegex(ValueError, "decomposition keys"):
            adaptive_sketch_target_a(
                blocks,
                target,
                prior,
                k=1,
                rank_schedule=[1],
                decompositions={"aligned": decompose_psd(blocks["aligned"])},
            )

    def test_ridge_statistics_match_direct_squared_loss(self):
        train_x = self.rng.normal(size=(20, self.dimension))
        target_x = self.rng.normal(size=(11, self.dimension))
        true_weights = self.rng.normal(size=(self.dimension, 3))
        train_y = train_x @ true_weights + 0.1 * self.rng.normal(size=(20, 3))
        target_y = target_x @ true_weights + 0.1 * self.rng.normal(size=(11, 3))
        first = ridge_statistics(train_x[:8], train_y[:8])
        second = ridge_statistics(train_x[8:], train_y[8:])
        combined = combine_ridge_statistics([first, second])
        target_stats = ridge_statistics(target_x, target_y)
        regularization = 0.4
        weights = np.linalg.solve(
            train_x.T @ train_x + regularization * np.eye(self.dimension),
            train_x.T @ train_y,
        )
        expected = float(np.mean(np.sum((target_x @ weights - target_y) ** 2, axis=1)))
        actual = ridge_squared_risk(combined, target_stats, regularization)
        self.assertAlmostEqual(actual, expected, places=11)

    def test_invalid_psd_input_is_rejected(self):
        with self.assertRaises(ValueError):
            target_a_objective(np.diag([1.0, -1.0]), np.eye(2), np.eye(2))


if __name__ == "__main__":
    unittest.main()
