import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import two_stage_classic_baselines as baselines  # noqa: E402


class ClassicBaselineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.names = ["a", "b", "c", "d", "e"]
        self.features = {
            name: baselines.normalize_rows(rng.normal(size=(8, 6)).astype(np.float32))
            for name in self.names
        }
        self.ranks, _, self.centroids, self.subspaces = baselines.pool_statistics(
            self.features, top_k=3,
        )

    def test_effective_rank_extremes(self):
        identity = np.eye(4, dtype=np.float32)
        repeated = np.ones((4, 1), dtype=np.float32)
        self.assertAlmostEqual(baselines.effective_rank(identity), 4.0, places=6)
        self.assertAlmostEqual(baselines.effective_rank(repeated), 1.0, places=6)

    def test_low_rank_scatter_does_not_count_roundoff_modes(self):
        rng = np.random.default_rng(29)
        matrix = rng.normal(size=(20, 3)) @ rng.normal(size=(3, 100))
        scatter = matrix.T @ matrix
        self.assertLessEqual(baselines.effective_rank_from_scatter(scatter), 3.0 + 1e-8)
        self.assertAlmostEqual(
            baselines.effective_rank_from_scatter(scatter),
            baselines.effective_rank(matrix),
            places=8,
        )

    def test_svd_nonconvergence_uses_deterministic_eigen_fallback(self):
        rng = np.random.default_rng(31)
        matrix = rng.normal(size=(5, 8))
        expected = np.linalg.svd(matrix, compute_uv=False)
        with mock.patch.object(
            baselines.np.linalg, "svd", side_effect=np.linalg.LinAlgError("forced"),
        ):
            singular, vh = baselines.stable_singular_values_and_vh(matrix)
            ranks, nuclear, _, subspaces = baselines.pool_statistics(
                {"matrix": matrix}, top_k=3,
            )
        np.testing.assert_allclose(singular, expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            vh.T @ np.diag(singular ** 2) @ vh,
            matrix.T @ matrix,
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertTrue(np.isfinite(ranks["matrix"]))
        self.assertTrue(np.isfinite(nuclear["matrix"]))
        self.assertEqual(subspaces["matrix"].shape, (3, matrix.shape[1]))

    def test_full_rank_gram_sketch_matches_exact_greedy(self):
        sketches = baselines.pool_gram_sketches(self.features, rank=6)
        selected = baselines.rank_l_gram_greedy(sketches, 3)
        self.assertEqual(selected, baselines.full_merged_rank_greedy(self.features, 3))
        for name, sketch in sketches.items():
            matrix = self.features[name].astype(np.float64)
            np.testing.assert_allclose(
                sketch.factor.T @ sketch.factor,
                matrix.T @ matrix,
                atol=1e-10,
            )

    def test_rank_l_ties_use_the_common_lexicographic_rule(self):
        identical = {
            "b": np.eye(3, dtype=np.float64),
            "a": np.eye(3, dtype=np.float64),
            "c": np.eye(3, dtype=np.float64),
        }
        sketches = baselines.pool_gram_sketches(identical, rank=3)
        self.assertEqual(
            baselines.rank_l_gram_greedy(sketches, 3),
            ["a", "b", "c"],
        )

    def test_truncated_gram_interval_contains_exact_log_rank(self):
        sketches = baselines.pool_gram_sketches(self.features, rank=2)
        selected = ["a", "c", "e"]
        statistics = baselines.gram_sketch_statistics(sketches[name] for name in selected)
        exact = np.log(baselines.effective_rank(np.concatenate([
            self.features[name] for name in selected
        ])))
        self.assertLessEqual(float(statistics["log_rank_lower"]), exact + 1e-10)
        self.assertGreaterEqual(float(statistics["log_rank_upper"]), exact - 1e-10)

    def test_roundoff_scale_sketch_modes_are_added_to_the_tail_bound(self):
        matrix = np.diag([1.0, 1e-18]).astype(np.float64)
        sketch = baselines.make_gram_sketch(matrix, rank=2)
        statistics = baselines.gram_sketch_statistics([sketch])
        self.assertGreater(
            float(statistics["numerical_tail_nuclear_bound"]), 0.0,
        )
        self.assertEqual(statistics["numerical_tail_rank_bound"], 1)
        self.assertGreaterEqual(
            float(statistics["tail_nuclear_bound"]),
            float(statistics["numerical_tail_nuclear_bound"]),
        )

    def test_similarity_matrices_are_symmetric_with_unit_diagonal(self):
        for representation in baselines.REPRESENTATIONS:
            matrix = baselines.similarity_matrix(
                self.features, self.names, representation, self.centroids, self.subspaces,
            )
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-8)
            np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-8)
            self.assertTrue(np.isfinite(matrix).all())

    def test_all_selectors_return_unique_valid_pools(self):
        similarity = baselines.similarity_matrix(
            self.features, self.names, "centroid", self.centroids, self.subspaces,
        )
        selections = [
            baselines.rank_only(self.ranks, 3),
            baselines.rank_l_gram_greedy(
                baselines.pool_gram_sketches(self.features, rank=3), 3,
            ),
            baselines.full_merged_rank_greedy(self.features, 3),
            baselines.facility_location(similarity, self.names, 3),
            baselines.k_center(similarity, self.names, self.ranks, 3),
            baselines.k_medoids_pam(similarity, self.names, 3),
            baselines.agglomerative_medoids(similarity, self.names, 3),
            baselines.leverage_score_selection(
                np.stack([self.centroids[name] for name in self.names]),
                self.names,
                3,
            ),
            baselines.dpp_greedy(similarity, self.names, self.ranks, 3),
            baselines.pool_vendi_greedy(similarity, self.names, self.ranks, 3),
            baselines.merged_vendi_greedy(self.features, 3),
        ]
        for selected in selections:
            baselines.validate_selection(selected, self.names, 3)

    def test_full_rank_greedy_maximizes_each_next_step(self):
        selected = baselines.full_merged_rank_greedy(self.features, 3)
        prefix = []
        for choice in selected:
            remaining = [name for name in self.names if name not in prefix]
            scores = {
                name: baselines.effective_rank(
                    np.concatenate([self.features[item] for item in prefix + [name]], axis=0)
                )
                for name in remaining
            }
            expected = max(sorted(remaining), key=lambda name: scores[name])
            self.assertEqual(choice, expected)
            prefix.append(choice)

    def test_exact_greedy_is_not_mislabeled_as_global_oracle(self):
        rng = np.random.default_rng(2)
        features = {
            chr(ord("a") + index): rng.normal(size=(4, 5))
            for index in range(5)
        }
        greedy = baselines.exact_merged_rank_greedy(features, 3)
        oracle = baselines.exhaustive_merged_rank_oracle(features, 3)
        greedy_score = baselines.effective_rank(np.concatenate([
            features[name] for name in greedy
        ]))
        oracle_score = baselines.effective_rank(np.concatenate([
            features[name] for name in oracle
        ]))
        self.assertGreater(oracle_score, greedy_score)
        with self.assertRaises(ValueError):
            baselines.exhaustive_merged_rank_oracle(
                features, 3, max_combinations=9,
            )

    def test_pam_result_is_one_swap_local_optimum(self):
        similarity = baselines.similarity_matrix(
            self.features, self.names, "subspace", self.centroids, self.subspaces,
        )
        selected_names = baselines.k_medoids_pam(similarity, self.names, 2)
        selected = [self.names.index(name) for name in selected_names]
        distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
        current = baselines.medoid_cost(distance, selected)
        for position in range(len(selected)):
            for candidate in set(range(len(self.names))) - set(selected):
                proposal = selected.copy()
                proposal[position] = candidate
                self.assertGreaterEqual(
                    baselines.medoid_cost(distance, proposal), current - 1e-10,
                )


if __name__ == "__main__":
    unittest.main()
