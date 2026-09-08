import sys
import unittest
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
            baselines.full_merged_rank_greedy(self.features, 3),
            baselines.facility_location(similarity, self.names, 3),
            baselines.k_center(similarity, self.names, self.ranks, 3),
            baselines.k_medoids_pam(similarity, self.names, 3),
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
