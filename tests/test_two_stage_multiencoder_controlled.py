import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "two_stage_multiencoder_controlled",
    SCRIPT_DIR / "two_stage_multiencoder_controlled.py",
)
controlled = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(controlled)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "two_stage_summary_only_controlled",
    SCRIPT_DIR / "two_stage_summary_only_controlled.py",
)
summary_only = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(summary_only)


class ControlledCollectionTests(unittest.TestCase):
    def setUp(self):
        self.features = {}
        self.indices = {}
        for dataset_index, dataset in enumerate(controlled.DATASETS):
            rng = np.random.default_rng(100 + dataset_index)
            self.features[dataset] = rng.normal(size=(80, 12)).astype(np.float32)
            self.indices[dataset] = np.arange(80, dtype=np.int64) + dataset_index * 1000

    def test_collection_is_deterministic_and_has_exact_requested_overlap(self):
        first = controlled.make_collection(
            self.features, self.indices, pool_size=10, pools_per_dataset=3,
            overlap=0.5, seed=17,
        )
        second = controlled.make_collection(
            self.features, self.indices, pool_size=10, pools_per_dataset=3,
            overlap=0.5, seed=17,
        )
        pools, pool_indices, families, lineage, parents = first
        for name in pools:
            np.testing.assert_array_equal(pools[name], second[0][name])
            np.testing.assert_array_equal(pool_indices[name], second[1][name])
        self.assertEqual(families, second[2])
        self.assertEqual(lineage, second[3])
        self.assertEqual(parents, second[4])

        for alias, parent in parents.items():
            overlap = np.intersect1d(pool_indices[alias], pool_indices[parent]).size
            self.assertEqual(overlap, 5)
            self.assertEqual(lineage[alias], parent)

    def test_base_pools_are_disjoint_within_each_dataset(self):
        _, pool_indices, _, _, _ = controlled.make_collection(
            self.features, self.indices, pool_size=10, pools_per_dataset=3,
            overlap=1.0, seed=19,
        )
        for dataset in controlled.DATASETS:
            names = [f"{dataset}__pool{index}" for index in range(3)]
            for left_index, left in enumerate(names):
                for right in names[left_index + 1 :]:
                    self.assertEqual(
                        np.intersect1d(pool_indices[left], pool_indices[right]).size,
                        0,
                    )

    def test_binary_metrics_are_exact_for_perfect_ranking(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float64)
        self.assertEqual(controlled.binary_auc(labels, scores), 1.0)
        self.assertEqual(controlled.average_precision(labels, scores), 1.0)

    def test_summary_only_selector_uses_fixed_size_pool_sketches(self):
        pools, _, _, _, _ = controlled.make_collection(
            self.features, self.indices, pool_size=10, pools_per_dataset=3,
            overlap=0.5, seed=23,
        )
        ranks, nuclear, subspaces, sketches = summary_only.pool_summaries(
            pools, top_k=3,
        )
        for name in pools:
            self.assertEqual(subspaces[name].shape, (3, 12))
            self.assertEqual(sketches[name].shape, (3, 12))
        selected = summary_only.collapse_sketch_sequence(
            ranks, nuclear, subspaces, sketches, top_k=3, max_budget=5,
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(selected)), 5)

    def test_sketch_statistics_handles_rank_deficiency(self):
        sketch = np.ones((20, 12), dtype=np.float32)
        effective_rank, nuclear, subspace = summary_only.sketch_statistics(
            sketch, top_k=5,
        )
        self.assertAlmostEqual(effective_rank, 1.0, places=6)
        self.assertGreater(nuclear, 0.0)
        self.assertEqual(subspace.shape, (1, 12))
        self.assertTrue(np.isfinite(subspace).all())


if __name__ == "__main__":
    unittest.main()
