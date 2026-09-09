import importlib.util
import argparse
import json
import sys
import tempfile
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

    def test_zero_overlap_alias_has_independent_lineage(self):
        _, pool_indices, _, lineage, parents = controlled.make_collection(
            self.features, self.indices, pool_size=10, pools_per_dataset=3,
            overlap=0.0, seed=29,
        )
        for alias, parent in parents.items():
            self.assertEqual(
                np.intersect1d(pool_indices[alias], pool_indices[parent]).size,
                0,
            )
            self.assertEqual(lineage[alias], alias)
            self.assertNotEqual(lineage[alias], lineage[parent])

    def test_binary_metrics_are_exact_for_perfect_ranking(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.2, 0.8, 0.1], dtype=np.float64)
        self.assertEqual(controlled.binary_auc(labels, scores), 1.0)
        self.assertEqual(controlled.average_precision(labels, scores), 1.0)

    def test_average_precision_is_invariant_to_tie_order(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.ones(4, dtype=np.float64)
        self.assertEqual(controlled.average_precision(labels, scores), 0.5)
        self.assertEqual(
            controlled.average_precision(labels[::-1], scores), 0.5,
        )

    def test_feature_loader_enforces_unlabeled_cache_and_records_index_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "resnet50__cifar10__train__n4"
            archive = root / f"{stem}.npz"
            metadata_path = root / f"{stem}.json"
            features = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
            indices = np.asarray([7, 2, 11, 5], dtype=np.int64)
            np.savez(archive, H=features, indices=indices)
            metadata = {
                "variant": "resnet50",
                "dataset": "cifar10",
                "split": "train",
                "sampling": "unlabeled_random",
                "stage1_sampling_reads_labels": False,
                "source_samples": 10,
                "requested_samples": 4,
                "sample_seed": 17,
                "source_file_sha256": ["source-sha"],
                "checkpoint": "checkpoint",
                "checkpoint_file_sha256": None,
                "model_state_sha256": "a" * 64,
                "preprocess_sha256": "b" * 64,
                "encoder_loader_sha256": "c" * 64,
                "feature_file_sha256": controlled.sha256_file(archive),
                "feature_shape": [4, 3],
            }
            metadata_path.write_text(json.dumps(metadata))

            loaded, loaded_indices, loaded_metadata = controlled.load_feature_set(
                root, "resnet50", "cifar10", "4",
            )
            np.testing.assert_array_equal(loaded_indices, indices)
            np.testing.assert_allclose(np.linalg.norm(loaded, axis=1), 1.0)
            self.assertEqual(len(loaded_metadata["computed_index_sha256"]), 64)

            metadata["sampling"] = "deterministic_proportional_stratified"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "not label-independent"):
                controlled.load_feature_set(root, "resnet50", "cifar10", "4")

    def test_parent_detection_uses_only_matched_same_dataset_negatives(self):
        names = [
            "a__alias", "a__pool0", "a__pool1",
            "b__alias", "b__pool0", "b__pool1",
        ]
        similarity = np.eye(len(names))
        similarity[0, 1] = similarity[1, 0] = 0.9
        similarity[0, 2] = similarity[2, 0] = 0.2
        similarity[3, 4] = similarity[4, 3] = 0.8
        similarity[3, 5] = similarity[5, 3] = 0.1
        families = {name: name.split("__")[0] for name in names}
        metrics = controlled.matched_parent_detection(
            similarity,
            names,
            families,
            {"a__alias": "a__pool0", "b__alias": "b__pool0"},
        )
        self.assertEqual(metrics["parent_matched_auroc"], 1.0)
        self.assertEqual(metrics["parent_matched_auprc"], 1.0)
        self.assertEqual(metrics["parent_top1_accuracy"], 1.0)
        self.assertEqual(metrics["parent_mean_reciprocal_rank"], 1.0)

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
        summaries, *_ = summary_only.complete_pool_summaries(pools, top_k=3)
        selected = controlled.classic.rank_l_gram_greedy(summaries, 5)
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

    def test_rank_only_uses_exact_source_local_scalar_not_truncated_rank(self):
        pools = {
            "pool": np.diag([4.0, 3.0, 2.0, 1.0]).astype(np.float32),
        }
        summaries, ranks, nuclear, subspaces, sketches = (
            summary_only.complete_pool_summaries(pools, top_k=1)
        )
        expected_rank = controlled.classic.effective_rank(pools["pool"])
        self.assertAlmostEqual(ranks["pool"], expected_rank)
        self.assertGreater(ranks["pool"], 1.0)
        self.assertAlmostEqual(nuclear["pool"], 10.0)
        self.assertEqual(subspaces["pool"].shape, (1, 4))
        self.assertEqual(sketches["pool"].shape, (1, 4))
        self.assertAlmostEqual(
            summaries["pool"].marginal_effective_rank, expected_rank,
        )

    def test_summary_byte_accounting_includes_transmitted_scalars(self):
        sizes = summary_only.summary_bytes(dimension=7, top_k=3)
        self.assertEqual(sizes["rank_only"], 8)
        self.assertEqual(sizes["rank_l_gram"], 8 * (3 * 7 + 3))
        self.assertEqual(sizes["collapse_sketch"], 8 * (3 * 7 + 2))
        self.assertEqual(sizes["dpp_subspace"], 8 * (1 + 3 * 7))
        self.assertEqual(sizes["facility_subspace"], 8 * 3 * 7)
        self.assertEqual(sizes["agglomerative_subspace"], 8 * 3 * 7)
        self.assertEqual(sizes["lineage_deduplicated_rank"], 8)

    def test_summary_run_reports_rank_l_and_matched_protocol(self):
        args = argparse.Namespace(
            pool_size=10,
            pools_per_dataset=3,
            top_k=3,
            budgets=[3],
            encoder="test",
            include_full_rank=True,
            n_random=2,
        )
        zero_rows = summary_only.run_collection(
            args, self.features, self.indices, seed=31, overlap=0.0,
        )
        self.assertIn("rank_l_gram", {row["method"] for row in zero_rows})
        self.assertIn("exact_merged_rank_greedy", {row["method"] for row in zero_rows})
        self.assertEqual(
            {row["replicate"] for row in zero_rows if row["method"] == "random"},
            {0, 1},
        )
        self.assertTrue(all(
            row["rank_l_selected_interval_covers_exact"] for row in zero_rows
        ))
        self.assertTrue(all(
            isinstance(row["rank_l_matches_exact_greedy_prefix"], bool)
            for row in zero_rows
        ))
        self.assertTrue(all(
            np.isnan(float(row["alignment_parent_matched_auroc"]))
            for row in zero_rows
        ))
        overlap_rows = summary_only.run_collection(
            args, self.features, self.indices, seed=31, overlap=0.5,
        )
        self.assertTrue(all(
            np.isfinite(float(row["alignment_parent_matched_auroc"]))
            for row in overlap_rows
        ))


if __name__ == "__main__":
    unittest.main()
