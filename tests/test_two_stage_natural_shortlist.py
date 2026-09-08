import argparse
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

import two_stage_classic_baselines as classic  # noqa: E402
import two_stage_natural_shortlist as natural  # noqa: E402


class NaturalShortlistTests(unittest.TestCase):
    def test_source_families_cover_sources_and_merge_known_duplicates(self):
        self.assertEqual(set(natural.SOURCE_FAMILIES), set(natural.SOURCES))
        self.assertEqual(len(set(natural.SOURCE_FAMILIES.values())), 17)
        self.assertEqual(
            natural.SOURCE_FAMILIES["cifar100"],
            natural.SOURCE_FAMILIES["cifar100_coarse"],
        )
        self.assertEqual(
            natural.SOURCE_FAMILIES["organamnist"],
            natural.SOURCE_FAMILIES["organcmnist"],
        )

    def test_scatter_effective_rank_extremes(self):
        self.assertAlmostEqual(
            natural.effective_rank_from_scatter(np.eye(4)), 4.0, places=6,
        )
        self.assertAlmostEqual(
            natural.effective_rank_from_scatter(np.ones((4, 4))), 1.0, places=6,
        )

    def test_scatter_greedy_matches_direct_merged_features(self):
        rng = np.random.default_rng(17)
        features = {
            name: natural.normalize_rows(rng.normal(size=(12, 8)).astype(np.float32))
            for name in ["a", "b", "c", "d", "e"]
        }
        expected = classic.full_merged_rank_greedy(features, 3)
        self.assertEqual(natural.full_rank_greedy(features, 3), expected)

    def test_target_split_is_disjoint_and_stratified(self):
        labels = np.repeat(np.arange(4), 10)
        train, validation = natural.split_train_validation(labels, seed=9)
        self.assertEqual(len(set(train) & set(validation)), 0)
        self.assertEqual(set(train) | set(validation), set(range(len(labels))))
        for label in np.unique(labels):
            self.assertGreater(np.sum(labels[train] == label), 0)
            self.assertGreater(np.sum(labels[validation] == label), 0)

    def test_stratified_kfold_covers_each_example_once(self):
        labels = np.repeat(np.arange(4), 11)
        splits = natural.stratified_kfold_indices(labels, folds=5, seed=9)
        validation = np.concatenate([fold[1] for fold in splits])
        np.testing.assert_array_equal(np.sort(validation), np.arange(len(labels)))
        for train, held_out in splits:
            self.assertEqual(len(set(train) & set(held_out)), 0)
            self.assertEqual(set(train) | set(held_out), set(range(len(labels))))
        for label in np.unique(labels):
            counts = [np.sum(labels[held_out] == label) for _, held_out in splits]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_stratified_kfold_rejects_too_few_class_examples(self):
        with self.assertRaises(ValueError):
            natural.stratified_kfold_indices(
                np.asarray([0, 0, 1, 1, 1]), folds=3, seed=9,
            )

    def test_target_validation_partitions_keep_seeds_separate(self):
        labels = np.repeat(np.arange(3), 10)
        partitions = natural.target_validation_partitions(
            labels, folds=5, seeds=[7, 11, 13],
        )
        self.assertEqual([seed for seed, _ in partitions], [7, 11, 13])
        self.assertTrue(all(len(folds) == 5 for _, folds in partitions))
        for _, folds in partitions:
            validation = np.concatenate([held_out for _, held_out in folds])
            np.testing.assert_array_equal(
                np.sort(validation), np.arange(len(labels)),
            )

    def test_stage2_sample_contains_every_class(self):
        labels = np.repeat(np.arange(20), 10)
        indices = natural.stratified_sample_indices(labels, count=50, seed=11)
        self.assertEqual(len(indices), 50)
        self.assertEqual(len(set(indices.tolist())), 50)
        self.assertEqual(set(labels[indices]), set(np.unique(labels)))
        with self.assertRaises(ValueError):
            natural.stratified_sample_indices(labels, count=10, seed=11)

    def test_target_label_mapping_uses_train_class_order(self):
        train, test = natural.remap_target_labels(
            np.asarray([10, 20, 10, 30]), np.asarray([30, 10, 20]),
        )
        np.testing.assert_array_equal(train, [0, 1, 0, 2])
        np.testing.assert_array_equal(test, [2, 0, 1])
        with self.assertRaises(ValueError):
            natural.remap_target_labels(np.asarray([0, 1]), np.asarray([0, 2]))

    def test_utility_tier_includes_boundary_ties(self):
        ranking = ["a", "b", "c", "d"]
        utility = {
            "a": {"validation": 0.9},
            "b": {"validation": 0.9},
            "c": {"validation": 0.8},
            "d": {"validation": 0.8},
        }
        self.assertEqual(natural.utility_tier(ranking, utility, 1, 1e-12), {"a", "b"})
        self.assertEqual(
            natural.utility_tier(ranking, utility, 3, 1e-12),
            {"a", "b", "c", "d"},
        )

    def test_summary_filters_manifest_sources_and_tracks_family_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "encoder": "encoder",
                "sources": ["cifar100", "cifar100_coarse", "mnist"],
                "source_families": {
                    "cifar100": "cifar100_shared_images",
                    "cifar100_coarse": "cifar100_shared_images",
                    "mnist": "handwritten_digits",
                },
                "targets": ["target"],
                "shortlist_sizes": [1],
                "selections": {
                    "1": {"rank_only": ["cifar100_coarse"]},
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            adaptation_path = root / "adaptation.csv"
            rows = [
                ("cifar100", 0.9),
                ("cifar100_coarse", 0.8),
                ("mnist", 0.7),
                ("usps", 1.0),
            ]
            with adaptation_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "encoder", "source", "adapter_seed", "target",
                    "validation_accuracy", "test_accuracy",
                ])
                writer.writeheader()
                for source, accuracy in rows:
                    writer.writerow({
                        "encoder": "encoder",
                        "source": source,
                        "adapter_seed": 0,
                        "target": "target",
                        "validation_accuracy": accuracy,
                        "test_accuracy": accuracy,
                    })
            natural.run_summarize(argparse.Namespace(
                manifest=manifest_path,
                adaptation_csv=[adaptation_path],
                out_dir=root,
                n_random=0,
                random_seed=1,
                tie_tolerance=1e-12,
            ))
            with (root / "encoder_shortlist_results.csv").open(newline="") as stream:
                result = next(csv.DictReader(stream))
            self.assertEqual(result["oracle_source"], "cifar100")
            self.assertEqual(result["best_candidate_recall"], "0")
            self.assertEqual(result["best_candidate_family_recall"], "1")
            self.assertEqual(result["candidate_count"], "3")


if __name__ == "__main__":
    unittest.main()
