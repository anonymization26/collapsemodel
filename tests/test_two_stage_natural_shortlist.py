from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import two_stage_classic_baselines as classic  # noqa: E402
import two_stage_natural_shortlist as natural  # noqa: E402


PROVENANCE = {
    "checkpoint": "test-checkpoint",
    "checkpoint_file_sha256": None,
    "model_state_sha256": "a" * 64,
    "preprocess_sha256": "b" * 64,
    "encoder_loader_sha256": "c" * 64,
}


def write_feature_cache(
    path: Path,
    matrix: np.ndarray,
    *,
    variant: str = "encoder",
    dataset: str = "dataset",
    split: str = "train",
    sampling: str = "unlabeled_random",
    reads_labels: bool = False,
    provenance: dict[str, object] | None = None,
) -> None:
    indices = np.arange(len(matrix), dtype=np.int64)
    labels = np.arange(len(matrix), dtype=np.int64) % 2
    np.savez(path, H=matrix, y=labels, indices=indices)
    path.with_suffix(".json").write_text(json.dumps({
        "variant": variant,
        "dataset": dataset,
        "split": split,
        "sampling": sampling,
        "stage1_sampling_reads_labels": reads_labels,
        "requested_samples": len(matrix),
        "source_samples": len(matrix),
        "feature_shape": list(matrix.shape),
        "feature_file_sha256": natural.sha256_file(path),
        **(PROVENANCE if provenance is None else provenance),
    }))


class NaturalShortlistTests(unittest.TestCase):
    def test_stage1_loader_accepts_feature_only_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            matrix = np.arange(24, dtype=np.float32).reshape(6, 4)
            np.savez(path, H=matrix)
            loaded = natural.load_features(path)
            self.assertEqual(loaded.shape, matrix.shape)
            np.testing.assert_allclose(np.linalg.norm(loaded, axis=1)[1:], 1.0)

    def test_stage1_cache_gate_rejects_label_informed_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            write_feature_cache(
                path,
                np.eye(3, dtype=np.float32),
                sampling="deterministic_proportional_stratified",
                reads_labels=True,
            )
            with self.assertRaises(ValueError):
                natural.stage1_cache_metadata(path, allow_label_informed=False)
            metadata = natural.stage1_cache_metadata(path, allow_label_informed=True)
            self.assertFalse(metadata["label_independent_sampling"])
            write_feature_cache(path, np.eye(3, dtype=np.float32))
            metadata = natural.stage1_cache_metadata(path, allow_label_informed=False)
            self.assertTrue(metadata["label_independent_sampling"])
            invalid = json.loads(path.with_suffix(".json").read_text())
            invalid["feature_file_sha256"] = "0" * 64
            path.with_suffix(".json").write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                natural.stage1_cache_metadata(path, allow_label_informed=True)

    def test_stage2_cache_gate_rejects_label_selected_target_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.npz"
            matrix = np.eye(4, dtype=np.float32)
            write_feature_cache(
                path, matrix, dataset="target", split="test",
            )
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            metadata["stage2_sampling_reads_labels"] = False
            metadata_path.write_text(json.dumps(metadata))
            audited = natural.stage2_cache_metadata(
                path,
                expected_encoder="encoder",
                expected_dataset="target",
                expected_split="test",
                expected_samples=4,
                label_read_field="stage2_sampling_reads_labels",
            )
            self.assertEqual(audited["encoder_provenance"], PROVENANCE)
            metadata["sampling"] = "stratified"
            metadata["stage2_sampling_reads_labels"] = True
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "not label-independent"):
                natural.stage2_cache_metadata(
                    path,
                    expected_encoder="encoder",
                    expected_dataset="target",
                    expected_split="test",
                    expected_samples=4,
                    label_read_field="stage2_sampling_reads_labels",
                )

    def test_screen_manifest_binds_dependencies_and_records_rank_l_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "sources"
            output_dir = root / "output"
            source_dir.mkdir()
            sources = ["cifar10", "mnist", "pathmnist"]
            rng = np.random.default_rng(71)
            for source in sources:
                path = natural.source_path(source_dir, "encoder", source, 12)
                write_feature_cache(
                    path,
                    rng.normal(size=(12, 4)).astype(np.float32),
                    dataset=source,
                )
            args = argparse.Namespace(
                out_dir=output_dir,
                source_dir=source_dir,
                encoder="encoder",
                sources=sources,
                cache_samples=12,
                stage1_samples=10,
                sample_seed=3,
                source_weighting="equal_source",
                top_k=2,
                shortlist_sizes=[1, 2],
                allow_label_informed_cache=False,
                exclude_source_domains=[],
            )
            natural.run_screen(args)
            manifest = json.loads(
                (output_dir / "encoder_screening_manifest.json").read_text()
            )
            self.assertEqual(len(manifest["source_cache_metadata_sha256"]), 3)
            self.assertEqual(manifest["encoder_provenance"], PROVENANCE)
            self.assertEqual(len(manifest["source_cache_index_sha256"]), 3)
            self.assertEqual(
                manifest["classic_script_sha256"],
                natural.sha256_file(ROOT / "scripts" / "two_stage_classic_baselines.py"),
            )
            self.assertEqual(len(manifest["rank_l_diagnostics"]), 2)
            self.assertTrue(all(
                item["selected_prefix_interval_covers_exact"]
                for item in manifest["rank_l_diagnostics"]
            ))

    def test_resume_rejects_partial_or_out_of_run_groups(self):
        targets = {"a", "b"}
        natural.validate_resume_groups(
            {("source", 0, "a"), ("source", 0, "b")},
            ["source"], range(1), targets,
        )
        with self.assertRaises(ValueError):
            natural.validate_resume_groups(
                {("source", 0, "a")}, ["source"], range(1), targets,
            )
        with self.assertRaises(ValueError):
            natural.validate_resume_groups(
                {("other", 0, "a")}, ["source"], range(1), targets,
            )
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

    def test_paired_bootstrap_confidence_tier_excludes_clear_loss(self):
        values = {
            "winner": [(0.90 + offset, seed) for seed, offset in enumerate([
                0.00, 0.01, -0.01, 0.005, -0.005,
            ])],
            "tied": [(0.90 + offset, seed) for seed, offset in enumerate([
                0.00, 0.01, -0.01, 0.005, -0.005,
            ])],
            "bad": [(0.50 + offset, seed) for seed, offset in enumerate([
                0.00, 0.01, -0.01, 0.005, -0.005,
            ])],
        }
        tier = natural.paired_bootstrap_confidence_tier(
            ["winner", "tied", "bad"], values, confidence=0.95,
            replicates=2_000, seed=7,
        )
        self.assertEqual(tier, {"winner", "tied"})

    def test_summary_validates_inputs_and_tracks_family_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "encoder": "encoder",
                "encoder_provenance": PROVENANCE,
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
            ]
            configuration = {
                "schema_version": 4,
                "script_sha256": natural.sha256_file(
                    ROOT / "scripts" / "two_stage_natural_shortlist.py"
                ),
                "screening_manifest_sha256": natural.sha256_file(manifest_path),
                "encoder": "encoder",
                "encoder_provenance": PROVENANCE,
                "assigned_sources": manifest["sources"],
                "targets": manifest["targets"],
                "source_cache_sha256": {source: "d" * 64 for source, _ in rows},
                "target_cache_sha256": {
                    "target:train": "e" * 64,
                },
                "source_cache_metadata_sha256": {
                    source: "f" * 64 for source, _ in rows
                },
                "target_cache_metadata_sha256": {
                    "target:train": "1" * 64,
                },
                "source_cache_index_sha256": {
                    source: "2" * 64 for source, _ in rows
                },
                "target_cache_index_sha256": {
                    "target:train": "3" * 64,
                },
                "target_test_access": "forbidden_during_adaptation_and_selection",
                "cache_samples": 10,
                "adapter_samples": 10,
                "allow_short_source": False,
                "source_sample_seed": 1,
                "target_cv_folds": 5,
                "target_cv_seeds": [1],
                "seed_start": 0,
                "n_seeds": 1,
                "width": 4,
                "steps": 1,
                "batch_size": 2,
                "ridge": 0.01,
                "adapter_state_format": "torch_state_dict_v1",
                "deterministic_algorithms": True,
                "shard_index": 0,
                "shard_count": 1,
            }
            canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            state_dir = root / "adapter_states"
            state_dir.mkdir()
            with adaptation_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "run_fingerprint", "encoder", "source", "adapter_seed", "target",
                    "validation_accuracy", "validation_accuracy_std",
                    "validation_fold_accuracies", "validation_partition_accuracies",
                    "validation_accuracy_std_across_partitions", "validation_examples",
                    "unique_validation_examples", "target_validation_protocol",
                    "target_cv_folds", "target_cv_partitions", "target_cv_seed",
                    "source_samples", "steps", "width", "batch_size",
                    "ridge", "deterministic_algorithms", "adapter_state_path",
                    "adapter_state_file_sha256", "adapter_state_sha256",
                    "training_seconds", "evaluation_seconds",
                ])
                writer.writeheader()
                for source, accuracy in rows:
                    state_path = state_dir / f"encoder__{source}__seed0.pt"
                    state_path.write_bytes(source.encode())
                    writer.writerow({
                        "run_fingerprint": fingerprint,
                        "encoder": "encoder",
                        "source": source,
                        "adapter_seed": 0,
                        "target": "target",
                        "validation_accuracy": accuracy,
                        "validation_accuracy_std": 0.0,
                        "validation_fold_accuracies": "|".join([str(accuracy)] * 5),
                        "validation_partition_accuracies": str(accuracy),
                        "validation_accuracy_std_across_partitions": 0.0,
                        "validation_examples": 10,
                        "unique_validation_examples": 10,
                        "target_validation_protocol": "stratified_5_fold",
                        "target_cv_folds": 5,
                        "target_cv_partitions": 1,
                        "target_cv_seed": 1,
                        "source_samples": 10,
                        "steps": 1,
                        "width": 4,
                        "batch_size": 2,
                        "ridge": 0.01,
                        "deterministic_algorithms": True,
                        "adapter_state_path": f"adapter_states/{state_path.name}",
                        "adapter_state_file_sha256": natural.sha256_file(state_path),
                        "adapter_state_sha256": "4" * 64,
                        "training_seconds": 0.1,
                        "evaluation_seconds": 0.1,
                    })
            natural.adaptation_run_path(adaptation_path).write_text(json.dumps({
                "run_fingerprint": fingerprint,
                "configuration": configuration,
            }))
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
            selection_manifest = root / "encoder_shortlist_manifest.json"
            frozen_manifest, frozen_detail, frozen_rows = natural.validate_selection_freeze(
                selection_manifest,
            )
            self.assertEqual(frozen_manifest["schema_version"], 2)
            self.assertEqual(frozen_detail, root / "encoder_shortlist_results.csv")
            self.assertEqual(len(frozen_rows), 1)
            self.assertFalse(frozen_manifest["target_test_read_during_adaptation_or_selection"])
            self.assertEqual(result["oracle_source"], "cifar100")
            self.assertEqual(result["best_candidate_recall"], "0")
            self.assertEqual(result["best_candidate_family_recall"], "1")
            self.assertEqual(result["candidate_count"], "3")
            self.assertIn("best_candidate_recall_tol_0p001", result)

    def test_frozen_adapter_state_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "adaptation.csv"
            adapter = natural.Adapter(4, 3)
            relative, file_hash, state_hash = natural.save_adapter_state(
                adapter, output, "encoder", "source", 7,
            )
            loaded = natural.load_frozen_adapter(
                root / relative,
                file_hash,
                state_hash,
                dimension=4,
                width=3,
                device=natural.torch.device("cpu"),
            )
            self.assertEqual(natural.module_state_sha256(loaded), state_hash)

    def test_heldout_resume_binds_metrics_and_frozen_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "adapter.pt"
            checkpoint.write_bytes(b"frozen-state")
            state = {
                "path": checkpoint.name,
                "file_sha256": natural.sha256_file(checkpoint),
                "state_sha256": "a" * 64,
            }
            row = {
                "run_fingerprint": "fingerprint",
                "encoder": "encoder",
                "source": "source",
                "adapter_seed": "3",
                "target": "target",
                "test_accuracy": "0.75",
                "adapter_state_path": checkpoint.name,
                "adapter_state_file_sha256": state["file_sha256"],
                "adapter_state_sha256": state["state_sha256"],
                "evaluation_seconds": "0.1",
                "device": "npu:4",
            }
            expected = {("source", 3, "target")}
            completed = natural.validate_heldout_evaluation_resume(
                [row], expected, "fingerprint", "encoder",
                {"source:3": state}, root,
            )
            self.assertEqual(completed, expected)

            invalid_metric = {**row, "test_accuracy": "1.2"}
            with self.assertRaisesRegex(ValueError, "above 1.0"):
                natural.validate_heldout_evaluation_resume(
                    [invalid_metric], expected, "fingerprint", "encoder",
                    {"source:3": state}, root,
                )

            invalid_state = {**row, "adapter_state_sha256": "b" * 64}
            with self.assertRaisesRegex(ValueError, "frozen adapter state"):
                natural.validate_heldout_evaluation_resume(
                    [invalid_state], expected, "fingerprint", "encoder",
                    {"source:3": state}, root,
                )


if __name__ == "__main__":
    unittest.main()
