import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_two_stage_cross_encoder as summary  # noqa: E402


def row(encoder, target, method, recall, regret, shortlist):
    return {
        "encoder": encoder,
        "target": target,
        "method": method,
        "replicate": "-1",
        "shortlist_size": "3",
        "best_candidate_recall": str(recall),
        "top3_candidate_recall": str(recall),
        "validation_regret": str(regret),
        "normalized_validation_regret": str(regret),
        "test_regret_vs_exhaustive_validation_selection": str(regret),
        "removed_fraction": "0.75",
        "shortlist": shortlist,
        "oracle_source_family": f"family_{target}",
        "best_candidate_family_recall": str(recall),
        "top3_candidate_family_recall": str(recall),
        "selected_family_count": "3",
        "removed_family_fraction": "0.75",
    }


class CrossEncoderSummaryTests(unittest.TestCase):
    def test_detail_reader_requires_complete_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            root.mkdir()
            path = root / "e1_shortlist_results.csv"
            rows = [row("e1", "t1", "rank_only", 1, 0, "a|b|c")]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            manifest_path = root / "e1_shortlist_manifest.json"
            manifest_path.write_text(json.dumps({
                "encoder": "e1",
                "targets": ["t1"],
                "shortlist_sizes": [3],
                "methods": ["rank_only"],
                "n_random": 0,
                "target_test_used_for_selection": False,
                "detail_rows": 1,
                "detail_sha256": summary.sha256_file(path),
            }))
            read_rows, inputs = summary.read_detail_rows(
                Path(temporary), expected_encoders=["e1"],
            )
            self.assertEqual(read_rows, rows)
            self.assertIn(manifest_path, inputs)
            manifest = json.loads(manifest_path.read_text())
            manifest["detail_rows"] = 2
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "row count"):
                summary.read_detail_rows(Path(temporary), expected_encoders=["e1"])

    def test_heldout_reader_binds_immutable_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            root.mkdir()
            selection_path = root / "e1_shortlist_results.csv"
            selection_row = row("e1", "t1", "rank_only", 1, 0, "a|b|c")
            selection_row.pop("test_regret_vs_exhaustive_validation_selection")
            selection_row.update({
                "selected_source": "a",
                "oracle_source": "a",
            })
            with selection_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selection_row))
                writer.writeheader()
                writer.writerow(selection_row)
            selection_manifest_path = root / "e1_shortlist_manifest.json"
            adapter_state = {
                "path": "adapter_states/a.pt",
                "file_sha256": "a" * 64,
                "state_sha256": "b" * 64,
            }
            protocol = {
                "seed_start": 0,
                "n_seeds": 1,
                "width": 4,
                "ridge": 0.01,
                "deterministic_algorithms": True,
                "target_cache_sha256": {"t1:train": "c" * 64},
                "target_cache_metadata_sha256": {"t1:train": "d" * 64},
                "target_cache_index_sha256": {"t1:train": "e" * 64},
            }
            script_sha256 = "f" * 64
            selection_manifest_path.write_text(json.dumps({
                "schema_version": 2,
                "script_sha256": script_sha256,
                "encoder": "e1",
                "targets": ["t1"],
                "sources": ["a", "b", "c", "d"],
                "shortlist_sizes": [3],
                "methods": ["rank_only"],
                "n_random": 0,
                "target_test_used_for_selection": False,
                "target_test_read_during_adaptation_or_selection": False,
                "selection_frozen_before_target_test": True,
                "selection_detail": selection_path.name,
                "detail_rows": 1,
                "detail_sha256": summary.sha256_file(selection_path),
                "adaptation_protocol": protocol,
                "adapter_states": {"a:0": adapter_state},
            }))
            heldout_path = root / "e1_heldout_results.csv"
            heldout_row = {
                **selection_row,
                "oracle_selected_test_accuracy": "0.8",
                "selected_test_accuracy": "0.8",
                "test_regret_vs_exhaustive_validation_selection": "0.0",
                "heldout_adapter_seeds": "1",
            }
            with heldout_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(heldout_row))
                writer.writeheader()
                writer.writerow(heldout_row)
            baseline_path = root / "e1_heldout_stage2_baselines.csv"
            configuration = {
                "script_sha256": script_sha256,
                "selection_manifest_sha256": summary.sha256_file(selection_manifest_path),
                "selection_detail_sha256": summary.sha256_file(selection_path),
                "encoder": "e1",
                "targets": ["t1"],
                "evaluation_pairs": [["a", "t1"]],
                "adapter_states": {"a:0": adapter_state},
                "seed_start": 0,
                "n_seeds": 1,
                "width": 4,
                "ridge": 0.01,
                "deterministic_algorithms": True,
                "target_train_cache_sha256": {"t1:train": "c" * 64},
                "target_train_metadata_sha256": {"t1:train": "d" * 64},
                "target_train_index_sha256": {"t1:train": "e" * 64},
                "target_test_cache_sha256": {"t1:test": "1" * 64},
                "target_test_access": "after_selection_manifest_was_frozen",
            }
            canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            baseline_path.write_text(
                "run_fingerprint,encoder,baseline,adapter_seed,target,test_accuracy,"
                "evaluation_seconds,device\n"
                f"{fingerprint},e1,frozen_identity,-1,t1,0.7,0.1,npu:4\n"
                f"{fingerprint},e1,random_adapter,0,t1,0.6,0.1,npu:4\n"
            )
            raw_path = root / "e1_heldout_results_evaluations.csv"
            raw_path.write_text(
                "run_fingerprint,encoder,source,adapter_seed,target,test_accuracy,"
                "adapter_state_path,adapter_state_file_sha256,adapter_state_sha256,"
                "evaluation_seconds,device\n"
                f"{fingerprint},e1,a,0,t1,0.8,{adapter_state['path']},"
                f"{adapter_state['file_sha256']},{adapter_state['state_sha256']},"
                "0.1,npu:4\n"
            )
            sidecar_path = root / "e1_heldout_results.csv.run.json"
            sidecar_path.write_text(json.dumps({
                "run_fingerprint": fingerprint,
                "configuration": configuration,
            }))
            heldout_manifest_path = root / "e1_heldout_results_manifest.json"
            heldout_manifest_path.write_text(json.dumps({
                "encoder": "e1",
                "script_sha256": script_sha256,
                "heldout_results_sha256": summary.sha256_file(heldout_path),
                "heldout_result_rows": 1,
                "target_test_access": "after_selection_manifest_was_frozen",
                "target_test_used_for_selection": False,
                "heldout_stage2_baselines": baseline_path.name,
                "heldout_stage2_baselines_sha256": summary.sha256_file(baseline_path),
                "selection_manifest": selection_manifest_path.name,
                "selection_manifest_sha256": summary.sha256_file(selection_manifest_path),
                "selection_detail_sha256": summary.sha256_file(selection_path),
                "raw_evaluations": raw_path.name,
                "raw_evaluation_rows": 1,
                "raw_evaluations_sha256": summary.sha256_file(raw_path),
                "heldout_run_sidecar": sidecar_path.name,
                "heldout_run_sidecar_sha256": summary.sha256_file(sidecar_path),
                "test_evaluation_keys": 1,
            }))
            rows, inputs = summary.read_detail_rows(
                Path(temporary), expected_encoders=["e1"],
            )
            self.assertEqual(rows, [heldout_row])
            self.assertIn(heldout_manifest_path, inputs)
            controls, control_paths = summary.stage2_control_pairs(
                Path(temporary), rows,
            )
            self.assertIn(baseline_path, control_paths)
            self.assertAlmostEqual(controls[0]["oracle_minus_frozen_identity"], 0.1)
            self.assertAlmostEqual(controls[0]["oracle_minus_random_adapter"], 0.2)
            control_summary, bootstrap = summary.summarize_stage2_controls(
                controls, replicates=100, seed=7,
            )
            self.assertAlmostEqual(
                control_summary[0]["validation_oracle_source_adapter_test_accuracy_mean"],
                0.8,
            )
            self.assertEqual(len(bootstrap), 4)
            selected = summary.selected_vs_frozen_identity(rows, controls)
            self.assertAlmostEqual(selected[0]["selected_minus_frozen_identity_mean"], 0.1)
            selection_row["selected_source"] = "b"
            with selection_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selection_row))
                writer.writeheader()
                writer.writerow(selection_row)
            with self.assertRaisesRegex(ValueError, "selection freeze checksum"):
                summary.read_detail_rows(Path(temporary), expected_encoders=["e1"])

    def test_heldout_reader_rejects_a_substituted_raw_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "e1"
            root.mkdir()
            selection_path = root / "e1_shortlist_results.csv"
            selection_row = row("e1", "t1", "rank_only", 1, 0, "a|b|c")
            selection_row.pop("test_regret_vs_exhaustive_validation_selection")
            selection_row.update({"selected_source": "a", "oracle_source": "a"})
            with selection_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(selection_row))
                writer.writeheader()
                writer.writerow(selection_row)
            adapter_state = {
                "path": "adapter_states/a.pt",
                "file_sha256": "a" * 64,
                "state_sha256": "b" * 64,
            }
            protocol = {
                "seed_start": 0,
                "n_seeds": 1,
                "width": 4,
                "ridge": 0.01,
                "deterministic_algorithms": True,
                "target_cache_sha256": {"t1:train": "c" * 64},
                "target_cache_metadata_sha256": {"t1:train": "d" * 64},
                "target_cache_index_sha256": {"t1:train": "e" * 64},
            }
            script_sha256 = "f" * 64
            selection_manifest_path = root / "e1_shortlist_manifest.json"
            selection_manifest_path.write_text(json.dumps({
                "schema_version": 2,
                "script_sha256": script_sha256,
                "encoder": "e1",
                "targets": ["t1"],
                "sources": ["a", "b", "c", "d"],
                "shortlist_sizes": [3],
                "methods": ["rank_only"],
                "n_random": 0,
                "target_test_used_for_selection": False,
                "target_test_read_during_adaptation_or_selection": False,
                "selection_frozen_before_target_test": True,
                "selection_detail": selection_path.name,
                "detail_rows": 1,
                "detail_sha256": summary.sha256_file(selection_path),
                "adaptation_protocol": protocol,
                "adapter_states": {"a:0": adapter_state},
            }))
            heldout_path = root / "e1_heldout_results.csv"
            heldout_row = {
                **selection_row,
                "oracle_selected_test_accuracy": "0.8",
                "selected_test_accuracy": "0.8",
                "test_regret_vs_exhaustive_validation_selection": "0.0",
                "heldout_adapter_seeds": "1",
            }
            with heldout_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(heldout_row))
                writer.writeheader()
                writer.writerow(heldout_row)
            configuration = {
                "script_sha256": script_sha256,
                "selection_manifest_sha256": summary.sha256_file(selection_manifest_path),
                "selection_detail_sha256": summary.sha256_file(selection_path),
                "encoder": "e1",
                "targets": ["t1"],
                "evaluation_pairs": [["a", "t1"]],
                "adapter_states": {"a:0": adapter_state},
                "seed_start": 0,
                "n_seeds": 1,
                "width": 4,
                "ridge": 0.01,
                "deterministic_algorithms": True,
                "target_train_cache_sha256": {"t1:train": "c" * 64},
                "target_train_metadata_sha256": {"t1:train": "d" * 64},
                "target_train_index_sha256": {"t1:train": "e" * 64},
                "target_test_cache_sha256": {"t1:test": "1" * 64},
                "target_test_access": "after_selection_manifest_was_frozen",
            }
            canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            baseline_path = root / "e1_heldout_stage2_baselines.csv"
            baseline_path.write_text(
                "run_fingerprint,encoder,baseline,adapter_seed,target,test_accuracy,"
                "evaluation_seconds,device\n"
                f"{fingerprint},e1,frozen_identity,-1,t1,0.7,0.1,npu:4\n"
                f"{fingerprint},e1,random_adapter,0,t1,0.6,0.1,npu:4\n"
            )
            raw_path = root / "e1_heldout_results_evaluations.csv"
            raw_path.write_text(
                "run_fingerprint,encoder,source,adapter_seed,target,test_accuracy,"
                "adapter_state_path,adapter_state_file_sha256,adapter_state_sha256,"
                "evaluation_seconds,device\n"
                f"{fingerprint},e1,a,0,t2,0.8,{adapter_state['path']},"
                f"{adapter_state['file_sha256']},{adapter_state['state_sha256']},"
                "0.1,npu:4\n"
            )
            sidecar_path = root / "e1_heldout_results.csv.run.json"
            sidecar_path.write_text(json.dumps({
                "run_fingerprint": fingerprint,
                "configuration": configuration,
            }))
            heldout_manifest_path = root / "e1_heldout_results_manifest.json"
            heldout_manifest_path.write_text(json.dumps({
                "encoder": "e1",
                "script_sha256": script_sha256,
                "heldout_results_sha256": summary.sha256_file(heldout_path),
                "heldout_result_rows": 1,
                "target_test_access": "after_selection_manifest_was_frozen",
                "target_test_used_for_selection": False,
                "heldout_stage2_baselines": baseline_path.name,
                "heldout_stage2_baselines_sha256": summary.sha256_file(baseline_path),
                "selection_manifest": selection_manifest_path.name,
                "selection_manifest_sha256": summary.sha256_file(selection_manifest_path),
                "selection_detail_sha256": summary.sha256_file(selection_path),
                "raw_evaluations": raw_path.name,
                "raw_evaluation_rows": 1,
                "raw_evaluations_sha256": summary.sha256_file(raw_path),
                "heldout_run_sidecar": sidecar_path.name,
                "heldout_run_sidecar_sha256": summary.sha256_file(sidecar_path),
                "test_evaluation_keys": 1,
            }))
            with self.assertRaisesRegex(ValueError, "raw evaluations are incomplete"):
                summary.read_detail_rows(Path(temporary), expected_encoders=["e1"])

    def test_aggregate_counts_encoder_level_passes(self):
        rows = [
            row("e1", "t1", "rank_only", 1, 0, "a|b|c"),
            row("e1", "t2", "rank_only", 1, 0, "a|b|c"),
            row("e2", "t1", "rank_only", 0, 1, "a|b|c"),
            row("e2", "t2", "rank_only", 1, 0, "a|b|c"),
        ]
        aggregate = summary.aggregate_rows(rows)[0]
        self.assertEqual(aggregate["encoders"], 2)
        self.assertEqual(aggregate["encoder_target_pairs"], 4)
        self.assertEqual(aggregate["passing_encoders"], 1)
        self.assertAlmostEqual(aggregate["best_candidate_recall_mean"], 0.75)
        self.assertFalse(aggregate["passes_overall"])

    def test_paired_deltas_and_outcomes_use_correct_direction(self):
        rows = [
            row("e1", "t1", "dpp_subspace", 1, 0.1, "a|b|c"),
            row("e1", "t1", "rank_only", 0, 0.2, "a|d|e"),
        ]
        pair = summary.paired_dpp_vs_rank(rows)[0]
        self.assertEqual(pair["dpp_minus_rank_best_recall"], 1.0)
        self.assertAlmostEqual(pair["dpp_minus_rank_validation_regret"], -0.1)
        self.assertAlmostEqual(pair["shortlist_jaccard"], 0.2)
        aggregate = summary.summarize_pairs([pair])[0]
        self.assertEqual(aggregate["recall_wins"], 1)
        self.assertEqual(aggregate["regret_wins"], 1)

    def test_cluster_bootstrap_uses_target_and_source_family_clusters(self):
        rows = [
            row("e1", "t1", "dpp_subspace", 1, 0.0, "a|b|c"),
            row("e1", "t1", "rank_only", 0, 0.2, "a|d|e"),
            row("e1", "t2", "dpp_subspace", 0, 0.2, "a|b|c"),
            row("e1", "t2", "rank_only", 0, 0.2, "a|d|e"),
        ]
        pairs = summary.paired_dpp_vs_rank(rows)
        bootstrap = summary.cluster_bootstrap_summaries(pairs, replicates=100, seed=7)
        recall = next(
            value for value in bootstrap
            if value["cluster_level"] == "target"
            and value["metric"] == "dpp_best_candidate_recall"
        )
        self.assertEqual(recall["clusters"], 2)
        self.assertAlmostEqual(recall["estimate"], 0.5)
        self.assertEqual(
            {value["cluster_level"] for value in bootstrap},
            {"target", "source_family"},
        )

    def test_leave_one_encoder_out_excludes_each_encoder(self):
        rows = [
            row("e1", "t1", "rank_only", 1, 0, "a|b|c"),
            row("e2", "t1", "rank_only", 0, 1, "a|b|c"),
        ]
        output = summary.leave_one_encoder_out(rows)
        by_encoder = {value["excluded_encoder"]: value for value in output}
        self.assertAlmostEqual(by_encoder["e1"]["best_candidate_recall_mean"], 0.0)
        self.assertAlmostEqual(by_encoder["e2"]["best_candidate_recall_mean"], 1.0)

    def test_selected_identity_outcomes_are_mutually_exclusive_at_tolerance(self):
        rows = [
            {
                **row("e1", "t1", "rank_only", 1, 0, "a|b|c"),
                "selected_test_accuracy": "0.7000000000005",
            },
            {
                **row("e1", "t2", "rank_only", 1, 0, "a|b|c"),
                "selected_test_accuracy": "0.8",
            },
            {
                **row("e1", "t3", "rank_only", 1, 0, "a|b|c"),
                "selected_test_accuracy": "0.6",
            },
        ]
        controls = [
            {"encoder": "e1", "target": target, "frozen_identity_test_accuracy": 0.7}
            for target in ("t1", "t2", "t3")
        ]
        result = summary.selected_vs_frozen_identity(rows, controls)[0]
        self.assertEqual(result["selected_beats_frozen_identity_count"], 1)
        self.assertEqual(result["selected_ties_frozen_identity_count"], 1)
        self.assertEqual(result["selected_loses_to_frozen_identity_count"], 1)


if __name__ == "__main__":
    unittest.main()
