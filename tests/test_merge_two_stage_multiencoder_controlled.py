import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import merge_two_stage_multiencoder_controlled as merge  # noqa: E402


def result_row(method: str, merged_reff: float) -> dict[str, str]:
    return {
        "encoder": "encoder_a",
        "construction_seed": "7",
        "overlap": "0.5",
        "budget": "3",
        "method": method,
        "replicate": "0",
        "merged_reff": str(merged_reff),
    }


def corrected_payload() -> dict[str, object]:
    return {
        "encoder": "encoder_a",
        "construction_seeds": [7],
        "overlaps": [0.5],
        "budgets": [3],
        "include_full_rank": False,
        "n_random": 1,
        "shard_index": 0,
        "shard_count": 1,
        "assigned_collections": [[7, 0.5]],
        "feature_metadata": {"dataset": {"feature_file_sha256": "d" * 64}},
        "encoder_provenance": {
            "checkpoint": "checkpoint",
            "checkpoint_file_sha256": None,
            "model_state_sha256": "a" * 64,
            "preprocess_sha256": "b" * 64,
            "encoder_loader_sha256": "c" * 64,
        },
    }


def complete_shard_rows() -> list[dict[str, str]]:
    rows = []
    for method in sorted(merge.CONTROLLED_METHODS | {"random"}):
        row = result_row(method, 12.0)
        row["selected"] = "a|b|c"
        rows.append(row)
    return rows


class ControlledMergeTests(unittest.TestCase):
    def test_primary_method_is_not_mislabeled_as_legacy_collapse(self):
        rows = [
            result_row("rank_only", 10.0),
            result_row("rank_l_gram", 12.0),
            result_row("dpp_subspace", 11.0),
        ]
        pairwise = merge.summarize_pairwise(rows, "rank_l_gram")[0]
        decision = merge.summarize_stopping_decision(rows, "rank_l_gram")[0]
        self.assertEqual(pairwise["primary_method"], "rank_l_gram")
        self.assertEqual(pairwise["primary_minus_rank_mean"], 2.0)
        self.assertNotIn("collapse_minus_rank_mean", pairwise)
        self.assertEqual(decision["primary_method"], "rank_l_gram")
        self.assertEqual(decision["primary_win_count"], 1)

    def test_duplicate_result_key_is_rejected(self):
        row = result_row("rank_l_gram", 12.0)
        with self.assertRaisesRegex(ValueError, "duplicate result key"):
            merge.register_result_keys([row, dict(row)], set())

    def test_corrected_manifest_fields_are_required(self):
        with self.assertRaisesRegex(ValueError, "corrected-protocol fields"):
            merge.compatibility_signature({"feature_dir": "/tmp/features"})

    def test_encoder_provenance_requires_model_state_hash(self):
        payload = {
            "encoder_provenance": {
                "checkpoint": "checkpoint",
                "checkpoint_file_sha256": None,
                "model_state_sha256": "a" * 64,
                "preprocess_sha256": "b" * 64,
                "encoder_loader_sha256": "c" * 64,
            }
        }
        self.assertEqual(
            merge.validate_encoder_provenance(payload)["model_state_sha256"],
            "a" * 64,
        )
        payload["encoder_provenance"]["model_state_sha256"] = "short"
        with self.assertRaisesRegex(ValueError, "invalid encoder provenance hash"):
            merge.validate_encoder_provenance(payload)

    def test_rank_l_diagnostics_are_aggregated_from_primary_rows(self):
        rows = []
        for budget, error, bound, certified, prefix in [
            (1, 0.1, 0.2, 1, "True"),
            (2, 0.2, 0.3, 1, "False"),
        ]:
            row = result_row("rank_l_gram", 12.0)
            row.update({
                "budget": str(budget),
                "rank_l_selected_absolute_log_error": str(error),
                "rank_l_selected_log_error_bound": str(bound),
                "rank_l_selected_interval_covers_exact": "True",
                "rank_l_certified_steps": str(certified),
                "rank_l_all_steps_certified": "False",
                "rank_l_matches_exact_greedy_prefix": prefix,
            })
            rows.append(row)
        summary = merge.summarize_rank_l_diagnostics(rows, "rank_l_gram")[0]
        self.assertEqual(summary["selected_interval_coverage_rate"], 1.0)
        self.assertAlmostEqual(summary["selected_absolute_log_error_mean"], 0.15)
        self.assertEqual(summary["certified_steps_total"], 1)
        self.assertEqual(summary["evaluated_steps_total"], 2)
        self.assertEqual(summary["exact_greedy_prefix_match_rate"], 0.5)

    def test_manifest_shard_requires_exact_row_coverage(self):
        payload = corrected_payload()
        rows = complete_shard_rows()
        merge.validate_manifest_shard(payload, rows)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge.validate_manifest_shard(payload, rows[:-1])

    def test_encoder_shards_reject_missing_or_changed_provenance(self):
        first = corrected_payload()
        first.update({
            "shard_index": 0,
            "shard_count": 2,
            "assigned_collections": [[7, 0.5]],
        })
        second = dict(first)
        second["shard_index"] = 1
        second["assigned_collections"] = []
        self.assertEqual(
            merge.validate_encoder_shards("encoder_a", [first, second]),
            first["encoder_provenance"],
        )
        changed = dict(second)
        changed["encoder_provenance"] = dict(second["encoder_provenance"])
        changed["encoder_provenance"]["model_state_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            merge.validate_encoder_shards("encoder_a", [first, changed])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge.validate_encoder_shards("encoder_a", [first])


if __name__ == "__main__":
    unittest.main()
