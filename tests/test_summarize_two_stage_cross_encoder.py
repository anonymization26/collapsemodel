import sys
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


if __name__ == "__main__":
    unittest.main()
