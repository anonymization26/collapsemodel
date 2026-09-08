import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_two_stage_source_lodo as source_lodo  # noqa: E402


def row(excluded_source, encoder, method, recall):
    return {
        "excluded_source": excluded_source,
        "encoder": encoder,
        "target": "target",
        "method": method,
        "replicate": "-1",
        "shortlist_size": "10",
        "removed_fraction": "0.5",
        "best_candidate_recall": str(recall),
        "top3_candidate_recall": str(recall),
        "validation_regret": "0",
        "normalized_validation_regret": "0",
        "test_regret_vs_exhaustive_validation_selection": "0",
    }


class SourceLodoSummaryTests(unittest.TestCase):
    def test_aggregate_tracks_exclusion_units(self):
        rows = [
            row("a", "e1", "rank_only", 1),
            row("b", "e1", "rank_only", 0),
        ]
        overall = next(
            value for value in source_lodo.aggregate_rows(rows)
            if value["scope"] == "all_source_lodo"
        )
        self.assertEqual(overall["excluded_sources"], 2)
        self.assertAlmostEqual(overall["best_candidate_recall_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
