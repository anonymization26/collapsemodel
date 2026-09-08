import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_two_stage_lodo as lodo  # noqa: E402


def row(domain, budget, removal, method="rank_only", replicate="-1"):
    return {
        "excluded_domain": domain,
        "encoder": "e",
        "target": "t",
        "method": method,
        "replicate": replicate,
        "shortlist_size": str(budget),
        "removed_fraction": str(removal),
        "best_candidate_recall": "1",
        "top3_candidate_recall": "1",
        "best_candidate_family_recall": "1",
        "top3_candidate_family_recall": "1",
        "validation_regret": "0",
        "normalized_validation_regret": "0",
        "test_regret_vs_exhaustive_validation_selection": "0",
        "selected_family_count": str(budget),
        "removed_family_fraction": str(removal),
    }


class LodoSummaryTests(unittest.TestCase):
    def test_primary_budget_is_closest_point_removing_at_least_half(self):
        rows = [
            row("a", 3, 0.7),
            row("a", 5, 0.5),
            row("a", 6, 0.4),
            row("b", 3, 0.8),
            row("b", 8, 0.52),
        ]
        self.assertEqual(lodo.choose_primary_budgets(rows), {"a": 5, "b": 8})

    def test_primary_rows_include_random_replicates_at_selected_budget(self):
        rows = [
            row("a", 3, 0.7),
            row("a", 5, 0.5),
            row("a", 5, 0.5, method="random", replicate="0"),
        ]
        selected = lodo.primary_rows(rows, {"a": 5})
        self.assertEqual(len(selected), 2)
        self.assertEqual({value["method"] for value in selected}, {"rank_only", "random"})


if __name__ == "__main__":
    unittest.main()
