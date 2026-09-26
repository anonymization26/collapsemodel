import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_e2b_complementarity as audit


class ComplementarityTests(unittest.TestCase):
    def fixture(self, c_validation=0.9, c_test=0.1):
        validation = {c: {"brier_score": v} for c, v in
                      {"a": .1, "b": .2, "c": c_validation, "d": .8}.items()}
        test = {c: {"brier_score": v} for c, v in
                {"a": .5, "b": .7, "c": c_test, "d": .8}.items()}
        return ["c", "a", "b", "d"], ["a", "b", "d", "c"], validation, test

    def run_task(self, **kwargs):
        return audit.evaluate_task(*self.fixture(**kwargs), "brier_score", 2, top_q=1)

    def test_rescuing_oracle_does_not_imply_validation_selects_it(self):
        pair, choices, events = self.run_task()
        self.assertTrue(pair["has_a_rescue"])
        self.assertTrue(pair["a_rescues_any_oracle"])
        self.assertFalse(pair["union_selects_a_only"])
        self.assertFalse(pair["union_better_than_mmd"])
        self.assertAlmostEqual(pair["omission_reduction_from_union"], .4)
        self.assertEqual(events[0]["candidate"], "c")
        self.assertFalse(events[0]["selected_by_union"])

    def test_rescue_can_reach_better_final_selection(self):
        pair, choices, _ = self.run_task(c_validation=.05)
        self.assertTrue(pair["a_rescue_reaches_final_improvement"])
        union = next(r for r in choices if r["method"] == "union")
        self.assertAlmostEqual(union["delta_vs_mmd"], -.4)
        self.assertEqual(union["selection"], "c")

    def test_union_can_worsen_test_loss_despite_lower_validation_loss(self):
        pair, choices, _ = self.run_task(c_validation=.05, c_test=.9)
        self.assertTrue(pair["union_worse_than_mmd"])
        self.assertFalse(pair["a_rescue_reaches_final_improvement"])
        union = next(r for r in choices if r["method"] == "union")
        self.assertAlmostEqual(union["delta_vs_mmd"], .4)

    def test_union_has_exactly_budget_matched_mmd_comparator(self):
        a, m, _, _ = self.fixture()
        lists = audit.make_shortlists(a, m, 2)
        self.assertEqual(len(lists["union"]), 3)
        self.assertEqual(lists["mmd_matched_union_size"], ["a", "b", "d"])
        for method in ("interleave_mmd_first", "interleave_a_first", "mmd_2s_then_a"):
            self.assertEqual(len(lists[method]), 2)
            self.assertEqual(len(set(lists[method])), 2)

    def test_interleaving_skips_duplicates_and_reports_both_start_orders(self):
        a, m = ["a", "b", "c", "d", "e"], ["a", "b", "e", "d", "c"]
        self.assertEqual(audit.interleave(m, a, 3), ["a", "b", "e"])
        self.assertEqual(audit.interleave(a, m, 3), ["a", "b", "c"])

    def test_cascade_does_not_recover_candidates_removed_by_first_filter(self):
        a = ["e", "d", "c", "b", "a"]
        m = ["a", "b", "c", "d", "e"]
        self.assertEqual(audit.make_shortlists(a, m, 2)["mmd_2s_then_a"], ["d", "c"])

    def test_lexicographic_validation_ties_ignore_ranking_order(self):
        validation = {"a": {"brier_score": .1}, "b": {"brier_score": .1}}
        self.assertEqual(audit.choose(["b", "a"], validation, "brier_score"), "a")

    def test_oracle_ties_are_not_misreported_as_rescues(self):
        values = {"a": {"brier_score": .2}, "b": {"brier_score": .1}, "c": {"brier_score": .1}}
        pair, choices, _ = audit.evaluate_task(["b", "a", "c"], ["c", "a", "b"],
                                                values, values, "brier_score", 1, top_q=1)
        self.assertTrue(pair["has_a_rescue"])
        self.assertFalse(pair["a_rescues_any_oracle"])
        self.assertAlmostEqual(pair["a_rescued_fractional_top_q"], .5)
        for method in ("a_opt", "mmd"):
            row = next(r for r in choices if r["method"] == method)
            self.assertTrue(row["any_test_oracle_retained"])
            self.assertAlmostEqual(row["fractional_recall_at_top_q"], .5)

    def test_choices_do_not_depend_on_test_losses(self):
        _, first, _ = self.run_task(c_validation=.05, c_test=.1)
        _, second, _ = self.run_task(c_validation=.05, c_test=.9)
        self.assertEqual([(r["method"], r["selection"]) for r in first],
                         [(r["method"], r["selection"]) for r in second])

    def test_rejects_duplicate_or_incomplete_candidates_and_nonfinite_losses(self):
        a, m, validation, test = self.fixture()
        with self.assertRaises(ValueError):
            audit.validate_task(["a"] * 4, m, validation, test, ["brier_score"], 4)
        with self.assertRaises(ValueError):
            audit.make_shortlists(a, m, 5)
        test["a"]["brier_score"] = float("nan")
        with self.assertRaises(ValueError):
            audit.validate_task(a, m, validation, test, ["brier_score"], 4)

    def test_aggregation_weights_domains_not_repeat_counts(self):
        rows = [{"scope": "new", "target_domain": "a", "delta": -1., "improved": True}] * 4
        rows += [{"scope": "new", "target_domain": "b", "delta": 3., "improved": False}]
        domains, summaries = audit.aggregate(rows, ("scope",), ("delta",), ("improved",))
        self.assertEqual(len(domains), 2)
        self.assertEqual(summaries[0]["delta_mean"], 1.)
        self.assertEqual(summaries[0]["improved_rate"], .5)
        self.assertEqual(summaries[0]["improved_tasks"], 4)

    def test_source_payload_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "rankings.json"
            payload.write_text(json.dumps({"a": ["x"]}))
            manifest = {"files": {"rankings.json": audit.sha256_file(payload)}}
            with patch.object(audit, "ROOT", root):
                self.assertEqual(audit.checked_payload(root, manifest, "rankings.json", {}), {"a": ["x"]})
                payload.write_text("{}")
                with self.assertRaises(ValueError):
                    audit.checked_payload(root, manifest, "rankings.json", {})

    def test_source_manifest_identity_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {"encoder": "test"}
            core = {"stage": "screen", "context": context, "status": "complete", "files": {}}
            artifact = {**core, "artifact_id": audit.canonical_json_sha256(core)}
            (root / "manifest.json").write_text(json.dumps(artifact))
            with patch.object(audit, "ROOT", root):
                audit.checked_manifest(root, context, "screen", {})
                with self.assertRaises(ValueError):
                    audit.checked_manifest(root, {"encoder": "wrong"}, "screen", {})

    def test_identical_rankings_have_no_rescue_and_no_final_change(self):
        a, _, validation, test = self.fixture()
        pair, choices, events = audit.evaluate_task(a, a, validation, test, "brier_score", 2)
        self.assertFalse(pair["has_a_rescue"])
        self.assertFalse(pair["has_mmd_rescue"])
        self.assertEqual(events, [])
        self.assertTrue(all(r["delta_vs_mmd"] == 0 for r in choices))

    def test_zero_rescue_report_still_has_machine_readable_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rescue.csv"
            audit.dump_csv(path, [], audit.RESCUE_FIELDS)
            self.assertEqual(path.read_text().strip().split(","), list(audit.RESCUE_FIELDS))


if __name__ == "__main__":
    unittest.main()
