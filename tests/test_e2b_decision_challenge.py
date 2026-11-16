import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_e2b_decision_challenge import partition_candidates, score_method, within_composition
from time_e2b_decisions import execute
from run_e2b_readout_projection import fit_readout, load_spec, model_scores, prediction_losses


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(51)
        self.blocks = {f"{d}_{i}": rng.normal(size=(6, 4)) for d in ("a", "b") for i in range(3)}
        self.domains = {n: n[0] for n in self.blocks}
        self.target = rng.normal(size=(7, 4))

    def test_scores_match_definitions(self):
        a = score_method(self.blocks, self.domains, self.target, 3, "target_a")
        m = score_method(self.blocks, self.domains, self.target, 3, "second_moment_mmd")
        means = score_method(self.blocks, self.domains, self.target, 3, "mean_matching")
        ct = self.target.T @ self.target / len(self.target)
        for key in a:
            x = np.concatenate([self.blocks[n] for n in key.split("|")])
            gram = x.T @ x
            eigen, vectors = np.linalg.eigh(gram)
            expected = np.sum(np.sum(vectors * (ct @ vectors), axis=0) / (1 + eigen))
            self.assertAlmostEqual(a[key], expected)
            self.assertAlmostEqual(m[key], np.sum((gram / len(x) - ct) ** 2))
            self.assertAlmostEqual(means[key], np.sum((x.mean(0) - self.target.mean(0)) ** 2))

    def test_domain_score_cannot_distinguish_same_composition(self):
        scores = score_method(self.blocks, self.domains, self.target, 3, "domain_moment")
        self.assertAlmostEqual(scores["a_0|a_1|b_0"], scores["a_1|a_2|b_2"])

    def test_partition_is_label_free_preserves_members_and_is_order_invariant(self):
        loaded = {"candidate_ids": {n: [f"{n}:{i}" for i in range(6)] for n in self.blocks},
                  "candidate_domains": self.domains}
        first, domains = partition_candidates(loaded, "hash_partition", "fixed")
        loaded["labels"] = "must not be inspected"
        loaded["candidate_ids"] = {n: list(reversed(v)) for n,v in reversed(list(loaded["candidate_ids"].items()))}
        second, _ = partition_candidates(loaded, "hash_partition", "fixed")
        self.assertEqual(first, second)
        self.assertEqual(set(sum(first.values(), [])), set(sum(loaded["candidate_ids"].values(), [])))
        self.assertTrue(all(len(v) == 6 for v in first.values()))
        self.assertEqual(len(domains), 6)

    def test_conditioning_removes_singletons_and_only_ranks_inside_stratum(self):
        scores = score_method(self.blocks, self.domains, self.target, 3, "mean_matching")
        ordered = sorted(scores)
        losses = {c: {"brier_score": i / 100} for i,c in enumerate(ordered)}
        rows = within_composition({"ascending": ordered, "descending": ordered[::-1]},
                                  losses, self.domains, ["brier_score"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["candidate_count"] == 9 for r in rows))
        self.assertTrue(all(r["oracle_retained"] == (r["group"] == "ascending") for r in rows))

    def test_timing_exhaustive_uses_validation_not_test_oracle(self):
        spec = load_spec(ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json")
        labels = {n: np.arange(6) % 3 for n in self.blocks}
        records, losses = execute(self.blocks, labels, self.domains, self.target, self.target,
            np.arange(7) % 3, 3, "exhaustive", "ridge", [2, 5], spec, 7)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["shortlist_size"], 20)
        for metric, choice in records[0]["selections"].items():
            self.assertEqual(choice, min(losses, key=lambda c: (losses[c][metric], c)))
        self.assertGreater(records[0]["decision_seconds"], records[0]["score_seconds"])

    def test_float32_cache_uses_float64_ridge_statistics(self):
        blocks = {n: x.astype(np.float32) for n, x in self.blocks.items()}
        target = self.target.astype(np.float32)
        labels = {n: np.arange(6) % 3 for n in blocks}
        truth = np.arange(len(target)) % 3
        spec = load_spec(ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json")
        _, losses = execute(blocks, labels, self.domains, target, target, truth,
                            3, "exhaustive", "ridge", [2, 5], spec, 7)
        for c, observed in losses.items():
            names = c.split("|")
            w, active, _ = fit_readout(np.concatenate([blocks[n] for n in names]),
                np.concatenate([labels[n] for n in names]), 3, "ridge", spec)
            expected = prediction_losses(model_scores(target, w, active), truth)
            for metric in expected:
                self.assertAlmostEqual(observed[metric], expected[metric], places=12)

    def test_small_random_prefix_needs_no_unselected_block_features(self):
        seed = 7
        choices = sorted(score_method(self.blocks, self.domains, self.target, 3, "random"))
        first = choices[np.random.default_rng(seed).permutation(len(choices))[0]]
        used = set(first.split("|"))
        blocks = {n: x if n in used else None for n, x in self.blocks.items()}
        labels = {n: np.arange(6) % 3 if n in used else None for n in self.blocks}
        spec = load_spec(ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json")
        checkpoints, losses = execute(blocks, labels, self.domains, self.target, self.target,
            np.arange(7) % 3, 3, "random", "ridge", [1], spec, seed)
        self.assertEqual(set(losses), {first})
        self.assertEqual(checkpoints[0]["shortlist_size"], 1)


if __name__ == "__main__":
    unittest.main()
