import itertools
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.collapse_core import (  # noqa: E402
    collapse_4s,
    collapse_4s_predict,
    effective_rank_from_scatter,
    effective_rank_from_singular_values,
    is_superadditive,
    superadditivity_delta,
)
from metrics.reff import (  # noqa: E402
    make_controlled_pair,
    predict_reff_collapse,
    reff_theoretical_prediction,
)
from metrics.subspace_alignment import compute_subspace_alignment  # noqa: E402
import two_stage_classic_baselines as classic  # noqa: E402


class CollapseCoreTests(unittest.TestCase):
    def test_all_public_predictors_delegate_to_one_tie_rule(self):
        expected = collapse_4s_predict(2.0, 500.0, gamma=1.0, alpha=1.0)
        self.assertEqual(expected, 2.0)
        self.assertEqual(classic.collapse_predict(2.0, 500.0, 1.0, 1.0), expected)
        self.assertEqual(
            reff_theoretical_prediction(2.0, 500.0, alpha=1.0, gamma=1.0),
            expected,
        )

    def test_superadditivity_uses_higher_nuclear_mass_source(self):
        self.assertEqual(
            superadditivity_delta(10.0, 2.0, 500.0, gamma=1.0),
            8.0,
        )
        self.assertEqual(
            superadditivity_delta(510.0, 2.0, 500.0, gamma=2.0),
            10.0,
        )
        self.assertFalse(is_superadditive(2.0 + 1e-12, 2.0, 500.0, gamma=1.0))
        self.assertTrue(is_superadditive(2.1, 2.0, 500.0, gamma=1.0))

    def test_directional_endpoint_is_exact(self):
        gamma = np.nextafter(0.5, 0.0)
        result = collapse_4s(11.0, 37.0, gamma=gamma, alpha=1.0)
        self.assertEqual(result.prediction, 11.0)
        self.assertEqual(result.q, 1.0)
        self.assertTrue(np.isinf(result.rho_c))

    def test_svd_and_scatter_paths_share_the_same_cutoff(self):
        singular = np.array([1.0, 1e-6, 1e-8])
        source_shape = (2048, 3)
        from_svd = effective_rank_from_singular_values(singular, source_shape)
        from_scatter = effective_rank_from_scatter(
            np.diag(singular * singular),
            source_shape=source_shape,
        )
        self.assertEqual(from_svd, 1.0)
        self.assertEqual(from_scatter, from_svd)

    def test_alignment_excludes_numerical_nullspace(self):
        h_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        h_b = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        result = compute_subspace_alignment(h_a, h_b, k=3)
        self.assertEqual(result["k_used"], 1)
        self.assertAlmostEqual(result["sa_k"], 0.0)

    def test_e1a_generator_matches_proportional_spectrum_formula(self):
        maximum_error = 0.0
        for alpha, gamma in itertools.product((0.0, 0.3, 1.0), (0.2, 1.0, 5.0)):
            h_a, h_b = make_controlled_pair(
                d=32,
                k=8,
                n=40,
                target_alpha=alpha,
                gamma=gamma,
                seed=7,
            )
            result = predict_reff_collapse(h_a, h_b, k=8)
            maximum_error = max(
                maximum_error,
                abs(result["r_eff_merged"] - result["r_pred_theory"]),
            )
        self.assertLess(maximum_error, 1e-10)


if __name__ == "__main__":
    unittest.main()
