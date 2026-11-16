"""Synthetic plumbing tests only; no scientific measurements are fixtures."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
from em_geometry.metrics import (correlations, equal_count, equal_width,
                                  residual_layers, equal_count_sweep)


class MetricsTests(unittest.TestCase):
    def test_tied_ranks_match_scipy(self):
        x, y = [1., 1., 3., 4.], [4., 2., 2., 1.]
        c = correlations(x, y)
        self.assertEqual(c.spearman, stats.spearmanr(x, y).statistic)
        self.assertEqual(c.pearson, stats.pearsonr(x, y).statistic)
        self.assertEqual(c.mean_rp, (c.spearman + c.pearson) / 2)

    def test_constant_and_too_short_are_nan(self):
        for x, y in [([1, 1], [2, 3]), ([1, 2], [3, 3]), ([1], [2]), ([], [])]:
            self.assertTrue(np.isnan(correlations(x, y).mean_rp))

    def test_invalid_pairs_fail(self):
        for x, y in [([1, 2], [1]), ([1, np.nan], [1, 2]), ([[1]], [[2]])]:
            with self.assertRaises(ValueError):
                correlations(x, y)

    def test_count_float32_sort_and_dropped_tail(self):
        d = np.array([4, 0, 2, 1, 3], dtype=np.float32)
        x, y = equal_count(d, -d, 2)
        self.assertEqual(x.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(x, [.5, 2.5])
        np.testing.assert_array_equal(y, [-.5, -2.5])

    def test_sort_modes_are_distinct_explicit_protocols(self):
        d = np.array([2, 1, 1, 0] * 12, dtype=np.float32)
        y = np.arange(len(d))
        original = np.argsort
        for mode in ("default", "stable"):
            with patch("em_geometry.metrics.np.argsort", wraps=original) as sorter:
                x, by = equal_count(d, y, 5, sort=mode)
                if mode == "default":
                    sorter.assert_called_once_with(d)
                else:
                    sorter.assert_called_once_with(d, kind="stable")
            o = original(d) if mode == "default" else original(d, kind="stable")
            np.testing.assert_array_equal(by, y[o[:45]].reshape(9, 5).mean(axis=1))
            self.assertEqual(len(x), 9)
        with self.assertRaises(ValueError):
            equal_count(d, y, sort="invented")

    def test_width_midpoints_nonempty_and_count_not_conflated(self):
        x, y = equal_width([0., 1., 10.], [0., 4., 8.], 5)
        np.testing.assert_allclose(x, [1., 9.])
        np.testing.assert_allclose(y, [2., 8.])
        self.assertEqual(len(equal_count([0., 1., 10.], [0., 4., 8.], 2)[0]), 1)

    def test_no_full_bins(self):
        x, y = equal_count([1, 2], [2, 1], 20)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_excludes_embedding_final_and_constant_residual(self):
        d = np.column_stack([np.arange(8), np.arange(8), np.ones(8),
                             np.arange(8)[::-1]])
        self.assertEqual(residual_layers(d), [1])
        curve, best = equal_count_sweep(d, -np.arange(8), 2)
        self.assertEqual([r["layer"] for r in curve], [1])
        self.assertEqual(best["layer"], 1)
        self.assertAlmostEqual(best["mean_rp"], -1.)

    def test_no_eligible_or_finite_sweep_fails(self):
        with self.assertRaises(ValueError):
            equal_count_sweep(np.ones((4, 4)), np.arange(4), 2)
        with self.assertRaises(ValueError):
            equal_count_sweep(np.tile(np.arange(4)[:, None], (1, 4)), np.ones(4), 2)


if __name__ == "__main__":
    unittest.main()
