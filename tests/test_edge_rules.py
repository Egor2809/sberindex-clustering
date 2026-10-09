import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sbercluster.dynamics import cluster_events
from sbercluster.edge_study import seasonal_profile
from sbercluster.edge_rules import (
    cosine_similarity,
    distance_knn,
    dtw_distance_matrix,
    edge_jaccard,
    lagged_correlation,
    residual_series,
    similarity_knn,
    structure,
)


def brute_dtw(a, b, window):
    t = len(a)
    cost = np.full((t + 1, t + 1), np.inf)
    cost[0, 0] = 0
    for p in range(1, t + 1):
        for q in range(max(1, p - window), min(t, p + window) + 1):
            cost[p, q] = np.square(a[p - 1] - b[q - 1]).sum() + min(cost[p - 1, q], cost[p, q - 1], cost[p - 1, q - 1])
    return np.sqrt(cost[t, t])


class EdgeRuleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.series = rng.normal(size=(12, 6, 2))

    def test_residual_series_removes_common_month_pattern(self):
        common = np.exp(np.linspace(0, 1, 10))[:, None, None]
        own = np.exp(np.random.default_rng(1).normal(size=(10, 4, 2)))
        np.testing.assert_allclose(residual_series(common * own), residual_series(own))

    def test_lag_zero_equals_pearson_average(self):
        r, lag = lagged_correlation(self.series, 0)
        expected = np.mean([np.corrcoef(self.series[:, :, m].T) for m in range(2)], axis=0)
        np.testing.assert_allclose(r, expected, atol=1e-12)
        self.assertFalse(lag.any())

    def test_lagged_correlation_finds_leader(self):
        base = np.sin(np.arange(16) / 2.0)
        series = np.stack([base, np.roll(base, 2), np.cos(np.arange(16))], axis=1)[:, :, None]
        r, lag = lagged_correlation(series, 3)
        self.assertEqual(lag[0, 1], 2)
        self.assertEqual(lag[1, 0], -2)
        self.assertGreater(r[0, 1], 0.9)
        self.assertAlmostEqual(r[0, 1], r[1, 0])

    def test_dtw_matches_reference_recursion(self):
        d = dtw_distance_matrix(self.series, window=2)
        self.assertTrue(np.allclose(d, d.T))
        self.assertAlmostEqual(d[1, 4], brute_dtw(self.series[:, 1], self.series[:, 4], 2))
        self.assertTrue(np.all(np.diag(d) == 0))

    def test_knn_graphs_are_symmetric_and_loop_free(self):
        s = cosine_similarity(np.abs(self.series[0]) + 0.1)
        for a in (similarity_knn(s, 2), distance_knn(1 - s, 2)[0]):
            self.assertEqual((a != a.T).nnz, 0)
            self.assertFalse(a.diagonal().any())
            self.assertTrue(np.all(np.asarray((a > 0).sum(axis=1)).ravel() >= 2))

    def test_negative_similarity_never_creates_edges(self):
        a = similarity_knn(-np.ones((4, 4)), 2)
        self.assertEqual(a.nnz, 0)

    def test_structure_on_triangle_with_tail(self):
        a = csr_matrix([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
        result = structure(a, {'group': [0, 0, 0, 1]})
        self.assertEqual(result['edges'], 4)
        self.assertEqual(result['components'], 1)
        self.assertAlmostEqual(result['transitivity'], 3 / 5)
        self.assertAlmostEqual(result['within_group'], 3 / 4)
        self.assertAlmostEqual(edge_jaccard(a, a), 1.0)


class SeasonalProfileTests(unittest.TestCase):
    def test_linear_trend_is_not_absorbed(self):
        pattern = np.sin(np.arange(12))
        series = (0.5 * np.arange(24) + np.tile(pattern, 2))[:, None]
        np.testing.assert_allclose(seasonal_profile(series)[:, 0], pattern - pattern.mean(), atol=1e-12)


class ClusterEventTests(unittest.TestCase):
    def test_split_merge_birth_death(self):
        ids = list('abcdefgh')
        split = cluster_events(ids, [0, 0, 0, 0, 1, 1, 1, 1], ids, [0, 0, 2, 2, 1, 1, 1, 1])
        self.assertIn('split', [e['event'] for e in split])
        merge = cluster_events(ids, [0, 0, 2, 2, 1, 1, 1, 1], ids, [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertIn('merge', [e['event'] for e in merge])
        churn = cluster_events(ids, [0, 0, 0, 0, 1, 1, 1, 1], ids, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual({e['event'] for e in churn}, {'death', 'birth'})

    def test_one_to_one_size_changes(self):
        ids = list('abcdefghij')
        events = cluster_events(ids, [0] * 5 + [1] * 5, ids, [0] * 7 + [1] * 3)
        kinds = {tuple(e['from']): e['event'] for e in events}
        self.assertEqual(kinds[(0,)], 'grow')
        self.assertEqual(kinds[(1,)], 'shrink')
        same = cluster_events(ids, [0] * 5 + [1] * 5, ids, [1] * 5 + [0] * 5)
        self.assertEqual([e['event'] for e in same], ['continue', 'continue'])


if __name__ == '__main__':
    unittest.main()
