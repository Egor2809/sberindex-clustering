import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sbercluster.graph import knn_graph
from sbercluster.temporal_quality import (SIX, matched_size_shift, mean_valid, next_month_network, select_omega,
                                          series_rows, six_icvi, summarise)


def planted_graph(blocks, p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    n = len(blocks)
    same = blocks[:, None] == blocks[None, :]
    upper = np.triu(rng.random((n, n)) < np.where(same, p_in, p_out), 1)
    return csr_matrix((upper | upper.T).astype(float))


class TemporalQualityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.truth = np.repeat([0, 1, 2], 20)
        centers = np.array([[0, 0], [6, 0], [0, 6]], float)
        self.monthly = [centers[self.truth] + rng.normal(scale=0.5, size=(60, 2)) + 0.1 * t for t in range(3)]
        self.graphs = [knn_graph(x, 5)[0] for x in self.monthly]

    def test_six_icvi_on_separated_monthly_blobs(self):
        row = six_icvi(self.monthly[0], self.truth, self.graphs[0])
        self.assertEqual(set(SIX) | {'k', 'min_cluster_size'}, set(row))
        self.assertEqual(row['k'], 3)
        self.assertGreater(row['SW'], 0.7)
        self.assertGreater(row['AVI'], 0.95)
        self.assertGreater(row['MQ'], 0.6)
        self.assertLess(row['S_Dbw'], 1.0)
        mixed = six_icvi(self.monthly[0], np.tile([0, 1, 2], 20), self.graphs[0])
        self.assertLess(mixed['SW'], row['SW'])
        self.assertLess(mixed['MQ'], 0.1)

    def test_single_group_month_is_undefined_not_zero(self):
        row = six_icvi(self.monthly[0], np.zeros(60, int), self.graphs[0])
        self.assertTrue(all(row[key] is None for key in SIX))
        self.assertEqual(next_month_network(np.zeros(60, int), self.graphs[1]), {'MQ_next': None, 'AVI_next': None})

    def test_series_rows_align_next_month_and_previous_month(self):
        labels = [self.truth, self.truth, np.where(self.truth == 2, 1, self.truth)]
        rows = series_rows('toy', None, None, labels, self.monthly, self.graphs, ['a', 'b', 'c'])
        self.assertEqual([r['period'] for r in rows], ['a', 'b', 'c'])
        self.assertIsNone(rows[0]['ARI_prev'])
        self.assertIsNone(rows[-1]['MQ_next'])
        self.assertAlmostEqual(rows[1]['ARI_prev'], 1.0)
        self.assertEqual(rows[1]['churn_prev'], 0.0)
        self.assertAlmostEqual(rows[2]['churn_prev'], 20 / 60)
        self.assertAlmostEqual(rows[2]['size_shift_prev'], 20 / 60)
        self.assertEqual(rows[2]['k'], 2)
        summary = summarise(rows)
        self.assertEqual((summary['k_min'], summary['k_max'], summary['MQ_next_valid']), (2, 3, 2))

    def test_mq_next_separates_persistent_groups_from_month_noise(self):
        blocks = np.repeat(np.arange(4), 30)
        g_next = planted_graph(blocks, 0.3, 0.01, 11)
        persistent = next_month_network(blocks, g_next)
        noise = np.random.default_rng(5).permutation(blocks)
        overfit = next_month_network(noise, g_next)
        self.assertGreater(persistent['MQ_next'], 0.5)
        self.assertGreater(persistent['AVI_next'], 0.8)
        self.assertLess(abs(overfit['MQ_next']), 0.06)
        self.assertLess(overfit['AVI_next'], 0.4)

    def test_select_omega_largest_within_relative_drop(self):
        table = [{'omega': 0, 'MQ_mean': 0.50}, {'omega': 0.01, 'MQ_mean': 0.495}, {'omega': 0.03, 'MQ_mean': 0.4905},
                 {'omega': 0.06, 'MQ_mean': 0.489}, {'omega': 0.1, 'MQ_mean': 0.47}]
        choice = select_omega(table, 0.02)
        self.assertAlmostEqual(choice['threshold_MQ'], 0.49)
        self.assertEqual(choice['eligible'], [0, 0.01, 0.03])
        self.assertEqual(choice['selected'], 0.03)
        self.assertEqual(select_omega(table, 0.0)['selected'], 0)
        bumpy = table[:2] + [{'omega': 0.03, 'MQ_mean': 0.40}, {'omega': 0.06, 'MQ_mean': 0.499}]
        self.assertEqual(select_omega(bumpy, 0.02)['selected'], 0.06)

    def test_matched_size_shift_and_mean_valid(self):
        a = np.array([0, 0, 0, 1, 1, 2])
        self.assertEqual(matched_size_shift(a, a + 5), 0.0)
        self.assertAlmostEqual(matched_size_shift(a, np.array([0, 0, 0, 1, 1, 1])), 1 / 6)
        self.assertEqual(mean_valid([None, 1.0, 3.0, float('nan')]), (2.0, 2))
        self.assertEqual(mean_valid([None]), (None, 0))


if __name__ == '__main__':
    unittest.main()
