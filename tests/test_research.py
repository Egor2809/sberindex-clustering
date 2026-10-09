import unittest
import numpy as np

from sbercluster.research import annual_profile, block_month_indices, local_scale_graph, fit_candidate


class ResearchContracts(unittest.TestCase):
    def test_local_affinity_matches_hand_calculation(self):
        x = np.array([[0.], [1.], [3.], [7.]])
        graph, info = local_scale_graph(x, k=2, scale_neighbor=1)
        self.assertAlmostEqual(graph[0, 1], np.exp(-1))
        self.assertAlmostEqual(graph[0, 2], np.exp(-9 / 2))
        self.assertEqual(graph[0, 3], 0)
        self.assertEqual((graph != graph.T).nnz, 0)
        self.assertTrue(np.all(graph.diagonal() == 0))
        self.assertEqual(info['components'], 1)

    def test_local_graph_is_invariant_to_uniform_units(self):
        x = np.array([[0., 0.], [1., 2.], [3., 1.], [7., 5.], [9., 12.]])
        a, _ = local_scale_graph(x, k=3, scale_neighbor=2)
        b, _ = local_scale_graph(100 * x, k=3, scale_neighbor=2)
        np.testing.assert_allclose(a.toarray(), b.toarray())

    def test_duplicate_scale_is_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, 'zero'):
            local_scale_graph(np.array([[0.], [0.], [1.]]), k=2, scale_neighbor=1)

    def test_reference_profile_uses_monthly_median(self):
        monthly = np.array([[[1., 2.]], [[3., 4.]], [[100., 200.]]])
        np.testing.assert_equal(annual_profile(monthly), [[3., 4.]])

    def test_block_resampling_is_reproducible_and_calendar_bounded(self):
        a = block_month_indices(1729)
        np.testing.assert_equal(a, block_month_indices(1729))
        self.assertEqual(len(a), 12)
        self.assertTrue(np.all((0 <= a) & (a < 12)))
        for block in a.reshape(4, 3):
            np.testing.assert_equal((block[1:] - block[:-1]) % 12, [1, 1])

    def test_new_fit_paths_keep_execution_gate(self):
        with self.assertRaises(PermissionError):
            fit_candidate(np.ones((4, 2)), None, {'method': 'spectral_local', 'k': 2},
                          1729, {'execution': {'allow_clustering': False}})


if __name__ == '__main__':
    unittest.main()
