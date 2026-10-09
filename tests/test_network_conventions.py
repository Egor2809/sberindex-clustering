import unittest
import numpy as np
from scipy.sparse import csr_matrix
from sbercluster.metrics import network_indices, network_quality_candidates

class NetworkConventions(unittest.TestCase):
    def test_nonfinite_graph_weights_and_missing_labels_rejected(self):
        for value in (np.inf, np.nan, -1):
            graph = np.array([[0, value], [value, 0]])
            for function in (network_indices, network_quality_candidates):
                for binary in (False, True):
                    with self.subTest(value=value, function=function.__name__, binary=binary):
                        with self.assertRaises(ValueError):
                            function(graph, [0, 1], binary=binary)
        with self.assertRaises(ValueError):
            network_indices(np.array([[0, 1], [1, 0]]), [0, np.nan])

    def test_weighted_indices_are_finite_at_extreme_scales(self):
        graph = csr_matrix([[0,2,0,0],[2,0,1,0],[0,1,0,2],[0,0,2,0]], dtype=float)
        labels = [0,0,1,1]
        original = graph.copy()
        for scale in (1e-300, 1., 1e307):
            indices = network_indices(graph * scale, labels, binary=False)
            quality = network_quality_candidates(graph * scale, labels, binary=False)
            self.assertAlmostEqual(indices['AVI'], .8)
            self.assertAlmostEqual(indices['AVU'], 1.)
            self.assertAlmostEqual(quality['TurboMQ'], 1.6)
            self.assertAlmostEqual(quality['NewmanQ'], .3)
        self.assertEqual((original != graph).nnz, 0)

    def test_small_cross_edge_is_not_lost_by_subtracting_internal_mass(self):
        graph = csr_matrix([[0,1e300,0,0],[1e300,0,1,0],
                            [0,1,0,1e300],[0,0,1e300,0]], dtype=float)
        indices = network_indices(graph,[0,0,1,1],binary=False)
        self.assertEqual(indices['AVI'], 1.)
        self.assertEqual(indices['AVU'], 1.)
        self.assertEqual(indices['AVU_undefined_ordered_pairs'], 0)

    def test_duplicate_sparse_entries_count_as_one_binary_edge(self):
        # Each unordered edge has two storage entries in both directions.
        graph = csr_matrix((np.ones(8), np.array([1,1,0,0,3,3,2,2]),
                            np.array([0,2,4,6,8])), shape=(4,4))
        original_data = graph.data.copy()
        result = network_quality_candidates(graph, [0,0,1,1])
        self.assertAlmostEqual(result['NewmanQ'], .5)
        self.assertEqual(result['TurboMQ'], 2)
        np.testing.assert_array_equal(graph.data, original_data)

    def test_low_k_avu_is_a_structural_constant(self):
        for weights in [(1, 2, 7), (5, 1, 1)]:
            a = np.array([[0, weights[0], weights[1]], [weights[0], 0, weights[2]], [weights[1], weights[2], 0]], float)
            self.assertAlmostEqual(network_indices(a, [0, 1, 2], binary=False)['AVU'], 2/3)
            self.assertAlmostEqual(network_indices(a, [0, 0, 1], binary=False)['AVU'], 1)

if __name__ == '__main__':unittest.main()
