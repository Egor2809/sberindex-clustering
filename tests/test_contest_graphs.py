import unittest
import numpy as np
from scipy.sparse import csr_matrix
from sbercluster.contest_graphs import distance_knn, mix_layers, unit_mean_strength, trajectory_graph, graph_summary
from sbercluster.graph import knn_graph


class LayerTests(unittest.TestCase):
    def test_graph_summary_counts_edges_after_sparse_canonicalization(self):
        graph = csr_matrix((np.array([1.,1.,0.,1.,1.]), np.array([1,1,2,0,0]),
                            np.array([0,3,5,5])), shape=(3,3))
        result = graph_summary(graph)
        self.assertEqual(result['edges'], 1)
        self.assertEqual(result['isolates'], 1)
        self.assertEqual(result['components'], 2)
        self.assertAlmostEqual(result['mean_degree'], 2/3)
        self.assertEqual(graph.nnz, 5)
        self.assertEqual(graph_summary(csr_matrix((0,0)))['mean_degree'], 0.)

    def test_normalization_is_invariant_to_extreme_finite_weight_scale(self):
        graph = csr_matrix([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
        baseline = unit_mean_strength(graph).toarray()
        for scale in (1e-300, 1e308):
            scaled = unit_mean_strength(graph * scale)
            np.testing.assert_allclose(scaled.toarray(), baseline)
            self.assertAlmostEqual(float(scaled.sum()), 3.)

    def test_invalid_neighbor_count_and_empty_feature_axis_rejected(self):
        for k in (True, 1.5, 0, 3):
            for function, data in ((knn_graph, np.arange(3)[:,None]),
                                   (distance_knn, np.abs(np.arange(3)[:,None]-np.arange(3))),
                                   (trajectory_graph, np.zeros((3,3,1)))):
                with self.subTest(k=k, function=function.__name__), self.assertRaises(ValueError):
                    function(data, k=k)
        with self.assertRaises(ValueError):
            knn_graph(np.empty((3,0)), k=1)
        with self.assertRaises(ValueError):
            trajectory_graph(np.empty((3,3,0)), k=1)

    def test_distance_kernel_bandwidth_does_not_overflow_its_median(self):
        graph, _, _, info = distance_knn(np.array([[0,1e308],[1e308,0]]), k=1)
        self.assertEqual(info['bandwidth_km'], 1e308)
        self.assertAlmostEqual(graph[0,1], np.exp(-.5))
        self.assertEqual(info['edges'], 1)

    def test_attribute_distance_overflow_rejected_before_nan_graph(self):
        with self.assertRaisesRegex(ValueError, 'overflowed'):
            knn_graph(np.array([[0.], [1e200], [2e200]]), k=1)

    def test_missing_distance_is_not_a_close_neighbor(self):
        d=np.array([[0,1,np.inf],[1,0,3],[np.inf,3,0]])
        a,_,_,info=distance_knn(d,k=1)
        self.assertEqual(a[0,2],0)
        self.assertGreater(a[0,1],a[1,2])
        self.assertEqual(info['components'],1)

    def test_mixture_is_invariant_to_layer_scale_and_has_exact_endpoints(self):
        a=csr_matrix([[0,2,0],[2,0,1],[0,1,0]],dtype=float)
        b=csr_matrix([[0,0,1],[0,0,2],[1,2,0]],dtype=float)
        np.testing.assert_allclose(mix_layers(a,b,.3).toarray(),mix_layers(a*100,b/20,.3).toarray())
        np.testing.assert_allclose(mix_layers(a,b,1).toarray(),unit_mean_strength(a).toarray())
        self.assertAlmostEqual(float(mix_layers(a,b,.3).sum()),3)

    def test_common_time_shock_does_not_change_residual_graph(self):
        rng=np.random.default_rng(12)
        x=rng.normal(size=(8,6,2))
        shock=np.arange(8)[:,None,None]*np.array([2.,-3.])[None,None,:]
        a,_=trajectory_graph(x,k=2)
        b,_=trajectory_graph(x+shock,k=2)
        np.testing.assert_allclose(a.toarray(),b.toarray(),atol=1e-13)

    def test_empty_layers_rejected(self):
        with self.assertRaises(ValueError):
            unit_mean_strength(csr_matrix((3,3)))
