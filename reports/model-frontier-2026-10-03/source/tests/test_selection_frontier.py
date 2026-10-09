import json
import unittest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sbercluster.selection_frontier import fit_frontier, predict_frontier, silhouette_from_distances


class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(9).normal(size=(45, 3))
        self.x[:15] += 3
        self.x[15:30] -= 3

    def test_score_matches_sklearn_including_singleton(self):
        labels = np.r_[np.zeros(15, int), np.ones(29, int), 2]
        score, samples = silhouette_from_distances(cdist(self.x, self.x), labels)
        np.testing.assert_allclose(samples, silhouette_samples(self.x, labels), atol=2e-15)
        self.assertAlmostEqual(score, samples.mean())

    def test_saved_distance_rule_reproduces_transformed_huber_fit(self):
        labels, model, _ = fit_frontier(self.x, {"k": 3, "shrinkage": .6, "huber_quantile": .5})
        np.testing.assert_array_equal(labels, predict_frontier(self.x, json.loads(json.dumps(model))))
        shifted = self.x[:6] + .2
        expected = cdist(shifted @ model['transform'], model['centers'], 'sqeuclidean').argmin(axis=1)
        np.testing.assert_array_equal(expected, predict_frontier(shifted, model))

    def test_tuning_is_monotone_and_full_model_predicts_exactly(self):
        z, model, info = fit_frontier(self.x, {"k": 3, "tune_silhouette": True, "max_passes": 2})
        np.testing.assert_array_equal(z, predict_frontier(self.x, model))
        values = [row['SW'] for row in info['trace']]
        self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
        self.assertGreaterEqual(min(info['cluster_sizes']), info['min_required_size'])

    def test_reject_invalid_shape_and_spec(self):
        for spec in ({'k': True}, {'k': 3, 'shrinkage': float('nan')},
                     {'k': 3, 'huber_quantile': 0}, {'k': 3, 'min_fraction': .4}):
            with self.assertRaises(ValueError):
                fit_frontier(self.x, spec)
        _, model, _ = fit_frontier(self.x, {'k': 3})
        with self.assertRaises(ValueError):
            predict_frontier(self.x[:, :2], model)

    def test_ch_guard_is_computed_in_original_space(self):
        reference, _, _ = fit_frontier(self.x, {'k': 3})
        z, model, info = fit_frontier(self.x, {'k': 3, 'shrinkage': .5, 'min_ch_ratio': .95,
                                              'tune_silhouette': True, 'max_passes': 3},
                                      initial_labels=reference)
        self.assertGreaterEqual(calinski_harabasz_score(self.x, z),
                                .95 * calinski_harabasz_score(self.x, reference) - 1e-10)
        np.testing.assert_array_equal(z, predict_frontier(self.x, model))
        self.assertIsNotNone(info['maximum_common_space_sse'])


if __name__ == '__main__':
    unittest.main()
