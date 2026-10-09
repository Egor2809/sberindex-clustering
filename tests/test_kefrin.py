import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from sbercluster.kefrin import kefrin


class KefrinTests(unittest.TestCase):
    def test_recovers_planted_groups_from_features_and_graph(self):
        rng = np.random.default_rng(0)
        truth = np.repeat([0, 1, 2], 40)
        x = rng.normal(size=(120, 3)) + 4 * np.eye(3)[truth]
        a = (truth[:, None] == truth[None, :]).astype(float) * (rng.random((120, 120)) < 0.3)
        a = np.maximum(a, a.T)
        np.fill_diagonal(a, 0)
        self.assertEqual(adjusted_rand_score(truth, kefrin(x, a, 3, n_init=5)), 1.0)

    def test_graph_breaks_feature_tie(self):
        rng = np.random.default_rng(1)
        truth = np.repeat([0, 1], 50)
        x = rng.normal(size=(100, 2))
        a = (truth[:, None] == truth[None, :]).astype(float)
        np.fill_diagonal(a, 0)
        self.assertGreater(adjusted_rand_score(truth, kefrin(x, a, 2, n_init=5)), 0.9)


if __name__ == "__main__":
    unittest.main()
