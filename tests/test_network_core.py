import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sbercluster.network_core import (canonical, central_members, combine_layers, compose_types, consensus, merge_small, normalize_layer,
                                      select_variant)


def two_cliques(n=12, bridge=0.05):
    a = np.zeros((2 * n, 2 * n))
    a[:n, :n] = 1
    a[n:, n:] = 1
    a[0, n] = a[n, 0] = bridge
    np.fill_diagonal(a, 0)
    return csr_matrix(a)


RULE = {'min_seed_ari': 0.8, 'min_size': 20, 'k_min': 3, 'k_max': 12,
        'pareto': [{'key': 'oot_SW', 'direction': 1}, {'key': 'oot_C_MQ', 'direction': 1}],
        'rank': [{'key': 'oot_SW', 'direction': 1, 'weight': 1.0}, {'key': 'oot_S_Dbw', 'direction': -1, 'weight': 1.0},
                 {'key': 'oot_C_MQ', 'direction': 1, 'weight': 1.0}],
        'tiebreak': [{'key': 'seed_ari', 'direction': 1}, {'key': 'k', 'direction': -1}]}


class NetworkCoreTests(unittest.TestCase):
    def test_layers_contribute_equal_weight(self):
        rng = np.random.default_rng(3)
        p = rng.random((8, 8)) * 50
        c = rng.random((8, 8)) * 0.01
        p, c = csr_matrix(p + p.T), csr_matrix(c + c.T)
        p.setdiag(0)
        c.setdiag(0)
        self.assertAlmostEqual(normalize_layer(p).sum(), 1.0)
        self.assertAlmostEqual(normalize_layer(c).sum(), 1.0)
        g = combine_layers(p, c, 0.5)
        self.assertAlmostEqual(g.sum(), 1.0)
        np.testing.assert_allclose(g.toarray(), 0.5 * normalize_layer(p).toarray() + 0.5 * normalize_layer(c).toarray())
        np.testing.assert_allclose(combine_layers(p, c, 1.0).toarray(), normalize_layer(p).toarray())

    def test_consensus_finds_both_communities_deterministically(self):
        a = two_cliques()
        first, info = consensus(a, 1.0, [1, 2, 3, 4, 5], -1, 2)
        second, _ = consensus(a, 1.0, [1, 2, 3, 4, 5], -1, 2)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.repeat([0, 1], 12))
        self.assertAlmostEqual(info['seed_ari'], 1.0)

    def test_small_group_joins_strongest_neighbour(self):
        a = two_cliques(n=10).toarray()
        a = np.pad(a, ((0, 2), (0, 2)))
        a[20, 21] = a[21, 20] = 1
        a[20, 15] = a[15, 20] = 0.9
        a[21, 2] = a[2, 21] = 0.3
        z = merge_small(csr_matrix(a), np.r_[np.zeros(10), np.ones(10), [2, 2]], 5)
        self.assertEqual(len(np.unique(z)), 2)
        self.assertEqual(z[20], z[15])
        self.assertEqual(z[21], z[15])
        isolated = np.pad(two_cliques(n=10).toarray(), ((0, 1), (0, 1)))
        z = merge_small(csr_matrix(isolated), np.r_[np.zeros(10), np.ones(10), [2]], 5)
        self.assertEqual(len(np.unique(z)), 2)

    def test_canonical_orders_by_size(self):
        np.testing.assert_array_equal(canonical([5, 7, 7, 7, 5, 9]), [1, 0, 0, 0, 1, 2])

    def test_selection_rule_on_toy_table(self):
        rows = [{'variant': 'unstable', 'seed_ari': 0.6, 'min_size': 40, 'k': 5, 'oot_SW': 0.9, 'oot_C_MQ': 0.9, 'oot_S_Dbw': 0.1},
                {'variant': 'too_many', 'seed_ari': 0.95, 'min_size': 40, 'k': 20, 'oot_SW': 0.8, 'oot_C_MQ': 0.8, 'oot_S_Dbw': 0.1},
                {'variant': 'dominated', 'seed_ari': 0.9, 'min_size': 40, 'k': 4, 'oot_SW': 0.1, 'oot_C_MQ': 0.1, 'oot_S_Dbw': 0.1},
                {'variant': 'profile', 'seed_ari': 0.9, 'min_size': 40, 'k': 4, 'oot_SW': 0.3, 'oot_C_MQ': 0.2, 'oot_S_Dbw': 0.5},
                {'variant': 'network', 'seed_ari': 0.9, 'min_size': 40, 'k': 6, 'oot_SW': 0.2, 'oot_C_MQ': 0.4, 'oot_S_Dbw': 0.4}]
        chosen, info = select_variant(rows, RULE)
        self.assertEqual(info['admissible'], ['dominated', 'profile', 'network'])
        self.assertEqual(sorted(info['front']), ['network', 'profile'])
        self.assertEqual(chosen, 'network')
        self.assertEqual(select_variant(rows[:2], RULE)[0], None)

    def test_types_compose_level_and_subtype(self):
        level1 = np.array([0, 0, 1, 1, 1, 2, 0])
        types = compose_types(level1, {0: np.array([5, 3, 5]), 1: None})
        self.assertEqual(types.tolist(), ['0.1', '0.2', '1.0', '1.0', '1.0', '2', '0.1'])

    def test_central_members_prefer_the_middle(self):
        x = np.array([[0.0], [1.0], [2.0], [10.0], [3.0]])
        np.testing.assert_array_equal(central_members(x, np.array([0, 1, 2, 4]), 2), [1, 2])


if __name__ == '__main__':
    unittest.main()
