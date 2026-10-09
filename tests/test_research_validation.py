import unittest
import numpy as np
from sbercluster.research_validation import (
    nearest_indices, geographic_distances, consensus_peers,
    regional_bootstrap_difference, eta_squared,
)


class FrozenValidation(unittest.TestCase):
    def test_neighbors_exclude_self_and_preserve_ties(self):
        distance = np.array([[0,1,1],[1,0,2],[1,2,0]],float)
        got = nearest_indices(distance,1)
        self.assertEqual([x.tolist() for x in got], [[1],[0],[0]])

    def test_empty_allowed_peer_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'No peers'):
            nearest_indices(np.zeros((3,3)),2,np.eye(3,dtype=bool))

    def test_haversine_symmetric_and_one_degree(self):
        distance=geographic_distances([0,0],[0,1])
        self.assertAlmostEqual(distance[0,1],111.19508,places=4)
        np.testing.assert_equal(distance,distance.T)
        np.testing.assert_equal(distance.diagonal(),[0,0])

    def test_constant_regional_improvement_has_exact_interval(self):
        result=regional_bootstrap_difference([1,2,3,4],[3,4,5,6],['a','a','b','b'])
        self.assertEqual(result['mean_improvement'],2)
        np.testing.assert_equal(result['ci_95'],[2,2])

    def test_consensus_identical_months_recovers_nearest_peer(self):
        x=np.array([[0.],[1.],[3.],[7.]])
        peers,freq=consensus_peers(np.stack([x]*12),k=1,candidate_k=2)
        self.assertEqual([a.tolist() for a in peers],[[1],[0],[1],[2]])
        self.assertEqual(freq,[[1.]]*4)

    def test_eta_squared_known_extremes(self):
        self.assertEqual(eta_squared([0,0,5,5],[0,0,1,1]),1.)
        self.assertEqual(eta_squared([0,5,0,5],[0,0,1,1]),0.)


if __name__=='__main__': unittest.main()
