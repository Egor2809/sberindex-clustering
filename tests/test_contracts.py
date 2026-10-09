import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sbercluster.io import CATEGORIES, TOTAL, canonical_period
from sbercluster.features import profile_matrix, make_slices
from sbercluster.graph import knn_graph
from sbercluster.metrics import network_indices, s_dbw, aligned_stability
from sbercluster.models import fit_static, require_execution
from sbercluster.dynamics import transitions
from sbercluster.validation import heldout_peer_concordance


class Contracts(unittest.TestCase):
    def test_api_timezone_does_not_move_month(self):
        self.assertEqual(canonical_period(["2022-12-31T21:00:00.000Z"]).iloc[0], "2023-01-01")

    def test_wrong_month_start_rejected(self):
        with self.assertRaises(ValueError):
            canonical_period(["2023-01-31"])

    def test_composition_scale_invariance_and_residual(self):
        f = pd.DataFrame([[10]*5+[100], [100]*5+[1000]], columns=CATEGORIES+[TOTAL])
        x = profile_matrix(f)
        np.testing.assert_allclose(x[0], x[1])
        self.assertEqual(x.shape, (2,6))
        np.testing.assert_allclose((x*x).sum(axis=1), .5)

    def test_parent_not_summed_and_negative_residual_rejected(self):
        f = pd.DataFrame([[25]*5+[100]], columns=CATEGORIES+[TOTAL])
        with self.assertRaises(ValueError): profile_matrix(f)

    def test_graph_symmetry_no_loops_and_union_degree(self):
        a, info = knn_graph(np.array([[0.],[1.],[3.],[8.]]), k=1)
        self.assertEqual((a-a.T).nnz, 0)
        self.assertTrue(np.all(a.diagonal()==0))
        self.assertTrue(np.all(a.getnnz(axis=1)>=1))
        self.assertGreater(info["sigma"],0)

    def test_identical_features_have_finite_tie_behavior(self):
        a, _ = knn_graph(np.zeros((4,2)), k=1)
        self.assertTrue(np.isfinite(a.data).all())
        self.assertTrue(np.all(a.data == 1))
        self.assertEqual(a.diagonal().sum(),0)

    def test_network_hand_computed_path(self):
        a = csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
        m = network_indices(a,[0,0,1,1])
        self.assertAlmostEqual(m["AVI"],2/3)
        self.assertAlmostEqual(m["AVU"],1)

    def test_disconnected_network_undefined_is_explicit(self):
        m=network_indices(csr_matrix([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]),[0,0,1,1])
        self.assertEqual(m["AVI"],1)
        self.assertIsNone(m["AVU"])
        self.assertEqual(m["AVU_undefined_ordered_pairs"],2)

    def test_sdbw_separated_point_masses(self):
        self.assertEqual(s_dbw(np.array([[0.],[0.],[10.],[10.]]),[0,0,1,1]),0)

    def test_sdbw_empty_balls_not_silent_zero(self):
        with self.assertRaises(ValueError): s_dbw(np.array([[0.],[1.],[9.],[10.]]),[0,0,1,1])

    def test_label_permutation_not_transition(self):
        result=transitions(['a','b','c','d'],[0,0,1,1],['d','b','a','c'],[7,4,4,7])
        self.assertEqual(result['ARI'],1)
        self.assertEqual(result['matched_churn'],0)

    def test_split_preserves_many_to_many_accounting(self):
        result=transitions(['a','b','c','d'],[0,0,0,0],['a','b','c','d'],[1,1,2,2])
        self.assertEqual(len(result['events']),2)
        self.assertEqual(sum(x['count'] for x in result['events']),4)

    def test_future_outcome_does_not_select_neighbors(self):
        x=np.array([[0.],[1.],[3.]])
        a=heldout_peer_concordance(x,[1,2,3],k=1)
        b=heldout_peer_concordance(x,[999,-1,20],k=1)
        np.testing.assert_array_equal(a['neighbors'],b['neighbors'])
        self.assertTrue(np.all(a['neighbors'].ravel()!=np.arange(3)))

    def test_model_execution_requires_both_switches(self):
        for flag,cfg_flag in [(False,False),(True,False),(False,True)]:
            with self.assertRaises(PermissionError):
                require_execution({'execution':{'allow_clustering':cfg_flag}},execute=flag)

    def test_disabled_fit_never_calls_estimator(self):
        with patch('sklearn.cluster.KMeans.fit_predict',side_effect=AssertionError('fit called')):
            with self.assertRaises(PermissionError):
                fit_static('kmeans',np.ones((3,1)),None,[],{'execution':{'allow_clustering':False}},execute=False)


    def test_registry_interval_is_end_exclusive(self):
        from sbercluster.canonical import active_registry
        observations=pd.DataFrame({'territory_id':[1,1], 'period':['2023-12-01','2024-01-01']})
        registry=pd.DataFrame({'territory_id':[1,1], 'year_from':[2018,2024], 'year_to':[2024,9999], 'name':['old','new']})
        got=active_registry(observations,registry)
        self.assertEqual(got.name.tolist(),['old','new'])

    def test_registry_overlap_is_not_silently_deduplicated(self):
        from sbercluster.canonical import active_registry
        observations=pd.DataFrame({'territory_id':[1], 'period':['2024-01-01']})
        registry=pd.DataFrame({'territory_id':[1,1], 'year_from':[2018,2024], 'year_to':[9999,9999]})
        with self.assertRaises(ValueError): active_registry(observations,registry)

    def test_future_values_do_not_change_development_scaling(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/pilot.json').read_text(encoding='utf-8'))
        rows=[]
        for year in [2023,2024]:
            for entity,base in [('a',1),('b',2),('c',4)]:
                row={'entity_id':entity,'period':f'{year}-12-01',TOTAL:100.0}
                row.update({c:float((j+1)*base) for j,c in enumerate(CATEGORIES)})
                rows.append(row)
        frame=pd.DataFrame(rows)
        slices_a,scale_a=make_slices(frame,cfg)
        frame.loc[frame.period.str.startswith('2024'),CATEGORIES]=70.0
        slices_b,scale_b=make_slices(frame,cfg)
        np.testing.assert_allclose(slices_a[0][2],slices_b[0][2])
        self.assertEqual(scale_a,scale_b)


if __name__ == '__main__': unittest.main()
