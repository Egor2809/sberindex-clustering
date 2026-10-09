import unittest
import numpy as np
from sbercluster.contest_analysis import drift_analysis,transition_summary
from sbercluster.models import fit_temporal
from sbercluster.dynamics import transitions


class DynamicsTests(unittest.TestCase):
    def test_drift_prototypes_are_finite_and_share_the_feature_space(self):
        x=np.zeros((24,4,2));labels=np.array([0,0,1,1])
        for centers in (np.array([[np.nan,0],[1,1]]),np.zeros((2,1)),np.empty((0,2))):
            with self.subTest(shape=centers.shape),self.assertRaises(ValueError):
                drift_analysis(x,centers,labels)
        with self.assertRaises(ValueError):
            drift_analysis(x,np.zeros((2,2)),[0,0,1])

    def test_drift_matrix_and_counts_use_the_supplied_prototype_count(self):
        for k in (2,5):
            centers=np.arange(k,dtype=float)[:,None]
            labels=np.repeat(np.arange(k),2)
            x=np.tile(centers[labels][None,:,:],(24,1,1))
            result,raw,relative=drift_analysis(x,centers,labels)
            self.assertEqual(np.shape(result['raw']['transition_matrix']),(k,k))
            self.assertEqual(len(result['monthly_counts'][0]),k)
            np.testing.assert_array_equal(raw,labels)
            np.testing.assert_array_equal(relative,labels)

    def test_transition_labels_are_not_silently_truncated_or_rounded(self):
        with self.assertRaises(ValueError):
            transitions(['a', 'b'], [0, 1, 2], ['a', 'b'], [0, 1])
        with self.assertRaises(ValueError):
            transitions(['a', 'b'], [0, 1], ['a', 'b'], [0, .5])
        with self.assertRaises(ValueError):
            transition_summary([0, .5], [0, 1])

    def test_disjoint_cohorts_still_account_for_entries_and_exits(self):
        result = transitions(['a', 'b'], [0, 1], ['c'], [0])
        self.assertEqual((result['common_n'], result['entered'], result['exited']), (0, 1, 2))
        self.assertIsNone(result['ARI'])

    def test_common_drift_removal_recovers_original_prototypes(self):
        centers=np.array([[0.,0.],[2.,0.],[0.,2.],[2.,2.]])
        base=np.repeat(centers,3,axis=0)
        months=np.tile(base,(12,1,1))
        x=np.concatenate([months,months+np.array([2.,0.])])
        before=np.repeat(np.arange(4),3)
        result,raw,relative=drift_analysis(x,centers,before)
        self.assertGreater(result['raw']['changed'],0)
        self.assertEqual(result['relative']['changed'],0)
        np.testing.assert_array_equal(relative,before)

    def test_fixed_prototype_flow_conserves_population(self):
        r=transition_summary([0,0,1,2],[1,0,1,3])
        self.assertEqual(np.array(r['transition_matrix']).sum(),4)
        self.assertEqual(r['changed'],2)

    def test_unknown_boundaries_do_not_silently_pass_default_gate(self):
        cfg={'execution':{'allow_clustering':True},'data_contract':{'stable_territories_verified':False}}
        with self.assertRaises(ValueError):
            fit_temporal([],[],cfg,execute=True)

    def test_descriptive_gate_still_requires_aligned_ids(self):
        cfg={'execution':{'allow_clustering':True},'data_contract':{'stable_territories_verified':False,
            'temporal_scope':'retrospective_constant_id_sensitivity','boundary_review_evidence':['registry']}}
        with self.assertRaises(ValueError):
            fit_temporal([('2023-01',['a','b'],None),('2023-02',['b','a'],None)],[None,None],cfg,execute=True)
