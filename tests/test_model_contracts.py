"""Analytical API checks and tiny backend fixtures; no full model series."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from sbercluster.cli import run
from sbercluster.contest import metric_row
from sbercluster.io import sha256
from sbercluster.models import _igraph, fit_static, fit_temporal, require_execution


class ModelContracts(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.], [1.], [9.], [10.]])
        self.ids = ['a', 'b', 'c', 'd']
        self.z = np.array([0, 0, 1, 1])
        self.a = csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=float)
        self.cfg = {'execution': {'allow_clustering': True}, 'seed': 17,
                    'data_contract': {'stable_territories_verified': True},
                    'validation': {'development_end': '2023-12-01'},
                    'clustering': {'resolution': 1., 'leiden_iterations': 4,
                                   'temporal_coupling_relative': .1, 'n_clusters': 2}}
        self.slices = [('2023-01-01', self.ids, self.x), ('2023-02-01', self.ids, self.x)]

    def test_execution_gate_rejects_truthy_non_boolean_switch(self):
        for value in ('false', 1, [], None):
            with self.subTest(value=value), self.assertRaises(PermissionError):
                require_execution(self.cfg, value)

    def test_igraph_does_not_silently_drop_invalid_lower_triangle(self):
        for graph in (csr_matrix([[0,0],[7,0]]), csr_matrix([[0,np.inf],[np.inf,0]]),
                      csr_matrix([[1,2],[2,0]]), csr_matrix([[0,-1],[-1,0]])):
            with self.subTest(graph=graph.toarray()), self.assertRaises(ValueError):
                _igraph(graph, ['a','b'])
        graph = _igraph(csr_matrix([[0,7],[7,0]]), ['a','b'])
        self.assertEqual(graph.ecount(), 1)
        self.assertEqual(graph.es['weight'], [7.])
        self.assertEqual(graph.vs['id'], ['a','b'])

    def test_static_shapes_and_graph_are_validated_before_backend(self):
        cases = [(self.x[:3], self.a, self.ids), (self.x, self.a[:3,:3], self.ids),
                 (self.x, self.a, ['a','a','c','d']), (self.x, self.a, ['a',None,'c','d']),
                 (self.x, self.a, ['a',pd.NA,'c','d']), (self.x, self.a, ['a',np.datetime64('NaT'),'c','d']),
                 (np.empty((4,0)), self.a, self.ids)]
        with patch('sklearn.cluster.KMeans.fit_predict') as fit:
            for x, graph, ids in cases:
                with self.subTest(shape=x.shape,ids=ids), self.assertRaises(ValueError):
                    fit_static('kmeans', x, graph, ids, self.cfg, True)
            fit.assert_not_called()

    def test_static_real_small_fit_preserves_inputs_and_seed(self):
        old = self.a.copy()
        first = fit_static('kmeans',self.x,self.a,self.ids,self.cfg,True)
        second = fit_static('kmeans',self.x,self.a,self.ids,self.cfg,True)
        np.testing.assert_array_equal(first,second)
        self.assertEqual((self.a != old).nnz,0)
        self.assertEqual(len(np.unique(first)),2)

    def test_temporal_calendar_is_canonical_ordered_and_consecutive(self):
        for periods in [('2023-01-01','2023-03-01'), ('2023-02-01','2023-01-01'),
                        ('2023-01-01','2023-01-01'), ('2023-01','2023-02'),
                        ('2023-01-02','2023-02-01'), ('2023-13-01','2024-01-01')]:
            slices = [(p,self.ids,self.x) for p in periods]
            with self.subTest(periods=periods), self.assertRaises(ValueError):
                fit_temporal(slices,[self.a,self.a],self.cfg,True)

    def test_temporal_features_and_graphs_match_frozen_vertex_space(self):
        cases = [[self.x,self.x[:3]], [self.x,np.column_stack([self.x,self.x])],
                 [self.x,np.full_like(self.x,np.nan)], [self.x,np.empty((4,0))]]
        for features in cases:
            slices = [(p,ids,x) for (p,ids,_),x in zip(self.slices,features)]
            with self.subTest(shapes=[x.shape for x in features]), self.assertRaises(ValueError):
                fit_temporal(slices,[self.a,self.a],self.cfg,True)
        with self.assertRaises(ValueError):
            fit_temporal(self.slices,[self.a,self.a[:3,:3]],self.cfg,True)

    def test_temporal_invalid_weights_and_iterations_rejected(self):
        for field, values in [('temporal_coupling_relative',[-.1,np.nan,np.inf,True,'0.1']),
                              ('resolution',[-1,np.nan,True,'1']),
                              ('leiden_iterations',[0,1.5,True])]:
            for value in values:
                cfg=deepcopy(self.cfg);cfg['clustering'][field]=value
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    fit_temporal(self.slices,[self.a,self.a],cfg,True)

    def test_negative_iterations_are_legitimate_convergence_mode(self):
        cfg=deepcopy(self.cfg);cfg['clustering']['leiden_iterations']=-1
        with patch('leidenalg.find_partition_temporal',return_value=([self.z,self.z],.2)) as fit:
            _, info=fit_temporal(self.slices,[self.a,self.a],cfg,True)
            self.assertEqual(fit.call_args.kwargs['n_iterations'],-1)
            self.assertAlmostEqual(info['omega'],.15)

    def test_temporal_omega_uses_only_development_graphs(self):
        slices=[('2023-12-01',self.ids,self.x),('2024-01-01',self.ids,self.x)]
        with patch('leidenalg.find_partition_temporal',return_value=([self.z,self.z],0.)):
            _, normal=fit_temporal(slices,[self.a,self.a],self.cfg,True)
            _, shifted=fit_temporal(slices,[self.a,self.a*100],self.cfg,True)
        self.assertEqual(normal['omega'],shifted['omega'])

    def test_malformed_backend_memberships_cannot_be_silently_zipped(self):
        for memberships in ([self.z], [self.z,self.z[:3]], [self.z,self.z.astype(float)]):
            with patch('leidenalg.find_partition_temporal',return_value=(memberships,0.)):
                with self.assertRaises(ValueError):
                    fit_temporal(self.slices,[self.a,self.a],self.cfg,True)
        with patch('leidenalg.find_partition',return_value=SimpleNamespace(membership=self.z[:3])):
            with self.assertRaises(ValueError):
                fit_static('leiden_static',self.x,self.a,self.ids,self.cfg,True)

    def test_requested_candidate_metadata_cannot_overwrite_measured_metrics(self):
        measured=metric_row(self.x,self.z,self.a,{'id':'collapsed_k4','k':4,'SW':999},self.z)
        self.assertEqual(measured['k'],2)
        self.assertEqual(measured['requested_k'],4)
        self.assertNotEqual(measured['SW'],999)

    def cli_fixture(self, root, methods):
        panel_path=root/'data/processed/panel.csv';panel_path.parent.mkdir(parents=True)
        pd.DataFrame({'entity_id':self.ids,'period':['2023-01-01']*4}).to_csv(panel_path,index=False)
        (panel_path.parent/'manifest.json').write_text(json.dumps({'panel_sha256':sha256(panel_path)}))
        cfg=deepcopy(self.cfg);cfg['clustering']['methods']=methods
        cfg['graph']={'k':1,'symmetry':'union','bandwidth':'median_knn_distance'}
        stack=ExitStack()
        stack.enter_context(patch('sbercluster.cli.validate_contract'))
        stack.enter_context(patch('sbercluster.cli.validate_prepared_panel'))
        stack.enter_context(patch('sbercluster.cli.make_slices',return_value=(self.slices,{})))
        return cfg,stack

    def test_cli_failure_and_interrupt_are_saved_as_terminal_statuses(self):
        for failure,expected in [(ValueError('backend failed'),'failed'),(KeyboardInterrupt(),'interrupted')]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);cfg,stack=self.cli_fixture(root,['kmeans'])
                with stack, patch('sbercluster.cli.fit_static',side_effect=failure):
                    with self.assertRaises(type(failure)):
                        run(cfg,root,True)
                outputs=list((root/'runs').iterdir());self.assertEqual(len(outputs),1)
                status=json.loads((outputs[0]/'status.json').read_text('utf-8'))
                self.assertEqual(status['status'],expected)
                self.assertEqual(status['error_type'],type(failure).__name__)

    def test_cli_accepts_explicit_retrospective_scope_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,stack=self.cli_fixture(root,['leiden_temporal'])
            cfg['data_contract']={'stable_territories_verified':False,
                'temporal_scope':'retrospective_constant_id_sensitivity','boundary_review_evidence':['registry']}
            with stack,patch('sbercluster.cli.fit_temporal',return_value=([self.z,self.z],{})) as fit:
                result=run(cfg,root,True)
                fit.assert_called_once()
            self.assertEqual(json.loads((Path(result['run'])/'status.json').read_text())['status'],'completed')
