import gzip
import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sbercluster.round2 import (same_region_graph, matched_change, external_comparison,
                               transition_votes, validate_temporal_reuse, load_temporal_reuse)
from sbercluster.published_inputs import load_published, ARCHIVE
from scipy.sparse import csr_matrix,save_npz


class RoundTwoChecks(unittest.TestCase):
    def test_published_baselines_preserve_string_ids_and_reject_malformed_partitions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ids=[f'{i:03}' for i in range(8)];z2=[0]*4+[1]*4;z4=[0,0,1,1,2,2,3,3]
            cfg={'features':{},'graph':{},'seed':1729,'followup':{
                'expected_road_sha256':'road','alpha':.25,'max_sweeps':50,'ward_archive':'reports/ward.csv'}}
            road=root/'reports/graphs/road_2024_2016.npz';road.parent.mkdir(parents=True)
            save_npz(road,csr_matrix((8,8)))
            cfg['followup']['expected_transport_graph_sha256']=hashlib.sha256(road.read_bytes()).hexdigest()
            ward=root/'reports/ward.csv'
            pd.DataFrame({'entity_id':ids,'representation':['annual_2023']*8,'candidate':['ward_k4']*8,'cluster':z4}).to_csv(ward,index=False)
            cfg['followup']['expected_ward_archive_sha256']=hashlib.sha256(ward.read_bytes()).hexdigest()
            base='review-20260924-small-k';joint='review-20260924-joint'
            directory=root/ARCHIVE/base;directory.mkdir(parents=True)
            (directory/'manifest.json').write_text(json.dumps({'source_files_sha256':{'labels.json':'baseline'}}),encoding='utf-8')
            joint_directory=root/ARCHIVE/joint;joint_directory.mkdir(parents=True)
            (joint_directory/'manifest.json').write_text('{}',encoding='utf-8')
            # The common metadata is already hash-checked by read_archive; exercise its consumers.
            common={'config':deepcopy(cfg),'ids':ids,'panel_sha256':'panel'}
            jp={**deepcopy(common),'baseline_labels_sha256':'baseline','baseline_provenance_sha256':'base-prov',
                'road_sha256':'road','road_date':'2024-12-31'}
            bp={**deepcopy(common),'source_provenance_sha256':'base-prov'}
            objects={(joint,'provenance.json'):jp,(base,'provenance.json'):bp,
                (joint,'labels.json.gz'):{'ids':ids,'labels':{'joint_alpha0_k4':z4,'joint_road_k4':z4,'joint_road_k2':z2}},
                (base,'labels.json.gz'):{'ids':ids,'labels':{'kmeans_k2':z2,'ward_k2':z2}},
                (joint,'status.json'):{'status':'completed','stage':'joint'},
                (base,'status.json'):{'status':'completed','stage':'small_k'},(joint,'graphs.json'):{'transport':{}}}
            with patch('sbercluster.published_inputs.read_archive',side_effect=lambda r,n,f:objects[n,f]):
                _,_,initial,_,_=load_published(root,cfg,ids,{'panel_sha256':'panel'})
                np.testing.assert_array_equal(initial[4]['ward'],z4)
                for bad in [z2[:-1],[float(v) for v in z2],[0]*8]:
                    objects[base,'labels.json.gz']['labels']['kmeans_k2']=bad
                    with self.assertRaisesRegex(ValueError,'partition'):load_published(root,cfg,ids,{'panel_sha256':'panel'})
                objects[base,'labels.json.gz']['labels']['kmeans_k2']=z2
                objects[joint,'status.json']['stage']='small_k'
                with self.assertRaises(ValueError):load_published(root,cfg,ids,{'panel_sha256':'panel'})

    def test_region_prior_has_no_cross_region_edges_and_equal_strength(self):
        r=np.array(['a','a','b','b','b','singleton']);g=same_region_graph(r)
        self.assertEqual((g-g.T).nnz,0);self.assertTrue(np.all(g.diagonal()==0))
        np.testing.assert_allclose(g.sum(axis=1).A1,[1,1,1,1,1,0]);self.assertEqual(g[:2,2:].nnz,0)

    def test_relabeling_does_not_count_as_transition(self):
        a=[0,0,1,1,2];b=[8,8,5,5,9]
        change,ambiguous=matched_change(a,b,return_ambiguity=True)
        self.assertFalse(change.any());self.assertFalse(ambiguous.any())
        self.assertEqual(matched_change([0,0,1,1],[4,5,5,5]).sum(),1)

    def test_matching_ties_are_label_invariant_and_flagged(self):
        a=[0,0,1,1];b=[0,1,0,1]
        change,ambiguous=matched_change(a,b,return_ambiguity=True)
        renamed,renamed_ambiguity=matched_change([80,80,-3,-3],[8,2,8,2],return_ambiguity=True)
        np.testing.assert_array_equal(change,renamed)
        np.testing.assert_array_equal(ambiguous,renamed_ambiguity)
        self.assertTrue(ambiguous.all());self.assertEqual(change.sum(),2)

    def test_ambiguity_is_local_and_never_a_unanimous_vote(self):
        a=[0,0,1,1,2,2];b=[0,1,0,1,2,2]
        change,ambiguous=matched_change(a,b,return_ambiguity=True)
        np.testing.assert_array_equal(ambiguous,[True,True,True,True,False,False])
        flags=np.stack([change[None,:]]*3);uncertain=np.stack([ambiguous[None,:]]*3)
        votes,counts,unanimous=transition_votes(flags,uncertain)
        self.assertEqual(votes.sum(),0);self.assertEqual(counts.sum(),12);self.assertFalse(unanimous.any())
        flags[:,:,4]=True
        votes,counts,unanimous=transition_votes(flags,uncertain)
        self.assertTrue(unanimous[0,4]);self.assertEqual(votes[0,4],3)

    @staticmethod
    def bridge(root,constant=False):
        rng=np.random.default_rng(3);n=80;ids=['tid_'+str(i) for i in range(n)]
        frame=pd.DataFrame({'entity_id':ids,'market_access':np.ones(n) if constant else rng.normal(size=n),
            'region_code':np.repeat(np.arange(8),10),'municipal_district_type':['A']*n,
            'log_expense_2023':rng.normal(size=n),'intracity_moscow_petersburg':np.zeros(n,bool)})
        directory=root/'reports/external-v4';directory.mkdir(parents=True)
        frame.to_csv(directory/'municipality_bridge.csv',index=False)
        return ids

    def test_paired_external_delta_invariant_to_arbitrary_cluster_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=self.bridge(root);z=np.arange(len(ids))%2
            rows=external_comparison(root,ids,{'base':z,'renamed':100-7*z},{2:'base'},20)['results']
        for row in rows:
            self.assertAlmostEqual(row['delta_partial_R2'],0,places=10)
            np.testing.assert_allclose(row['delta_ci95'],[0,0],atol=1e-10)

    def test_requested_k_keeps_baseline_when_only_two_clusters_are_occupied(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=self.bridge(root);n=len(ids)
            parts={'base2':np.arange(n)%2,'base4':np.arange(n)%4,
                   'dmon_k4_seed17':np.arange(n)%2,'explicit':np.arange(n)%2}
            rows=external_comparison(root,ids,parts,{2:'base2',4:'base4'},5,
                                     requested_ks={'explicit':4})['results']
        for row in rows:
            if row['id'] in ('dmon_k4_seed17','explicit'):
                self.assertEqual(row['baseline'],'base4');self.assertEqual(row['requested_k'],4)
                self.assertEqual(row['occupied_k'],2)
                self.assertEqual(row['partition_status'],'fewer_occupied_than_requested')

    def test_undefined_bootstrap_draws_use_the_same_paired_mask(self):
        # Per subset: base observed + 3 draws, then candidate observed + 3 draws.
        returns=[.3,.1,None,.3,.4,None,.2,.5]*2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=self.bridge(root);z=np.arange(len(ids))%2
            with patch('scripts.external_validation_v4.partial_r2',side_effect=returns):
                rows=external_comparison(root,ids,{'base':z,'candidate':z},{2:'base'},3)['results']
        for row in rows:
            self.assertEqual(row['valid_bootstrap_draws'],2)
            if row['id']=='candidate':
                self.assertEqual(row['valid_paired_bootstrap_draws'],1)
                np.testing.assert_allclose(row['delta_ci95'],[.2,.2])
                self.assertAlmostEqual(row['delta_partial_R2'],.1)

    def test_all_undefined_draws_are_serializable_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ids=self.bridge(root,constant=True);z=np.arange(len(ids))%2
            result=external_comparison(root,ids,{'base':z},{2:'base'},3)
        json.dumps(result,allow_nan=False)
        for row in result['results']:
            self.assertIsNone(row['partial_R2']);self.assertIsNone(row['delta_partial_R2'])
            self.assertIsNone(row['ci95']);self.assertIsNone(row['delta_ci95'])
            self.assertEqual(row['valid_bootstrap_draws'],0);self.assertEqual(row['valid_paired_bootstrap_draws'],0)

    @staticmethod
    def reuse_fixture():
        ids=['a','b','c','d'];periods=['2023-01-01','2023-02-01'];scaler={'scale':[1.,2.]}
        cfg={'seed':1729,'features':{'mode':'test'},'graph':{'k':15},
             'clustering':{'leiden_iterations':4,'resolution':.7},
             'validation':{'development_end':'2023-12-01'},
             'followup':{'expected_road_sha256':'road-sha','month_indices':[0,1],
                         'attribute_graph_weight':.5,'graph_layer':'attribute_road'}}
        prior={'config':deepcopy(cfg),'ids':ids,'panel_sha256':'panel-sha','scaler':scaler,
               'road_sha256':'road-sha','road_date':'2024-12-31','attribute_graph_weight':.5}
        old={'ids':ids,'periods':periods,'labels':[[0,0,1,1],[0,1,1,1]],
             'summary':{'omega_relative':.1,'months':2},
             'monthly_metrics':[{'period':p,'SW':.2} for p in periods]}
        status={'status':'completed','stage':'temporal','n':4};manifest={'panel_sha256':'panel-sha'}
        return old,prior,status,cfg,ids,periods,manifest,scaler

    def test_temporal_reuse_rejects_changed_scientific_inputs(self):
        fixture=self.reuse_fixture()
        validate_temporal_reuse(*fixture,.1,1729)
        migrated=list(deepcopy(fixture));migrated[7]=deepcopy(migrated[7]);migrated[7]['level_weight']=0.
        validate_temporal_reuse(*migrated,.1,1729)
        migrated[7]['level_weight']=.2
        with self.assertRaises(ValueError):validate_temporal_reuse(*migrated,.1,1729)
        mutations=[(0,lambda x:x.update(periods=['wrong','periods'])),
                   (0,lambda x:x.update(labels=[[0.,0.,1.,1.],[0.,1.,1.,1.]])),
                   (0,lambda x:x.update(labels=[[[0],[0],[1],[1]],[[0],[1],[1],[1]]])),
                   (1,lambda x:x.update(panel_sha256='changed')),
                   (1,lambda x:x.update(road_sha256='changed')),
                   (1,lambda x:x.update(attribute_graph_weight=.7)),
                   (1,lambda x:x['config'].update(data_contract={'identity_mode':'unique_name_provisional'})),
                   (1,lambda x:x.update(scaler={'scale':[9.,9.]})),
                   (1,lambda x:x['config']['clustering'].update(leiden_iterations=8)),
                   (2,lambda x:x.update(status='running'))]
        for index,change in mutations:
            args=list(deepcopy(fixture));change(args[index])
            with self.assertRaises(ValueError):validate_temporal_reuse(*args,.1,1729)

    def test_temporal_reuse_reads_raw_and_manifest_checked_gzip(self):
        for public in [False,True]:
            old,prior,status,cfg,ids,periods,manifest,scaler=self.reuse_fixture()
            cfg['followup']['published_inputs']=public
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);name='review-20260924-temporal'
                directory=root/('reports/review-2026-09-24' if public else 'runs')/name
                directory.mkdir(parents=True);filename='temporal_omega0.1.json'+('.gz' if public else '')
                objects={filename:old,'provenance.json':prior,'status.json':status};hashes={}
                for file,value in objects.items():
                    raw=json.dumps(value).encode();raw=gzip.compress(raw) if file.endswith('.gz') else raw
                    (directory/file).write_bytes(raw);hashes[file]=hashlib.sha256(raw).hexdigest()
                (directory/'manifest.json').write_text(json.dumps({'files_sha256':hashes}))
                labels,info,metrics=load_temporal_reuse(root,cfg,ids,periods,manifest,scaler,.1,1729)
                np.testing.assert_array_equal(labels,old['labels'])
                self.assertEqual(info['source_sha256'],hashes[filename]);self.assertTrue(info['reused'])
                self.assertEqual(metrics,old['monthly_metrics'])


if __name__=='__main__':
    unittest.main()
