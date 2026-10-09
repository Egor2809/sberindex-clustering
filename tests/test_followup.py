import json
import gzip
import runpy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from scipy.sparse import csr_matrix
from sbercluster.followup import permute_within_groups, run, load_stability_inputs


class FollowupTests(unittest.TestCase):
    def test_permutation_preserves_region_blocks_and_weight_multiset(self):
        a = csr_matrix(np.arange(36).reshape(6,6))
        regions = np.array(['a','a','a','b','b','b'])
        b, order = permute_within_groups(a, regions, 14)
        np.testing.assert_array_equal(regions[order], regions)
        np.testing.assert_array_equal(b.toarray(), a.toarray()[order][:,order])
        np.testing.assert_array_equal(np.sort(b.data), np.sort(a.data))
        _, again = permute_within_groups(a, regions, 14)
        np.testing.assert_array_equal(order, again)

    def test_missing_strata_rejected(self):
        with self.assertRaises(ValueError):
            permute_within_groups(csr_matrix((2,2)), ['a',None], 0)

    def test_existing_result_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'existing').mkdir()
            marker=root/'existing'/'status.json'
            marker.write_text('original')
            cfg={'execution':{'allow_clustering':True},
                 'followup':{'stage':'small_k','output':'existing'}}
            with self.assertRaises(FileExistsError):
                run(cfg, root)
            self.assertEqual(marker.read_text(), 'original')

    def test_execution_gate_precedes_data_loading(self):
        with self.assertRaises(PermissionError):
            run({'execution':{'allow_clustering':False}}, Path('.'))

    def test_joint_phase_reuses_baselines_and_counts_only_new_fits(self):
        repo = Path(__file__).resolve().parents[1]
        cfg = json.loads((repo/'configs/followup_joint.json').read_text('utf-8'))
        n = 8
        x = np.arange(n*5, dtype=float).reshape(n,5)
        ids = [str(i) for i in range(n)]
        reference = np.repeat(np.arange(4), 2)
        a = csr_matrix(np.eye(n, k=1)+np.eye(n, k=-1))
        slices = [('2023-01-01', ids, x)]
        inputs = (slices, np.stack([x]*24), x, ids, np.repeat(['a','b'],4),
                  reference, {}, {'panel_sha256':'test-panel'})
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            for path in ['scripts/run_bounded.py', 'scripts/run_contest.py',
                         'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json',
                         'artifacts/sources/acquisition/hackathonlicence/connection.parquet']:
                target = root/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test-only')
            base = root/cfg['followup']['baseline_dir']
            base.mkdir(parents=True)
            (base/'status.json').write_text(json.dumps({'status':'completed','stage':'small_k'}))
            (base/'provenance.json').write_text(json.dumps({'panel_sha256':'test-panel','config':cfg,'ids':ids}))
            (base/'labels.json').write_text(json.dumps({'ids':ids,'labels':{
                'kmeans_k2':[0,0,0,0,1,1,1,1], 'kmeans_k3':[0,0,0,1,1,1,2,2],
                'ward_k2':[0,0,0,1,1,1,1,1]}}))
            stack.enter_context(patch('sbercluster.followup.load_inputs', return_value=inputs))
            stack.enter_context(patch('sbercluster.followup.knn_graph', return_value=(a,{})))
            road_reader = stack.enter_context(patch('sbercluster.followup.read_road_graph', return_value=(a,{}, {'as_of':'2024-12-31'})))
            stack.enter_context(patch('sbercluster.followup.subprocess.check_output', return_value='test-commit'))
            stack.enter_context(patch('sbercluster.followup.all_metrics', side_effect=lambda x,z,g:
                {'SW':.1,'k':len(np.unique(z)), 'min_cluster_size':int(np.bincount(z).min())}))
            stack.enter_context(patch('sbercluster.followup.network_indices', return_value={}))
            fit = stack.enter_context(patch('sbercluster.followup.fit_candidate', return_value=(reference,{})))
            out = run(cfg, root)
            summary = json.loads((out/'summary.json').read_text('utf-8'))
            labels = json.loads((out/'labels.json').read_text('utf-8'))['labels']
            self.assertEqual(summary['new_static_partitions'],8)
            self.assertEqual(len(labels),11)
            self.assertEqual(fit.call_count,1)  # only the road-only spectral control
            self.assertEqual(labels['joint_alpha0_k4'],reference.tolist())
            provenance=json.loads((out/'provenance.json').read_text('utf-8'))
            self.assertEqual(provenance['road_date'],'2024-12-31')
            self.assertIn('baseline_labels_sha256',provenance)
            exporter=runpy.run_path(str(repo/'scripts/export_followup.py'))
            destination=exporter['export_run'](out, root/'export')
            with gzip.open(destination/'labels.json.gz','rt',encoding='utf-8') as stream:
                self.assertEqual(json.load(stream)['labels'],labels)
            public=json.loads((destination/'provenance.json').read_text('utf-8'))
            self.assertNotIn('source_dir',public['config']['followup'])
            self.assertNotIn('threads',public['config']['execution'])
            self.assertNotIn('scripts/run_bounded.py',public['implementation'])
            exported=json.loads((destination/'manifest.json').read_text('utf-8'))
            for name,expected in exported['files_sha256'].items():
                self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(),expected)
            with self.assertRaises(FileExistsError):
                exporter['export_run'](out,root/'export')
            # Reuse exactly the eight-vertex joint output above; never read parquet again.
            stable = json.loads((repo/'configs/followup_stability.json').read_text('utf-8'))
            archive = root/stable['followup']['ward_archive']
            archive.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(archive, 'wt', encoding='utf-8') as stream:
                stream.write('entity_id,representation,candidate,cluster\n')
                for entity, label in zip(ids,[0,0,1,1,2,3,3,3]):
                    stream.write(f'{entity},annual_2023,ward_k4,{label}\n')
            hashfile = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            stable['followup']['expected_road_sha256'] = provenance['road_sha256']
            stable['followup']['expected_transport_graph_sha256'] = hashfile(out/'transport_graph.npz')
            stable['followup']['expected_ward_archive_sha256'] = hashfile(archive)
            stability_out = run(stable,root)
            stability_rows = json.loads((stability_out/'metrics.json').read_text('utf-8'))
            self.assertEqual(len(stability_rows),8)
            self.assertEqual(json.loads((stability_out/'summary.json').read_text('utf-8'))['new_static_partitions'],8)
            self.assertEqual(road_reader.call_count,1)
            self.assertEqual(fit.call_count,1)
            self.assertEqual({r['k'] for r in stability_rows},{2,4})
            self.assertTrue(all('ARI_original_joint' in r and 'ARI_frozen_k4' in r for r in stability_rows))
            expected={(k,init,seed) for k in [2,4] for init,seeds in
                      [('kmeans',[2718,3141]),('ward',[1729,2718])] for seed in seeds}
            self.assertEqual({(r['k'],r['initialization'],r['vertex_order_seed']) for r in stability_rows},expected)
            exported_stability=exporter['export_run'](stability_out,root/'export')
            public=json.loads((exported_stability/'provenance.json').read_text('utf-8'))
            self.assertIn('transport_graph_sha256',public)
            self.assertNotIn('joint_dir',public['config']['followup'])
            for key in ['expected_transport_graph_sha256','expected_road_sha256','expected_ward_archive_sha256']:
                bad=json.loads(json.dumps(stable))
                bad['followup'][key]='invalid'
                with self.assertRaises(ValueError):
                    load_stability_inputs(root,bad,ids,{'panel_sha256':'test-panel'})
            with self.assertRaises(ValueError):
                load_stability_inputs(root,stable,list(reversed(ids)),{'panel_sha256':'test-panel'})
            with self.assertRaises(ValueError):
                load_stability_inputs(root,stable,ids,{'panel_sha256':'another-panel'})



    def test_configs_define_bounded_controlled_design(self):
        root=Path(__file__).resolve().parents[1]
        joint=json.loads((root/'configs/followup_joint.json').read_text('utf-8'))
        self.assertEqual(len(joint['followup']['permutation_seeds']),3)
        self.assertNotIn('resource_profile',joint['execution'])
        for stage in ['small_k','joint','temporal','temporal_pilot','stability']:
            cfg=json.loads((root/f'configs/followup_{stage}.json').read_text('utf-8'))
            self.assertTrue(cfg['execution']['allow_clustering'])
            self.assertTrue(cfg['followup']['output'].startswith('runs/review-20260924-'))


if __name__ == '__main__':
    unittest.main()
