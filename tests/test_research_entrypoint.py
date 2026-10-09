import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts import run_contest
from sbercluster import research, followup, round2
from sbercluster.input_contracts import validate_prepared_panel


PATH = Path(__file__).resolve().parents[1] / 'scripts/run_research.py'
SPEC = importlib.util.spec_from_file_location('run_research', PATH)
RUN_RESEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_RESEARCH)


class ResearchEntrypoint(unittest.TestCase):
    def test_prepared_panel_identity_and_manifest_contract(self):
        panel=pd.DataFrame({'entity_id':['tid_1','tid_2','tid_1','tid_2'],
                            'territory_id':['01',2,'1',2],
                            'period':['2023-01-01']*2+['2023-02-01']*2})
        manifest={'identity_mode':'source_territory_id','n_entities':2,
                  'months':['2023-01-01','2023-02-01']}
        cfg={'data_contract':{'identity_mode':'source_territory_id'}}
        validate_prepared_panel(panel,manifest,cfg)
        mutations=[lambda p,m:m.update(identity_mode='unique_name_provisional'),
                   lambda p,m:m.update(n_entities=3),lambda p,m:m.update(months=['2023-01-01']),
                   lambda p,m:p.__setitem__('entity_id',['name_a','tid_2','tid_1','tid_2']),
                   lambda p,m:p.__setitem__('territory_id',['1.5',2,1,2]),
                   lambda p,m:p.__setitem__('territory_id',[True,2,1,2]),
                   lambda p,m:p.__setitem__('period',['2023-01-02']*2+['2023-02-01']*2),
                   lambda p,m:p.__setitem__('period',['2023-01-01']*4)]
        for mutate in mutations:
            changed=panel.copy();contract=json.loads(json.dumps(manifest));mutate(changed,contract)
            with self.assertRaises(ValueError):validate_prepared_panel(changed,contract,cfg)
        # Preserve an explicitly declared noncanonical identity without pretending it is source ID.
        provisional=panel.drop(columns='territory_id').copy()
        provisional.entity_id=['name_a','name_b','name_a','name_b']
        declared={**manifest,'identity_mode':'unique_name_provisional'}
        validate_prepared_panel(provisional,declared,{'data_contract':{'identity_mode':'unique_name_provisional'}})

    def test_development_rejects_correctly_hashed_wrong_identity_before_features(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'data/processed';directory.mkdir(parents=True)
            panel=directory/'panel.csv';panel.write_text('entity_id,period\nname_a,2023-01-01\n',encoding='utf-8')
            (directory/'manifest.json').write_text(json.dumps({'panel_sha256':hashlib.sha256(panel.read_bytes()).hexdigest(),
                'identity_mode':'unique_name_provisional','n_entities':1,'months':['2023-01-01']}),encoding='utf-8')
            cfg={'data_contract':{'identity_mode':'source_territory_id'}}
            with patch.object(research,'make_slices') as features:
                with self.assertRaisesRegex(ValueError,'identity mode'):research.load_development(root,cfg)
                features.assert_not_called()

    def test_partition_equivalence_accepts_bijective_renaming(self):
        self.assertTrue(RUN_RESEARCH.partition_equivalent([0, 0, 1, 2], [8, 8, 4, 7]))

    def test_partition_equivalence_rejects_merge_or_split(self):
        self.assertFalse(RUN_RESEARCH.partition_equivalent([0, 0, 1], [4, 5, 6]))
        self.assertFalse(RUN_RESEARCH.partition_equivalent([0, 1, 2], [4, 4, 6]))

    def test_compare_labels_checks_ids_and_exact_numeric_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'labels').mkdir()
            (root / 'status.json').write_text(json.dumps({'status':'completed','phase':'screen'}), encoding='utf-8')
            (root / 'ids.json').write_text(json.dumps(['a', 'b', 'c']), encoding='utf-8')
            np.save(root / 'labels/annual_2023__kmeans_k4.npy', np.array([2, 2, 0]))
            published = root / 'published.csv'
            published.write_text('entity_id,cluster\na,2\nb,2\nc,0\n', encoding='utf-8')
            with patch('builtins.print'):
                self.assertEqual(RUN_RESEARCH.compare_labels(root, published), 0)

    def test_compare_rejects_duplicate_ids_malformed_labels_and_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'labels').mkdir()
            published=root/'published.csv'
            published.write_text('entity_id,cluster\na,0\nb,1\n',encoding='utf-8')
            for ids,labels,state in [(['a','a','b'],[0,0,1],'completed'),
                                     (['a','b'],[[0],[1]],'completed'),
                                     (['a','b'],[0.,1.],'completed'),
                                     (['a','b'],[0,1],'running')]:
                (root/'ids.json').write_text(json.dumps(ids),encoding='utf-8')
                (root/'status.json').write_text(json.dumps({'status':state,'phase':'screen'}),encoding='utf-8')
                np.save(root/'labels/annual_2023__kmeans_k4.npy',np.asarray(labels))
                with self.assertRaises(ValueError):RUN_RESEARCH.compare_labels(root,published)

    def test_portable_contest_runner_honors_threads_and_execution_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json'
            cfg={'execution':{'task':'followup_v4','allow_clustering':True,'threads':1}}
            path.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.dict(os.environ),patch('sbercluster.followup.run') as fit,patch('builtins.print'):
                self.assertEqual(run_contest.main(['--config',str(path),'--execute-clustering']),0)
                self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')
                self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'1')
                fit.assert_called_once()
            for value in [True,0,-1,1.5,'2']:
                cfg['execution']['threads']=value;path.write_text(json.dumps(cfg),encoding='utf-8')
                with self.assertRaises(ValueError):run_contest.main(['--config',str(path),'--execute-clustering'])
                with self.assertRaises(ValueError):self._invalid_research_threads(path,cfg)
            cfg['execution'].update(threads=1,allow_clustering=False)
            path.write_text(json.dumps(cfg),encoding='utf-8')
            with patch('sbercluster.followup.run') as fit:
                with self.assertRaises(PermissionError):run_contest.main(['--config',str(path),'--execute-clustering'])
                fit.assert_not_called()
            cfg['execution'].update(task='research_v2',allow_clustering=True)
            path.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.object(run_contest.subprocess,'call') as supervise:
                with self.assertRaises(ValueError):run_contest.main(['--config',str(path),'--execute-clustering','--supervisor','profile'])
                supervise.assert_not_called()
            cfg['execution']['task']='followup_v4';path.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.object(RUN_RESEARCH.subprocess,'call') as supervise:
                with self.assertRaises(ValueError):RUN_RESEARCH.main(['--config',str(path),'--execute-clustering','--supervisor','profile'])
                supervise.assert_not_called()

    @staticmethod
    def _invalid_research_threads(path,cfg):
        value=json.loads(json.dumps(cfg));value['execution']['task']='research_v2'
        path.write_text(json.dumps(value),encoding='utf-8')
        RUN_RESEARCH.run_direct(path)

    def test_failed_and_interrupted_stages_record_terminal_status(self):
        for module,stage in [(followup,'small_k'),(round2,'region_control')]:
            for exc,status in [(ValueError('bad input'),'failed'),(KeyboardInterrupt(),'interrupted')]:
                with tempfile.TemporaryDirectory() as folder:
                    root=Path(folder)
                    cfg={'execution':{'allow_clustering':True},'followup':{'stage':stage,'output':'run'}}
                    with patch.object(module,'load_inputs',side_effect=exc),patch('builtins.print'):
                        with self.assertRaises(type(exc)):module.run(cfg,root)
                    state=json.loads((root/'run/status.json').read_text('utf-8'))
                    self.assertEqual(state['status'],status);self.assertEqual(state['stage'],stage)
                    self.assertEqual(state['error_type'],type(exc).__name__)
                    with self.assertRaises(FileExistsError):module.run(cfg,root)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);cfg={'execution':{'allow_clustering':True},'research':{'phase':'screen'}}
            with patch.object(research,'load_development',return_value=(None,None,None,None,None,None)),\
                 patch.object(research,'create_run',return_value=out),patch.object(research,'screen',side_effect=RuntimeError('fit failed')):
                with self.assertRaises(RuntimeError):research.run(cfg,out)
            self.assertEqual(json.loads((out/'status.json').read_text('utf-8'))['status'],'failed')


if __name__ == '__main__':
    unittest.main()
