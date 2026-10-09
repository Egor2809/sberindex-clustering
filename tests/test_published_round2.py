"""Check that exported scientific evidence and the rendered atlas agree."""
import gzip,hashlib,json,unittest,tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from scripts import run_round2_stage, export_round2
from scripts.verify_analogue_case import verify
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/round2-2026-09-24'

class PublishedRoundTwo(unittest.TestCase):
    def test_all_evidence_and_scientific_source_hashes(self):
        manifests=list(REPORTS.glob('*/manifest.json'));self.assertGreaterEqual(len(manifests),4)
        for path in manifests:
            manifest=json.loads(path.read_text('utf-8'))
            for name,digest in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256((path.parent/name).read_bytes()).hexdigest(),digest,str(path/name))
            for digest in manifest['source_objects'].values():
                self.assertEqual(hashlib.sha256((REPORTS/'source_objects'/f'{digest}.py').read_bytes()).hexdigest(),digest)

    def test_analogue_case_matches_actual_atlas(self):
        self.assertEqual(verify()['verified_rows'],16)

    def test_completed_temporal_grid_has_every_registered_variant(self):
        directory=REPORTS/'round2-20260924-temporal_grid'
        if not directory.exists():self.skipTest('Full temporal evidence not published yet')
        summary=json.loads((directory/'temporal_summary.json').read_text('utf-8'))
        self.assertEqual({(r['omega'],r['seed']) for r in summary},{(w,s) for w in [0,.01,.03,.06,.1] for s in [1729,2718,3141]})
        self.assertEqual(len(summary),15)
        variants=list(directory.glob('temporal_omega*.gz'));self.assertEqual(len(variants),15)
        previous_ids=None
        for path in variants:
            data=json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(len(data['periods']),24);self.assertEqual(len(data['ids']),2016)
            if previous_ids is not None:self.assertEqual(data['ids'],previous_ids)
            previous_ids=data['ids'];self.assertEqual(len(data['labels']),24)
            self.assertTrue(all(len(row)==2016 for row in data['labels']))
            self.assertTrue(all(len(set(row))>=2 for row in data['labels']))


class StageReuseChecks(unittest.TestCase):
    @staticmethod
    def fixture(root):
        cfg={'schema_version':1,'seed':1,'data_contract':{'identity_mode':'source_territory_id'},
             'features':{'mode':'log_ratios_to_total'},'graph':{'k':1},'clustering':{},
             'validation':{'development_end':'2023-12-01'},
             'execution':{'task':'round2_v5','allow_clustering':True,'threads':1},
             'followup':{'stage':'region_control','output':'runs/sample','published_inputs':True}}
        panel=root/'data/processed/panel.csv';panel.parent.mkdir(parents=True)
        panel.write_text('entity_id,territory_id,period\ntid_1,1,2023-01-01\n',encoding='utf-8')
        panel_hash=hashlib.sha256(panel.read_bytes()).hexdigest()
        (panel.parent/'manifest.json').write_text(json.dumps({'panel_sha256':panel_hash}),encoding='utf-8')
        source=root/'sbercluster/round2.py';source.parent.mkdir();source.write_text('scientific source fixture',encoding='utf-8')
        h=hashlib.sha256(source.read_bytes()).hexdigest()
        run=root/cfg['followup']['output'];(run/'source_objects').mkdir(parents=True)
        (run/'source_objects'/f'{h}.py').write_bytes(source.read_bytes())
        values={'status.json':{'status':'completed','stage':'region_control','n':1},
                'summary.json':{'status':'completed','stage':'region_control','n':1},
                'provenance.json':{'config':deepcopy(cfg),'sources':{'sbercluster/round2.py':h},'panel_sha256':panel_hash},
                'graphs.json':{},'labels.json':{'ids':['tid_1'],'labels':{'base':[0]}},
                'metrics.json':[],'external_comparison.json':{}}
        for name,value in values.items():(run/name).write_text(json.dumps(value),encoding='utf-8')
        return cfg,run

    def test_reuse_checks_full_configuration_inputs_and_manifest_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,run=self.fixture(root)
            with patch('builtins.print'):out=export_round2.export(run,root/'reports/round2-2026-09-24')
            state=run_round2_stage.verify_completed(root,out,cfg,published=True)
            self.assertEqual(state['status'],'completed')
            mutations=[lambda c:c['data_contract'].update(identity_mode='unique_name_provisional'),
                       lambda c:c['validation'].update(development_end='2024-12-01'),
                       lambda c:c.update(schema_version=2),lambda c:c['execution'].update(task='followup_v4'),
                       lambda c:c['followup'].update(stage='dmon')]
            for change in mutations:
                bad=deepcopy(cfg);change(bad)
                with self.assertRaises(ValueError):run_round2_stage.verify_completed(root,out,bad,published=True)
            manifest=json.loads((out/'manifest.json').read_text('utf-8'))
            del manifest['files_sha256']['summary.json']
            (out/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'cover every'):run_round2_stage.verify_completed(root,out,cfg,published=True)
            with self.assertRaises(FileExistsError):export_round2.export(run,out.parent)
            panel=root/'data/processed/panel.csv';panel.write_text('changed input',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Prepared input'):run_round2_stage.verify_completed(root,run,cfg,published=False)

    def test_completed_raw_stage_exports_without_repeating_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,run=self.fixture(root);config=root/'config.json'
            config.write_text(json.dumps(cfg),encoding='utf-8')
            with patch.object(run_round2_stage,'ROOT',root),patch.object(run_round2_stage.subprocess,'run') as execute,patch('builtins.print'):
                self.assertEqual(run_round2_stage.main(['--config',str(config)]),0)
            execute.assert_called_once()
            self.assertTrue(execute.call_args.args[0][1].endswith('export_round2.py'))
            (run/'status.json').write_text(json.dumps({'status':'failed','stage':'region_control'}),encoding='utf-8')
            with patch.object(run_round2_stage,'ROOT',root),patch.object(run_round2_stage.subprocess,'run') as execute:
                with self.assertRaises(ValueError):run_round2_stage.main(['--config',str(config)])
                execute.assert_not_called()

    def test_export_failure_leaves_final_directory_retryable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,run=self.fixture(root);output=root/'export'
            (run/'labels.json').write_text('invalid json',encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):export_round2.export(run,output)
            self.assertFalse((output/run.name).exists())
            self.assertFalse(list(output.glob('.sample-*')))
            (run/'labels.json').write_text(json.dumps({'ids':['tid_1'],'labels':{'base':[0]}}),encoding='utf-8')
            with patch('builtins.print'):out=export_round2.export(run,output)
            self.assertTrue((out/'manifest.json').is_file())

    def test_export_rejects_completed_flag_without_required_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg,run=self.fixture(root);(run/'graphs.json').unlink()
            with self.assertRaisesRegex(ValueError,'evidence is incomplete'):export_round2.export(run,root/'export')
            self.assertFalse((root/'export'/run.name).exists())

    @staticmethod
    def change_stage(run,stage,**settings):
        for name in ('status.json','summary.json'):
            value=json.loads((run/name).read_text('utf-8'));value['stage']=stage
            (run/name).write_text(json.dumps(value),encoding='utf-8')
        value=json.loads((run/'provenance.json').read_text('utf-8'))
        value['config']['followup'].update(stage=stage,**settings)
        (run/'provenance.json').write_text(json.dumps(value),encoding='utf-8')

    def test_dmon_export_requires_every_model_and_training_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,run=self.fixture(root);output=root/'export'
            self.change_stage(run,'dmon',ks=[2,4],seeds=[17,18])
            with self.assertRaisesRegex(ValueError,'dmon_k'):export_round2.export(run,output)
            for k in (2,4):
                for seed in (17,18):
                    model=run/f'dmon_k{k}_seed{seed}';model.mkdir()
                    for name in ('best_model.npz','trace.json','info.json','labels.npy','assignments.npy'):
                        (model/name).write_bytes(b'[]' if name.endswith('.json') else b'fixture bytes')
            missing=run/'dmon_k4_seed18/assignments.npy';missing.unlink()
            with self.assertRaisesRegex(ValueError,'assignments.npy'):export_round2.export(run,output)
            self.assertFalse((output/run.name).exists())
            missing.write_bytes(b'fixture bytes')
            with patch('builtins.print'):out=export_round2.export(run,output)
            self.assertTrue((out/'dmon_k4_seed18/assignments.npy').is_file())

    def test_temporal_export_requires_every_registered_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,run=self.fixture(root);output=root/'export'
            self.change_stage(run,'temporal_grid',omegas=[0,.1],seeds=[17,18])
            for name in ('temporal_summary.json','temporal_stability.json'):(run/name).write_text('[]',encoding='utf-8')
            (run/'transition_events.csv').write_text('omega,period,entity_id\n',encoding='utf-8')
            for omega in (0,.1):
                for seed in (17,18):(run/f'temporal_omega{omega:g}_seed{seed}.json').write_text('{}',encoding='utf-8')
            missing=run/'temporal_omega0.1_seed18.json';missing.unlink()
            with self.assertRaisesRegex(ValueError,'temporal_omega0.1_seed18'):export_round2.export(run,output)
            self.assertFalse((output/run.name).exists())
            missing.write_text('{}',encoding='utf-8')
            with patch('builtins.print'):out=export_round2.export(run,output)
            self.assertEqual(len(list(out.glob('temporal_omega*.json.gz'))),4)

if __name__=='__main__':unittest.main()
