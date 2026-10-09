"""Preparation failures must preserve the previously accepted input snapshot."""
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl

from sbercluster.canonical import prepare
from sbercluster.data import (PREPARED_FILES, audit_and_prepare, publish_preparation,
                              preparation_stage, PreparationRollbackError)
from sbercluster.io import CATEGORIES, TOTAL, sha256
from sbercluster import canonical


class PreparationTransaction(unittest.TestCase):
    @staticmethod
    def inputs(root, csv=False):
        source=root/'artifacts/sources/acquisition'
        parquet=source/'hackathonlicence';parquet.mkdir(parents=True,exist_ok=True)
        rows=[{'territory_id':tid,'date':month,'category':category,
               'value':1000 if category==TOTAL else 100+tid}
              for tid in (1,2) for month in ('2023-01','2023-02')
              for category in CATEGORIES+[TOTAL]]
        pl.DataFrame(rows).write_parquet(parquet/'consumption.parquet')
        pl.DataFrame({'territory_id':[1,2],'market_access':[300.,200.]}).write_parquet(parquet/'market_access.parquet')
        pd.DataFrame({'territory_id':[1,2],'year_from':[2020,2020],'year_to':[2025,2025],
                      'municipal_district_name':['District A','District B'],
                      'region_name':['Region','Region'],'region_code':[1,1],'oktmo':['001','002'],
                      'municipal_district_center_lat':[55.,56.],
                      'municipal_district_center_lon':[37.,38.]}).to_excel(source/'t_dict_municipal_districts.xlsx',index=False)
        if csv:
            frame=pd.DataFrame(rows).rename(columns={'category':'category_15','date':'period'})
            frame['period']=frame.period+'-01';frame['mo']=frame.territory_id.map({1:'District A',2:'District B'})
            frame=frame.drop(columns='territory_id').assign(obs_status='A',freq='Месяц',unit_measure='руб.',unit_mult=0)
            path=root/'data/raw/reference.csv';path.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(path,sep=';',index=False,encoding='utf-8')
        return parquet

    @staticmethod
    def snapshot(root):
        names=PREPARED_FILES|{'reports/unrelated.json','data/processed/graphs/unchanged.npz'}
        return {name:(root/name).read_bytes() if (root/name).exists() else None for name in names}

    def accepted(self,root):
        parquet=self.inputs(root)
        prepare(root)
        for name in ('reports/unrelated.json','data/processed/graphs/unchanged.npz'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'unrelated accepted bytes')
        return parquet,self.snapshot(root)

    def test_script_csv_audit_cannot_replace_canonical_when_parquet_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();parquet,before=self.accepted(root)
            self.inputs(root,csv=True);(parquet/'consumption.parquet').unlink()
            script=root/'scripts/prepare.py';script.parent.mkdir()
            shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/prepare.py',script)
            with patch.object(sys,'path',sys.path.copy()):
                with self.assertRaises(FileNotFoundError):runpy.run_path(str(script),run_name='__main__')
            self.assertEqual(self.snapshot(root),before)
            self.assertFalse(list(root.glob('.prepare-*')))

    def test_invalid_market_preserves_every_previous_output(self):
        for rows in ({'territory_id':[1,1],'market_access':[300.,200.]},
                     {'territory_id':[1,2],'market_access':[np.nan,200.]},
                     {'territory_id':[1.5,2.],'market_access':[300.,200.]},
                     {'territory_id':[1,2],'unexpected':[300.,200.]}):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();parquet,before=self.accepted(root)
                self.inputs(root,csv=True)
                pl.DataFrame(rows).write_parquet(parquet/'market_access.parquet')
                with self.assertRaises(ValueError):prepare(root)
                self.assertEqual(self.snapshot(root),before)
                self.assertFalse(list(root.glob('.prepare-*')))

    def test_invalid_registry_and_consumption_preserve_previous_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();parquet,before=self.accepted(root)
            frame=pl.read_parquet(parquet/'consumption.parquet').with_columns(pl.lit(1.5).alias('territory_id'))
            frame.write_parquet(parquet/'consumption.parquet')
            with self.assertRaises(ValueError):prepare(root)
            self.assertEqual(self.snapshot(root),before)
            self.inputs(root)
            registry=root/'artifacts/sources/acquisition/t_dict_municipal_districts.xlsx'
            frame=pd.read_excel(registry);frame.loc[0,'year_to']=np.nan;frame.to_excel(registry,index=False)
            with self.assertRaises(ValueError):prepare(root)
            self.assertEqual(self.snapshot(root),before)

    def test_promotion_failure_restores_old_bytes_and_removes_new_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,before=self.accepted(root)
            self.inputs(root,csv=True)
            real_replace=os.replace;calls=[]
            def interrupted(origin,target):
                calls.append(Path(target).relative_to(root).as_posix())
                if len(calls)==5:raise OSError('injected publication failure')
                real_replace(origin,target)
            with patch('sbercluster.data.os.replace',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'injected'):prepare(root)
            self.assertEqual(self.snapshot(root),before)
            self.assertFalse(list(root.glob('.prepare-*')))

    def test_success_promotes_manifest_last_and_records_market_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();parquet,before=self.accepted(root)
            self.inputs(root,csv=True)
            real_replace=os.replace;order=[]
            def record(origin,target):
                order.append(Path(target).relative_to(root).as_posix());real_replace(origin,target)
            with patch('sbercluster.data.os.replace',side_effect=record):result=prepare(root)
            self.assertEqual(order[-1],'data/processed/manifest.json')
            self.assertEqual((root/'data/processed/panel.csv').read_bytes(),before['data/processed/panel.csv'])
            manifest=json.loads((root/'data/processed/manifest.json').read_text('utf-8'))
            self.assertEqual(manifest['identity_mode'],'source_territory_id')
            self.assertEqual(manifest['market_access_sha256'],sha256(parquet/'market_access.parquet'))
            self.assertEqual(result['market_access_sha256'],manifest['market_access_sha256'])
            self.assertEqual((root/'reports/unrelated.json').read_bytes(),before['reports/unrelated.json'])
            self.assertEqual(result['csv_export_diagnostics']['ambiguous_names'],0)

    def test_name_only_audit_failure_keeps_accepted_panel(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,before=self.accepted(root);self.inputs(root,csv=True)
            csv=root/'data/raw/reference.csv'
            frame=pd.read_csv(csv,sep=';');frame['unit_mult']=1;frame.to_csv(csv,sep=';',index=False)
            with self.assertRaises(ValueError):audit_and_prepare(csv,root)
            self.assertEqual(self.snapshot(root),before)

    def test_publisher_only_promotes_named_generated_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();stage=root/'stage';stage.mkdir()
            for name in ('data/processed/manifest.json','reports/unrelated.json'):
                path=stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}',encoding='utf-8')
            publish_preparation(stage,root)
            self.assertFalse((root/'reports/unrelated.json').exists())

    def test_failed_rollback_retains_backups_and_prevents_another_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,before=self.accepted(root)
            real_replace=os.replace;calls=[]
            def fail_twice(origin,target):
                calls.append(str(target))
                if len(calls) in (2,3):raise PermissionError('injected publication/rollback failure')
                real_replace(origin,target)
            with patch('sbercluster.data.os.replace',side_effect=fail_twice):
                with self.assertRaises(PreparationRollbackError) as caught:prepare(root)
            recovery=caught.exception.recovery_directory
            self.assertTrue(recovery.is_dir())
            self.assertEqual((recovery/'.backups/data/processed/canonical_long.csv').read_bytes(),
                             before['data/processed/canonical_long.csv'])
            self.assertTrue((root/'.prepare.lock').is_file())
            with self.assertRaises(FileExistsError):prepare(root)

    def test_active_lock_rejects_concurrent_preparation_without_changing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,before=self.accepted(root)
            with preparation_stage(root):
                with self.assertRaises(FileExistsError):prepare(root)
                self.assertEqual(self.snapshot(root),before)
            self.assertFalse((root/'.prepare.lock').exists())

    def test_changed_sources_are_rejected_before_publication(self):
        for name in ('consumption.parquet','market_access.parquet','t_dict_municipal_districts.xlsx','reference.csv'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder).resolve();_,before=self.accepted(root);self.inputs(root,csv=True)
                if name.endswith('.parquet'):
                    module,attribute=canonical,'read_parquet'
                elif name.endswith('.xlsx'):
                    module,attribute=canonical.pd,'read_excel'
                else:module,attribute=canonical,'read_export'
                original=getattr(module,attribute)
                def changed(path,*args,**kwargs):
                    result=original(path,*args,**kwargs)
                    if Path(path).name==name:
                        with Path(path).open('ab') as stream:stream.write(b'\n')
                    return result
                with patch.object(module,attribute,side_effect=changed):
                    with self.assertRaisesRegex(ValueError,'Source changed during preparation'):prepare(root)
                self.assertEqual(self.snapshot(root),before)
                self.assertFalse((root/'.prepare.lock').exists())

    def test_missing_optional_coordinates_and_excluded_metadata_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();parquet,_=self.accepted(root)
            source=root/'artifacts/sources/acquisition';registry=source/'t_dict_municipal_districts.xlsx'
            frame=pd.read_excel(registry);frame.loc[0,['municipal_district_center_lat','municipal_district_center_lon']]=np.nan
            extra={column:None for column in frame.columns};extra.update(territory_id=3,year_from=2020,year_to=2025)
            pd.concat([frame,pd.DataFrame([extra])],ignore_index=True).to_excel(registry,index=False)
            observations=pl.read_parquet(parquet/'consumption.parquet')
            extra_rows=observations.filter((pl.col('territory_id')==1)&(pl.col('date')=='2023-01')).with_columns(pl.lit(3,dtype=pl.Int64).alias('territory_id'))
            pl.concat([observations,extra_rows]).write_parquet(parquet/'consumption.parquet')
            result=prepare(root)
            self.assertEqual(result['complete_panel_territories'],2)
            self.assertEqual(result['incomplete_territories'],1)
            panel=pd.read_csv(root/'data/processed/panel.csv')
            self.assertTrue(panel.loc[panel.territory_id==1,'municipal_district_center_lat'].isna().all())

    def test_prior_audit_is_kept_only_for_matching_input_vintage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();_,_=self.accepted(root);path=root/'reports/data_audit.json'
            prior=json.loads(path.read_text('utf-8'));prior['csv_export_diagnostics']['n_unique_names']=123
            path.write_text(json.dumps(prior),encoding='utf-8')
            self.assertEqual(prepare(root)['csv_export_diagnostics']['n_unique_names'],123)
            prior=json.loads(path.read_text('utf-8'));prior.update(sha256='different vintage',stale_statistic=999)
            path.write_text(json.dumps(prior),encoding='utf-8')
            result=prepare(root)
            self.assertNotIn('stale_statistic',result)
            self.assertNotIn('n_unique_names',result['csv_export_diagnostics'])


if __name__=='__main__':unittest.main()
