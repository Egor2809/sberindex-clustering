"""Describe existing frozen memberships, without fitting any clustering model."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sbercluster.features import make_slices,scalers_equal
from sbercluster.input_contracts import validate_prepared_panel
from sbercluster.contest_analysis import drift_analysis,describe_profiles
from sbercluster.io import CATEGORIES,TOTAL,write_json,sha256,read_verified_artifact


def main():
    cfg=json.loads((ROOT/'configs/contest_pilot.json').read_text('utf-8'))
    path=ROOT/'data/processed/panel.csv'
    manifest=json.loads((path.parent/'manifest.json').read_text('utf-8'))
    if sha256(path)!=manifest['panel_sha256']:raise ValueError('Panel snapshot differs')
    panel=pd.read_csv(path,dtype={'entity_id':str,'period':str,'oktmo':str})
    validate_prepared_panel(panel,manifest,cfg)
    slices,scaler=make_slices(panel,cfg)
    ids=slices[0][1]
    if any(s[1]!=ids for s in slices):raise ValueError('Monthly IDs differ')
    monthly=np.stack([s[2] for s in slices])
    frozen=json.loads(read_verified_artifact(ROOT/'reports/experiments/2026-09-23-v2', 'validation/frozen_prototypes.json'))
    if not scalers_equal(scaler,frozen.get('scaler')):
        raise ValueError('Frozen model and current feature space differ')
    if frozen['ids']!=ids:raise ValueError('Frozen IDs differ')
    centers=np.array(frozen['centers'])
    labels=np.array(frozen['reference_labels'])
    dynamics,raw,relative=drift_analysis(monthly,centers,labels)
    for year in ['2023','2024']:
        p=panel[panel.period.str.startswith(year)]
        ratio=p[CATEGORIES].div(p[TOTAL],axis=0)*100
        ratio['entity_id']=p.entity_id.values
        dynamics['national_ratios_'+year]=ratio.groupby('entity_id')[CATEGORIES].median().median().tolist()
    registry=pd.read_excel(ROOT/'artifacts/sources/acquisition/t_dict_municipal_districts.xlsx')
    profiles=describe_profiles(panel,ids,labels,centers,np.median(monthly[:12],axis=0),
        pd.read_csv(ROOT/'data/processed/market_access_2024.csv'),registry)
    payload={'profiles':profiles,'dynamics':dynamics,
        'assignments':{'raw':dict(zip(ids,map(int,raw))),'relative':dict(zip(ids,map(int,relative)))},
        'network':{'candidates':[],'labels':{},'layers':[]},'temporal':{'variants':[]},
        'limitations':['Типы описывают потребление; отраслевая специализация не установлена.',
            'Поправка на общий сдвиг — ретроспективная проверка чувствительности.',
            'Границы внутри 2024 года отдельно не подтверждены.'],
        'scope':'descriptive; no new fitting; frozen2023 prototypes'}
    metrics=ROOT/'reports/contest-v3/published_partition_metrics.json'
    if metrics.is_file():
        payload['metric_comparison']=json.loads(metrics.read_text('utf-8'))
    write_json(ROOT/'reports/contest-v3/descriptive.json',payload)
    write_json(ROOT/'reports/contest-v3/descriptive_provenance.json',{
        'scope':'descriptive reassignment to existing 2023 centers; no fitting',
        'panel_sha256':sha256(path),
        'frozen_prototypes_sha256':sha256(ROOT/'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json'),
        'script_sha256':sha256(Path(__file__)),
        'analysis_sha256':sha256(ROOT/'sbercluster/contest_analysis.py'),
        'calendar_month_correction':'subtract cross-sectional median difference, then annual median',
        'map_changes_do_not_enter_features':True})
    print(json.dumps({'raw':dynamics['raw'],'relative':dynamics['relative'],'annual':dynamics['annual_median_adjustment'],'profiles':profiles},ensure_ascii=False))


if __name__=='__main__':main()
