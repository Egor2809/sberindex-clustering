"""Re-evaluate existing annual partitions with explicit ICVI conventions. No fitting."""
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
import pandas as pd
from sklearn import set_config
set_config(working_memory=8)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sbercluster.features import make_slices
from sbercluster.graph import knn_graph
from sbercluster.metrics import all_metrics
from sbercluster.io import sha256,write_json


def main():
    out=ROOT/'reports/contest-v3'
    source=ROOT/'reports/experiments/2026-09-23-v2'
    cfg=json.loads((ROOT/'configs/research_annual.json').read_text('utf-8'))
    path=ROOT/'data/processed/panel.csv'
    manifest=json.loads((path.parent/'manifest.json').read_text('utf-8'))
    if sha256(path)!=manifest['panel_sha256']:raise ValueError('Panel snapshot differs')
    panel=pd.read_csv(path,dtype={'entity_id':str,'period':str,'oktmo':str})
    slices,_=make_slices(panel[panel.period.str.startswith('2023')],cfg)
    ids=slices[0][1]
    annual=np.median(np.stack([s[2] for s in slices]),axis=0)
    graph,_=knn_graph(annual,15)
    assignments=pd.read_csv(source/'screen_partitions.csv.gz')
    assignments=assignments[assignments.representation.eq('annual_2023')]
    rows=[]
    for candidate,part in assignments.groupby('candidate',sort=True):
        if part.entity_id.duplicated().any() or set(part.entity_id)!=set(ids):
            raise ValueError('Published labels do not join exactly: '+candidate)
        labels=part.set_index('entity_id').reindex(ids).cluster.to_numpy(dtype=int)
        row={'candidate':candidate,**all_metrics(annual,labels,graph)}
        rows.append(row)
    write_json(out/'published_partition_metrics.json',rows)
    pd.DataFrame(rows).drop(columns=['S_Dbw_diagnostics'],errors='ignore').to_csv(out/'published_partition_metrics.csv',index=False)
    write_json(out/'evaluation_provenance.json',{'scope':'re-evaluation of existing annual2023 memberships; no fitting',
        'panel_sha256':sha256(path),'assignments_sha256':sha256(source/'screen_partitions.csv.gz'),
        'metric_implementation_sha256':sha256(ROOT/'sbercluster/metrics.py'),
        'script_sha256':sha256(Path(__file__)),'n':len(ids),'candidates':len(rows),
        'reference_graph':'annual2023 attribute union15NN; binary for all network indices'})
    print(json.dumps({'status':'completed_no_fitting','candidates':len(rows),'reference':next(r for r in rows if r['candidate']=='kmeans_k4')},ensure_ascii=False))


if __name__=='__main__':main()
