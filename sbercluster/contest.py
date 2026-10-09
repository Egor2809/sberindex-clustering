"""Exploratory contest extension: independent graph layers and descriptive dynamics.

The 2024 outcomes have already been inspected. No result here is a new holdout.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import json
from io import BytesIO
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from .features import make_slices, scalers_equal
from .graph import knn_graph
from .io import CATEGORIES,TOTAL,sha256,write_json
from .input_contracts import validate_prepared_panel
from .metrics import all_metrics
from .models import require_execution,fit_temporal
from .research import fit_candidate
from .dynamics import transitions
from .contest_graphs import read_road_graph,trajectory_graph,mix_layers,graph_summary
from .contest_analysis import drift_analysis,describe_profiles


def metric_row(x,z,reference_graph,spec,reference_labels):
    # Candidate metadata describes the requested fit; occupied K and all
    # measured fields belong to the actual partition on the common graph.
    return {**spec,**all_metrics(x,z,reference_graph),
        'requested_k':spec.get('k'),
        'ARI_reference':float(adjusted_rand_score(reference_labels,z))}


def run(cfg,root):
    require_execution(cfg,True)
    root=Path(root)
    stage=cfg['contest']['stage']
    if stage not in ('pilot','full'):
        raise ValueError('Expected pilot or full stage')
    out=root/cfg['contest']['output']
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite a previous run: {out}')
    out.mkdir(parents=True)
    write_json(out/'status.json',{'status':'running','stage':stage})
    print(json.dumps({'research_run':str(out),'stage':stage}),flush=True)
    started=time.perf_counter()
    panel_path=root/'data/processed/panel.csv'
    manifest=json.loads((panel_path.parent/'manifest.json').read_text('utf-8'))
    if sha256(panel_path)!=manifest['panel_sha256']:
        raise ValueError('Prepared panel fingerprint differs')
    panel=pd.read_csv(panel_path,dtype={'entity_id':str,'period':str,'oktmo':str})
    validate_prepared_panel(panel,manifest,cfg)
    slices,scaler=make_slices(panel,cfg)
    if len(slices)!=24 or [s[0] for s in slices] != pd.date_range('2023-01-01','2024-12-01',freq='MS').strftime('%Y-%m-%d').tolist():
        raise ValueError('Complete2023-2024 calendar required')
    ids=slices[0][1]
    if any(s[1]!=ids for s in slices):
        raise ValueError('Monthly cohort not aligned')
    monthly=np.stack([s[2] for s in slices])
    annual=np.median(monthly[:12],axis=0)
    vintage=root/'reports/experiments/2026-09-23-v2/validation'
    from .io import read_verified_artifact
    frozen=json.loads(read_verified_artifact(vintage.parent,'validation/frozen_prototypes.json'))
    if not scalers_equal(scaler,frozen.get('scaler')):
        raise ValueError('Frozen model and current feature space differ')
    if frozen['ids']!=ids:
        raise ValueError('Frozen model and panel identifiers differ')
    centers=np.asarray(frozen['centers'])
    assignment=pd.read_csv(BytesIO(read_verified_artifact(vintage.parent,'validation/reference_assignments.csv')),
                           dtype={'entity_id':str}).set_index('entity_id').reindex(ids)
    reference=assignment.cluster.to_numpy(dtype=int)
    if not np.array_equal(cdist(annual,centers).argmin(axis=1),reference):
        raise ValueError('Frozen2023 assignment no longer reproduces')
    source=Path(cfg['contest']['source_dir'])
    if not source.is_absolute():source=root/source
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'exploratory_retrospective_after_2024_inspection',
        'config':cfg,'panel_sha256':sha256(panel_path),'frozen_prototypes_sha256':sha256(vintage/'frozen_prototypes.json'),
        'source_hashes':{p.name:sha256(p) for p in [source/'hackathonlicence/connection.parquet',source/'t_dict_municipal_districts.xlsx']},
        'scaler':scaler,'implementation':{str(p.relative_to(root)).replace('\\','/'):sha256(p) for p in (root/'sbercluster').glob('*.py')}}
    evidence=cfg['data_contract'].get('boundary_review_evidence',[])
    provenance['boundary_evidence_hashes']={name:sha256(root/name) for name in evidence}
    import importlib.metadata
    import platform
    provenance['environment']={'python':platform.python_version(),'platform':platform.platform(),
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','polars','igraph','leidenalg']}}
    for relative in provenance['implementation']:
        target=out/'implementation_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/relative).read_bytes())
    write_json(out/'provenance.json',provenance)
    print(json.dumps({'phase':'features','n':len(ids),'months':len(slices)}),flush=True)
    attr,attrinfo=knn_graph(annual,cfg['graph']['k'])
    road,road_neighbors,roadinfo=read_road_graph(source/'hackathonlicence/connection.parquet',ids,cfg['graph']['k'])
    trajectory,trajectoryinfo=trajectory_graph(monthly[:12],cfg['graph']['k'])
    network={'layers':[{'name':'attribute',**graph_summary(attr)}, {'name':'transport',**roadinfo},
             {'name':'residual_trajectory',**trajectoryinfo}], 'candidates':[],'labels':{},
             'note':'All network indices in the main comparison use the same binary attribute reference graph. Road-snapshot and residual-trajectory reference graphs are reported separately. Mixing uses equal mean weighted degree; transport is dated2024-12-31.'}
    for name,a in [('attribute',attr),('transport',road),('trajectory',trajectory)]:save_npz(out/(name+'_graph.npz'),a)
    rows=[]
    comparison=[]
    def record(spec,z,graph):
        row=metric_row(annual,z,attr,spec,reference)
        rows.append(row)
        network['labels'][spec['id']]={key:int(label) for key,label in zip(ids,z)}
        # Same memberships evaluated on genuinely different reference networks.
        from .metrics import network_indices
        comparison.append({'id':spec['id'],'attribute':network_indices(attr,z),
                           'transport':network_indices(road,z),'trajectory':network_indices(trajectory,z)})
        np.save(out/(spec['id']+'.npy'),z,allow_pickle=False)
        write_json(out/'candidate_metrics.json',rows)
        print(json.dumps({'candidate':spec['id'],'SW':row['SW'],'k':row['k'],'min':row['min_cluster_size']}),flush=True)
    record({'id':'frozen_kmeans_k4','method':'kmeans_frozen','graph':'attribute','alpha':None},reference,attr)
    def fit_record(spec,g,fit_spec=None):
        try:
            z,info=fit_candidate(annual,g,fit_spec or spec,cfg['seed'],cfg)
        except ValueError as exc:
            if str(exc)!='Degenerate partition':raise
            row={**spec,'status':'unsupported_degenerate_partition','n':len(ids),
                **{key:None for key in ['SW','CH','CH_per_n','S_Dbw','S_Dbw_paper','AVI','AVU','TurboMQ','NewmanQ','ARI_reference']}}
            rows.append(row)
            write_json(out/'candidate_metrics.json',rows)
            print(json.dumps({'candidate':spec['id'],'status':row['status']}),flush=True)
            return
        spec['warnings']=info['warnings']
        record(spec,z,g)
    candidate_specs=[{'id':f'{m}_k{k}','method':m,'k':k,'graph':'attribute','alpha':None}
                     for m in ['kmeans','ward'] for k in ([2] if stage=='pilot' else [2,3])]
    for spec in candidate_specs:
        fit_record(spec,attr)
    alphas=[.5] if stage=='pilot' else cfg['contest']['alphas']
    for alpha in alphas:
        g=mix_layers(attr,road,alpha)
        spec={'id':f'spectral_mixed_a{alpha:g}','method':'spectral_mixed','k':4,
              'alpha':float(alpha),'graph':'attribute_transport'}
        fit_spec={**spec,'method':'spectral_global'}
        fit_record(spec,g,fit_spec)
    if stage!='pilot':
        for layer,g in [('transport',road),('trajectory',trajectory)]:
            spec={'id':'leiden_'+layer,'method':'leiden_global','resolution':.7,'graph':layer,'alpha':None}
            fit_record(spec,g)
    network['candidates']=rows
    write_json(out/'network_reference_comparison.json',comparison)
    dynamics,raw,relative=drift_analysis(monthly,centers,reference)
    if not np.array_equal(raw,assignment.cluster_2024.to_numpy(dtype=int)):
        raise ValueError('Frozen2024 assignment no longer reproduces')
    ratios=panel[CATEGORIES].div(panel[TOTAL],axis=0)*100
    ratios['year']=panel.period.str[:4]
    for year in ['2023','2024']:
        annual_ratios=panel[panel.period.str.startswith(year)].copy()
        ratio=annual_ratios[CATEGORIES].div(annual_ratios[TOTAL],axis=0)*100
        ratio['entity_id']=annual_ratios.entity_id.values
        dynamics['national_ratios_'+year]=ratio.groupby('entity_id')[CATEGORIES].median().median().to_list()
    registry=pd.read_excel(source/'t_dict_municipal_districts.xlsx',dtype={'oktmo':str})
    market=pd.read_csv(root/'data/processed/market_access_2024.csv')
    profiles=describe_profiles(panel,ids,reference,centers,annual,market,registry)
    changes=[]
    for c in range(4):
        selected=np.where(reference==c)[0]
        p=panel[panel.entity_id.isin([ids[i] for i in selected])].copy()
        for cat in CATEGORIES:
            p['ratio']=p[cat]/p[TOTAL]*100
            med=p.groupby([p.period.str[:4],'entity_id']).ratio.median().unstack(0)
            changes.append({'cluster':c,'category':cat,'n':len(selected),'median_2023':float(med['2023'].median()),
                'median_2024':float(med['2024'].median()),'median_paired_change_pp':float((med['2024']-med['2023']).median())})
    pd.DataFrame(changes).to_csv(out/'profile_paired_changes.csv',index=False)
    temporal={'variants':[],'note':'Joint retrospective optimization; increased persistence is induced by coupling, not proof of economic truth. Constant source IDs; within2024 boundaries not independently cleared.'}
    if stage=='full':
        # Twelve2023 layers plus twelve2024 layers; graph affinities use frozen2023 scaling.
        graphs=[mix_layers(knn_graph(x,cfg['graph']['k'])[0],road,.5) for _,_,x in slices]
        for omega in cfg['contest']['omegas']:
            temporal_cfg=deepcopy(cfg)
            temporal_cfg['clustering']['temporal_coupling_relative']=omega
            memberships,info=fit_temporal(slices,graphs,temporal_cfg,execute=True)
            stability=[transitions(ids,memberships[t-1],ids,memberships[t]) for t in range(1,len(slices))]
            from sklearn.metrics import silhouette_score
            measured=[0,11,12,23]
            sw=[float(silhouette_score(slices[t][2],memberships[t])) for t in measured
                if 2<=len(np.unique(memberships[t]))<len(ids)]
            row={'omega_relative':omega,'mean_ARI_adjacent':float(np.mean([s['ARI'] for s in stability])),
                'mean_churn':float(np.mean([s['matched_churn'] for s in stability])),
                'mean_SW':float(np.mean(sw)) if sw else None,'SW_months':[slices[t][0] for t in measured],
                'groups_min':min(len(np.unique(z)) for z in memberships),'groups_max':max(len(np.unique(z)) for z in memberships),**info}
            temporal['variants'].append(row)
            write_json(out/(f'temporal_omega{omega:g}.json'),{'ids':ids,'periods':[s[0] for s in slices],
                       'labels':[z.tolist() for z in memberships],'transitions':stability,'summary':row})
            print(json.dumps({'phase':'temporal',**row}),flush=True)
    payload={'profiles':profiles,'dynamics':dynamics,'assignments':{'raw':dict(zip(ids,map(int,raw))),
             'relative':dict(zip(ids,map(int,relative)))},'network':network,'temporal':temporal,
             'transport_neighbors':road_neighbors,
             'metrics_note':'MQ is Newman modularity (Modularity Quality); TurboMQ is reported separately. S_Dbw variants have distinct definitions.',
             'limitations':['Потребительские профили не устанавливают отраслевую специализацию.',
                'Новые эксперименты разведочные: результаты 2024 уже были изучены.',
                'Дорожные расстояния относятся к 31.12.2024; связи не являются поездками или потоками денег.',
                'Поправка на общий сдвиг описательная; изменения границ внутри2024 отдельно не подтверждены.']}
    write_json(out/'atlas_extension.json',payload)
    pd.DataFrame(rows).drop(columns=['S_Dbw_diagnostics','warnings'],errors='ignore').to_csv(out/'candidate_metrics.csv',index=False)
    write_json(out/'summary.json',{'stage':stage,'seconds':time.perf_counter()-started,'n':len(ids),
        'candidate_count':len(rows),'raw_changed':dynamics['raw']['changed'],'relative_changed':dynamics['relative']['changed'],
        'temporal_variants':len(temporal['variants']),'scope':provenance['scope']})
    write_json(out/'status.json',{'status':'completed','stage':stage})
    print(json.dumps({'status':'completed','output':str(out),'seconds':time.perf_counter()-started}),flush=True)
