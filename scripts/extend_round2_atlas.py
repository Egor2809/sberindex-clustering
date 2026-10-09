"""Attach completed round-two evidence to the existing atlas contract."""
from pathlib import Path
import json,gzip,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment

def extend(base,root,ids,reference_labels):
    directory=root/'reports/round2-2026-09-24'
    if not directory.exists():return
    latest_dmon = root/'reports/dmon-plateau-2026-09-25/dmon-plateau-20260925'
    def stage_dir(stage):
        return latest_dmon if stage=='dmon' and latest_dmon.exists() else directory/('round2-20260924-'+stage)
    def read(stage,name):
        path=stage_dir(stage)/name
        raw=path.read_bytes();return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
    for stage in ['region_control','dmon','temporal_grid']:
        if read(stage,'status.json')['status']!='completed':raise ValueError('Round-two atlas requires completed comparisons')
    metrics=read('dmon','metrics.json');memberships=read('dmon','labels.json.gz')
    if memberships['ids']!=ids:raise ValueError('DMoN IDs differ from atlas')
    for k in [2,4]:
        name=f'dmon_k{k}_seed1729'
        fit=next(r['fit'] for r in metrics if r['id']==name)
        stopping='плато loss и состава групп' if fit.get('plateau_reached') else 'заданный бюджет'
        raw=np.asarray(memberships['labels'][name]);unique,z=np.unique(raw,return_inverse=True)
        occupied=len(unique)
        if occupied<2:continue
        if occupied==4:
            cross=np.zeros((4,4),int);np.add.at(cross,(reference_labels,z),1)
            rows,cols=linear_sum_assignment(-cross);mapping=dict(zip(cols.tolist(),rows.tolist()))
            display=np.array([mapping[int(v)] for v in z])
        else:display=z
        base['map_models'][name]={'name':f'DMoN · K={k} · seed 1729','labels':dict(zip(ids,display.tolist())),
            'profiles':[{'id':int(c),'name':f'DMoN: группа {int(c)+1}','n':int((display==c).sum())} for c in np.unique(display)],
            'note':f"DMoN по формуле статьи: расходы 2023 + дорожный граф 31.12.2024; {fit['epochs']:,} эпох, {stopping}. Seed 1729 выбран заранее. Цвета K=4 сопоставлены с опорными группами по пересечению состава, а не по экономическому значению. Ретроспективный результат."}
    static=read('region_control','metrics.json')
    selected=[r for r in static if r['id'].startswith('joint_same_region')]+[r for r in metrics if r['id'].startswith('dmon')]
    base['metric_comparison'].extend({key:value for key,value in {**r,'candidate':r['id']}.items() if key not in ['fit','on_real_road','road_network']} for r in selected)
    rows=read('temporal_grid','temporal_summary.json');stability=read('temporal_grid','temporal_stability.json');variants=[]
    for omega in sorted(set(r['omega'] for r in rows)):
        group=[r for r in rows if r['omega']==omega];v={'omega_relative':omega,'groups_min':min(r['groups_min'] for r in group),'groups_max':max(r['groups_max'] for r in group),'seeds':len(group)}
        for field in ['mean_ARI_adjacent','mean_churn','mean_SW']:
            values=[r[field] for r in group if r[field] is not None]
            v[field]=sum(values)/len(values) if values else None
        v.update(next(r for r in stability if r['omega']==omega));variants.append(v)
    base['temporal']={'variants':variants,'note':'24 месяца, пять весов связи, по три seed. Показаны средние по seed, согласие между ними и однозначные смены по всем трём. Случаи с равнозначным сопоставлением меток исключены из устойчивых событий. Четыре итерации оптимизации; сходимость не заявляется. Сглаживание само по себе не подтверждает экономический переход.'}
    base['round2']={'regional_effect':read('region_control','regional_effect.json'),
        'external_comparison':read('region_control','external_comparison.json'),
        'dmon_summary':[{key:value for key,value in r.items() if key in ['id','k','SW','min_cluster_size']} for r in metrics],
        'report':'REPORT_RU.md'}
    for stage in ['region_control','dmon','temporal_grid']:
        path=stage_dir(stage)/'manifest.json'
        base.setdefault('source_hashes',{})[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
