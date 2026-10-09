"""Exploratory round-two comparisons on identical frozen attributes and graphs."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib.metadata, json, re, time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score
from .followup import load_inputs, load_stability_inputs
from .graph import knn_graph
from .published_inputs import load_published, read_archive
from .joint import fit_graph_regularized_kmeans, joint_objective
from .models import require_execution, fit_temporal
from .metrics import all_metrics, network_indices
from .contest_graphs import mix_layers
from .dynamics import transitions
from .io import sha256, write_json, code_revision


def same_region_graph(regions):
    """Complete within-region graph; each non-singleton vertex has unit strength."""
    regions=np.asarray(regions)
    if pd.isna(regions).any(): raise ValueError('Missing region')
    rr=[];cc=[];vv=[]
    for group in np.unique(regions):
        idx=np.flatnonzero(regions==group);n=len(idx)
        if n<2: continue
        a=np.repeat(idx,n);b=np.tile(idx,n);keep=a!=b
        rr.extend(a[keep]);cc.extend(b[keep]);vv.extend(np.full(n*(n-1),1/(n-1)))
    return csr_matrix((vv,(rr,cc)),shape=(len(regions),len(regions)))


def canonical_labels(labels):
    """Number clusters by their first vertex, independently of label values."""
    labels=np.asarray(labels)
    if labels.ndim!=1 or not len(labels): raise ValueError('Nonempty label vector required')
    _,first,inverse=np.unique(labels,return_index=True,return_inverse=True)
    mapping=np.empty(len(first),int);mapping[np.argsort(first)]=np.arange(len(first))
    return mapping[inverse]


def matched_change(a,b,*,return_ambiguity=False):
    """Canonical optimal matching, plus vertices whose status varies at a tie.

    Scalar churn is invariant across optimal matchings. A vertex is ambiguous
    exactly when its own contingency cell can be both matched and unmatched in
    optimal assignments; those vertices cannot support unanimous event claims.
    """
    ia,ib=canonical_labels(a),canonical_labels(b)
    if ia.shape!=ib.shape: raise ValueError('Aligned label vectors required')
    table=np.zeros((ia.max()+1,ib.max()+1),int);np.add.at(table,(ia,ib),1)
    r,c=linear_sum_assignment(-table);mapping=dict(zip(c,r))
    changes=np.array([mapping.get(int(v),-1) for v in ib])!=ia
    if not return_ambiguity: return changes
    optimum=int(table[r,c].sum());chosen=set(zip(r,c));uncertain=np.zeros_like(table,dtype=bool)
    for i,j in zip(*np.nonzero(table)):
        if (i,j) in chosen:
            cost=-table.copy();cost[i,j]=len(ia)+1
            rr,cc=linear_sum_assignment(cost)
            alternative=-int(cost[rr,cc].sum())
        else:
            remaining=np.delete(np.delete(table,i,axis=0),j,axis=1)
            rr,cc=linear_sum_assignment(-remaining)
            alternative=int(table[i,j]+remaining[rr,cc].sum())
        uncertain[i,j]=alternative==optimum
    return changes,uncertain[ia,ib]


def transition_votes(changes,ambiguities):
    """Count only unambiguous change votes; retain ambiguous vote counts."""
    changes=np.asarray(changes,dtype=bool);ambiguities=np.asarray(ambiguities,dtype=bool)
    if changes.shape!=ambiguities.shape or changes.ndim!=3 or not len(changes):
        raise ValueError('Expected aligned seed x transition x vertex masks')
    votes=(changes & ~ambiguities).sum(axis=0)
    return votes,ambiguities.sum(axis=0),votes==len(changes)


def requested_k(name,occupied,explicit=None):
    if explicit and name in explicit: return int(explicit[name])
    match=re.search(r'(?:^|_)k(\d+)(?:_|$)',name)
    return int(match.group(1)) if match else int(occupied)


def finite_interval(draws):
    valid=np.isfinite(draws)
    return (np.quantile(np.asarray(draws)[valid],[.025,.975]).tolist() if valid.any() else None),int(valid.sum())


def external_comparison(root, ids, partitions, baseline_names, bootstrap=500, *, requested_ks=None):
    """Paired region bootstrap; predefined requested K selects the baseline."""
    from scripts.external_validation_v4 import residual_matrix, partial_r2
    if not isinstance(bootstrap,(int,np.integer)) or bootstrap<1: raise ValueError('Positive bootstrap count required')
    bridge=pd.read_csv(root/'reports/external-v4/municipality_bridge.csv').set_index('entity_id').loc[ids].reset_index()
    if bridge.entity_id.duplicated().any(): raise ValueError('Duplicate external key')
    results=[]
    for subset in ['all','without_intracity']:
        keep=np.isfinite(bridge.market_access.to_numpy(float)) & np.isfinite(bridge.log_expense_2023.to_numpy(float))
        keep &= bridge.region_code.notna().to_numpy() & bridge.municipal_district_type.notna().to_numpy()
        if subset!='all': keep &= ~bridge.intracity_moscow_petersburg.astype(bool).to_numpy()
        f=bridge.loc[keep].copy();rcodes,rnames=pd.factorize(f.region_code,sort=True)
        rng=np.random.default_rng(1729)
        counts=[np.bincount(rng.integers(len(rnames),size=len(rnames)),minlength=len(rnames)) for _ in range(bootstrap)] if len(rnames) else []
        stats={}
        for name,z in partitions.items():
            z=canonical_labels(z)
            if len(z)!=len(ids): raise ValueError('External labels must match ordered IDs')
            if len(np.unique(z))>4: raise ValueError('This registered external comparison covers K<=4')
            f['cluster']=z[keep]
            if not len(f):
                stats[name]=(None,np.full(bootstrap,np.nan));continue
            matrix,_=residual_matrix(f,'market_access','region_type_level')
            cross=np.stack([matrix[rcodes==r].T@matrix[rcodes==r] for r in range(len(rnames))])
            observed=partial_r2(cross.sum(axis=0),True)
            draws=np.array([partial_r2(np.einsum('r,rij->ij',w,cross),True) for w in counts],float)
            stats[name]=(float(observed) if observed is not None and np.isfinite(observed) else None,draws)
        for name,(observed,draws) in stats.items():
            occupied=len(np.unique(partitions[name]));k=requested_k(name,occupied,requested_ks);base=baseline_names.get(k)
            interval,valid=finite_interval(draws)
            row={'id':name,'subset':subset,'k':occupied,'requested_k':k,'occupied_k':occupied,
                 'partition_status':'fewer_occupied_than_requested' if occupied<k else 'requested_k_occupied',
                 'n':len(f),'regions':len(rnames),'partial_R2':observed,'ci95':interval,
                 'controls':'region x type + log expense 2023','bootstrap':bootstrap,
                 'valid_bootstrap_draws':valid,'baseline':base,
                 'status':'ok' if observed is not None else 'undefined_residual_outcome_variance_or_empty_cohort'}
            if base in stats:
                base_observed,base_draws=stats[base]
                paired=np.isfinite(draws)&np.isfinite(base_draws)
                delta=draws[paired]-base_draws[paired]
                delta_interval,paired_n=finite_interval(delta)
                row.update(delta_partial_R2=observed-base_observed if observed is not None and base_observed is not None else None,
                           delta_ci95=delta_interval,valid_paired_bootstrap_draws=paired_n)
            results.append(row)
    return {'results':results,'bridge_sha256':sha256(root/'reports/external-v4/municipality_bridge.csv'),
            'scope':'Exploratory association on already-inspected market access; paired regional bootstrap conditional on fixed labels; not a significance test or causal effect. Requested K fixes the baseline even if a fit occupies fewer clusters. Undefined draws are excluded on a common paired mask for differences. Market accessibility and roads share infrastructural content, so independence of measurement is limited.'}

def safe_metrics(x,z,adjacency):
    k=len(np.unique(z))
    if k < 2:
        return {'n':len(x),'k':k,'min_cluster_size':len(x),'status':'collapsed_single_cluster',
                **{key:None for key in ['SW','CH','S_Dbw','AVI','AVU','MQ','TurboMQ','NewmanQ']},
                'undefined_reason':'At least two occupied clusters required for partition comparison'}
    return all_metrics(x,z,adjacency)


def validate_temporal_reuse(old,prior,status,cfg,ids,periods,manifest,scaler,omega,seed):
    """Validate scientific equivalence before reusing either raw or public data."""
    pc=prior['config'];settings=cfg['followup'];weight=settings.get('attribute_graph_weight',.5)
    if status.get('status')!='completed' or status.get('stage')!='temporal' or status.get('n')!=len(ids):
        raise ValueError('Completed temporal source with matching node count required')
    if old.get('ids')!=ids or prior.get('ids')!=ids or old.get('periods')!=periods:
        raise ValueError('Temporal source identifiers or periods differ')
    from .features import scalers_equal
    if prior.get('panel_sha256')!=manifest['panel_sha256'] or not scalers_equal(prior.get('scaler'),scaler):
        raise ValueError('Temporal source panel or scaler differs')
    if any(pc.get(key)!=cfg.get(key) for key in ['schema_version','data_contract','features','graph']) or pc.get('seed')!=seed:
        raise ValueError('Temporal source preprocessing or seed differs')
    if any(pc['clustering'].get(key)!=cfg['clustering'].get(key) for key in ['leiden_iterations','resolution']):
        raise ValueError('Temporal source optimization settings differ')
    if pc['validation']['development_end']!=cfg['validation']['development_end']:
        raise ValueError('Temporal coupling calibration period differs')
    if (prior.get('road_sha256')!=settings['expected_road_sha256'] or prior.get('road_date')!='2024-12-31'
            or prior.get('attribute_graph_weight')!=weight
            or pc['followup'].get('graph_layer')!='attribute_road'
            or pc['followup'].get('attribute_graph_weight')!=weight
            or pc['followup'].get('month_indices')!=settings['month_indices']):
        raise ValueError('Temporal road source, mixture or month indices differ')
    if (old['summary'].get('omega_relative')!=omega or old['summary'].get('months')!=len(periods)
            or len(old['labels'])!=len(periods) or any(len(z)!=len(ids) for z in old['labels'])
            or [row['period'] for row in old['monthly_metrics']]!=periods):
        raise ValueError('Temporal source variant, labels or monthly metrics differ')
    labels=np.asarray(old['labels'])
    if labels.shape!=(len(periods),len(ids)) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('Temporal labels must contain one integer per entity and month')


def load_temporal_reuse(root,cfg,ids,periods,manifest,scaler,omega,seed):
    name='review-20260924-temporal';public=cfg['followup'].get('published_inputs',False)
    directory=root/'reports/review-2026-09-24'/name if public else root/'runs'/name
    read=(lambda file:read_archive(root,name,file)) if public else (lambda file:json.loads((directory/file).read_text('utf-8')))
    filename=f'temporal_omega{omega:g}.json'+('.gz' if public else '')
    old,prior,status=read(filename),read('provenance.json'),read('status.json')
    validate_temporal_reuse(old,prior,status,cfg,ids,periods,manifest,scaler,omega,seed)
    info={**old['summary'],'reused':True,'source_sha256':sha256(directory/filename),
          'source_provenance_sha256':sha256(directory/'provenance.json'),
          'source_status_sha256':sha256(directory/'status.json')}
    return [np.asarray(z,int) for z in old['labels']],info,old['monthly_metrics']

def run(cfg, root):
    from .research import tracked_run
    require_execution(cfg,True);root=Path(root);s=cfg['followup'];stage=s['stage']
    if stage not in ('region_control','dmon','temporal_grid'):
        raise ValueError('Unknown round2 stage')
    out=root/s['output'];out.mkdir(parents=True,exist_ok=False)
    with tracked_run(out,stage=stage):
        _run_created(cfg,root,out)
    return out


def _run_created(cfg,root,out):
    s=cfg['followup'];stage=s['stage']
    start=time.perf_counter();print(json.dumps({'research_run':str(out),'stage':stage}),flush=True)
    slices,monthly,x,ids,regions,z0,scaler,manifest=load_inputs(root,cfg)
    road,roadinfo,initials,originals,reused=(load_published if s.get('published_inputs') else load_stability_inputs)(root,cfg,ids,manifest)
    attr,attrinfo=knn_graph(x,cfg['graph']['k']);graphs={'attribute':attr,'road':road}
    modules=['sbercluster/'+n+'.py' for n in ['round2','followup','joint','features','input_contracts','graph','metrics','models','contest_graphs','dynamics','io','published_inputs','research']]+['scripts/external_validation_v4.py','scripts/run_contest.py']
    if stage=='dmon':modules+=['sbercluster/dmon.py']
    hashes={}
    for rel in modules:
        p=root/rel;h=sha256(p);hashes[rel]=h;q=out/'source_objects'/f'{h}.py';q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
    provenance={'date_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,**code_revision(root),'panel_sha256':manifest['panel_sha256'],'scaler':scaler,'ids':ids,'sources':hashes,**reused,'scope':'Exploratory; 2024 inspected before all these comparisons','packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','pandas','igraph','leidenalg']}}
    write_json(out/'provenance.json',provenance);write_json(out/'graphs.json',{'attribute':attrinfo,'road':roadinfo})
    rows=[];labels={}
    def record(name,z,details):
        z=np.asarray(z,int);labels[name]=z.tolist();row={'id':name,**safe_metrics(x,z,attr),'ARI_frozen_k4':float(adjusted_rand_score(z0,z)),'fit':details,'road_network':network_indices(road,z)}
        if len(np.unique(z))>1:row['on_real_road']=joint_objective(x,road,z,.25)
        rows.append(row);write_json(out/'metrics.json',rows);write_json(out/'labels.json',{'ids':ids,'labels':labels});print(json.dumps({'candidate':name,'SW':row['SW'],'k':row['k']}),flush=True)
    if stage=='region_control':
        region_graph=same_region_graph(regions)
        for k in [2,4]:
            record(f'kmeans_k{k}',initials[k]['kmeans'],{'reused':True})
            record(f'joint_road_k{k}',originals[k],{'reused':True,'alpha':.25})
            z,info=fit_graph_regularized_kmeans(x,region_graph,initials[k]['kmeans'],alpha=.25,seed=cfg['seed'],max_sweeps=50)
            record(f'joint_same_region_k{k}',z,{'graph':'complete within-region unit vertex strength','alpha':.25,**info})
        old=read_archive(root,'review-20260924-joint','labels.json.gz') if s.get('published_inputs') else json.loads((root/s['joint_dir']/'labels.json').read_text('utf-8'))
        for name in sorted(old['labels']):
            if name.startswith('joint_region_shuffled_'):record(name,old['labels'][name],{'reused':True,'null_preserves':'regional mixing, weighted graph topology and within-region degree distribution; not municipality-specific degrees'})
        base=next(r for r in rows if r['id']=='kmeans_k4')['on_real_road']['graph_cut_fraction'];real=next(r for r in rows if r['id']=='joint_road_k4')['on_real_road']['graph_cut_fraction']
        ratios=[(base-r['on_real_road']['graph_cut_fraction'])/(base-real) for r in rows if r['id'].startswith('joint_region_shuffled_')]
        write_json(out/'regional_effect.json',{'fraction_of_road_cut_gain_recovered_by_permuted_graph':ratios,'mean_fraction':float(np.mean(ratios)),'caveat':'Not a causal variance decomposition. Permutation preserves more than region labels: graph topology and region mixing also remain.'})
        write_json(out/'external_comparison.json',external_comparison(root,ids,labels,{2:'kmeans_k2',4:'kmeans_k4'},s['bootstrap']))
    elif stage=='dmon':
        from .dmon import fit_dmon
        provenance['packages']['torch']=importlib.metadata.version('torch');write_json(out/'provenance.json',provenance)
        for k in s['ks']:
            record(f'kmeans_k{k}',initials[k]['kmeans'],{'reused':True})
            record(f'joint_road_k{k}',originals[k],{'reused':True})
            for seed in s['seeds']:
                name=f'dmon_k{k}_seed{seed}';z,info=fit_dmon(x,road,k,seed,s['epochs'],hidden_dim=s['hidden_dim'],learning_rate=s['learning_rate'],output_dir=out/name,plateau=s.get('plateau'))
                record(name,z,info)
        write_json(out/'external_comparison.json',external_comparison(root,ids,labels,{2:'kmeans_k2',4:'kmeans_k4'},s['bootstrap']))
    elif stage=='temporal_grid':
        indices=s['month_indices'];selected=[slices[i] for i in indices]
        gs=[mix_layers(knn_graph(t[2],cfg['graph']['k'])[0],road,s.get('attribute_graph_weight',.5)) for t in selected]
        variants=[];series={}
        for omega in s['omegas']:
            for seed in s['seeds']:
                name=f'omega{omega:g}_seed{seed}';prior=root/'runs/review-20260924-temporal'/f'temporal_omega{omega:g}.json'
                if seed==1729 and omega in (0,.1) and indices==list(range(24)):
                    memberships,info,ms=load_temporal_reuse(root,cfg,ids,[t[0] for t in selected],manifest,scaler,omega,seed)
                else:
                    c=deepcopy(cfg);c['seed']=seed;c['clustering']['temporal_coupling_relative']=omega
                    memberships,info=fit_temporal(selected,gs,c,execute=True)
                    ms=[{'period':t[0],**safe_metrics(t[2],z,g)} for t,z,g in zip(selected,memberships,gs)]
                matched=[matched_change(a,b,return_ambiguity=True) for a,b in zip(memberships[:-1],memberships[1:])]
                changes=np.stack([pair[0] for pair in matched]);ambiguities=np.stack([pair[1] for pair in matched])
                ari=[adjusted_rand_score(a,b) for a,b in zip(memberships[:-1],memberships[1:])]
                summary={'id':name,'omega':omega,'seed':seed,'months':len(selected),'mean_ARI_adjacent':float(np.mean(ari)),'mean_churn':float(changes.mean()),'mean_SW':float(np.mean([m['SW'] for m in ms if m['SW'] is not None])) if any(m['SW'] is not None for m in ms) else None,'valid_SW_months':sum(m['SW'] is not None for m in ms),'groups_min':min(len(np.unique(z)) for z in memberships),'groups_max':max(len(np.unique(z)) for z in memberships),'iterations':cfg['clustering']['leiden_iterations'],'reused':bool(info.get('reused',False))}
                summary['ambiguous_vertex_months']=int(ambiguities.sum())
                summary['transition_definition']='Change under canonical maximum-overlap matching; ambiguous vertices excluded from unanimous events; no agreement on destination implied'
                series[name]=(memberships,changes,ambiguities);variants.append(summary)
                write_json(out/f'temporal_{name}.json',{'ids':ids,'periods':[t[0] for t in selected],'labels':[z.tolist() for z in memberships],'monthly_metrics':ms,'summary':summary,'fit':info})
                write_json(out/'temporal_summary.json',variants);print(json.dumps(summary),flush=True)
        stability=[];events=[]
        for omega in s['omegas']:
            names=[f'omega{omega:g}_seed{seed}' for seed in s['seeds']];pairs=[]
            for i in range(len(names)):
                for j in range(i):pairs.extend(adjusted_rand_score(a,b) for a,b in zip(series[names[i]][0],series[names[j]][0]))
            votes,ambiguous_votes,unanimous=transition_votes(np.stack([series[name][1] for name in names]),np.stack([series[name][2] for name in names]))
            stability.append({'omega':omega,'pairwise_seed_monthly_ARI_mean':float(np.mean(pairs)) if pairs else None,'pairwise_seed_monthly_ARI_min':float(np.min(pairs)) if pairs else None,'unanimous_transition_events':int(unanimous.sum()),'territories_with_unanimous_transition':int(unanimous.any(axis=0).sum()),'any_seed_transition_events':int((votes>0).sum()),'ambiguous_vertex_months':int((ambiguous_votes>0).sum()),'ambiguous_seed_vertex_months':int(ambiguous_votes.sum()),'unanimous_among_any':float(unanimous.sum()/(votes>0).sum()) if (votes>0).any() else None})
            for t,j in zip(*np.where((votes>0)|(ambiguous_votes>0))):
                events.append({'omega':omega,'period':selected[t+1][0],'entity_id':ids[j],'seed_votes':int(votes[t,j]),'ambiguous_seed_count':int(ambiguous_votes[t,j]),'unanimous_unambiguous_change':bool(unanimous[t,j]),'seed_count':len(names),'original_low_marketplace_profile':bool(z0[j]==3)})
        write_json(out/'temporal_stability.json',stability);pd.DataFrame(events).to_csv(out/'transition_events.csv',index=False)
    else:raise ValueError('Unknown round2 stage')
    if rows:pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows]).to_csv(out/'metrics.csv',index=False)
    result={'status':'completed','stage':stage,'seconds':time.perf_counter()-start,'n':len(ids),'new_fits':2 if stage=='region_control' else len(s['ks'])*len(s['seeds']) if stage=='dmon' else sum(not r['reused'] for r in variants)}
    write_json(out/'summary.json',result);write_json(out/'status.json',result);print(json.dumps(result),flush=True)
    return out
