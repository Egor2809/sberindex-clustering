"""Independent actual sector-fold, prediction and sharp-bound audit; no fit."""
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
RUN=ROOT/'.local/core-round3-20261003/economic/strict-sectors'
WAGE=ROOT/'.local/core-round3-20261003/economic/strict-wage'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
    spec=json.loads((RUN/'protocol.json').read_text('utf-8')); status=json.loads((RUN/'status.json').read_text('utf-8'))
    result=json.loads((RUN/'results.json').read_text('utf-8'))
    assert stable(spec)==status['protocol_sha256']==result['protocol_sha256']
    assert digest(RUN/'results.json')==status['results_sha256']
    assert digest(RUN/'all-eligible-oof-predictions.csv')==status['predictions_sha256']
    for name,sha in spec['source_sha256'].items(): assert digest(RUN/'source'/name)==sha==digest(ROOT/name)
    for name,sha in spec['input_sha256'].items(): assert digest(ROOT/name)==sha
    ids=np.array(json.loads((ROOT/'reports/external-national-2026-10-03/frozen-labels.json').read_text('utf-8'))['ids'])
    cohort=pd.read_csv(ROOT/'reports/external-national-2026-10-03/cohort-2023.csv').set_index('entity_id').reindex(ids)
    eligible=np.array([i in set(spec['eligible_entity_ids']) for i in ids]); regions=cohort.region_code.to_numpy(int)
    assert len(spec['eligible_entity_ids'])==1797 and np.unique(regions[eligible]).tolist()==spec['eligible_regions']
    predictions=pd.read_csv(RUN/'all-eligible-oof-predictions.csv')
    variants=spec['representation_variants']; outcomes=spec['outcomes']
    assert len(predictions)==1797*3 and not predictions.duplicated(['entity_id','outcome']).any()
    assert np.isfinite(predictions[variants].to_numpy()).all()
    assert ((predictions[variants]>=0)&(predictions[variants]<=1)).all().all()
    metadata=[json.loads(line) for line in (RUN/'fold-probe-metadata.jsonl').read_text('utf-8').splitlines()]
    assert len(metadata)==216
    for item in metadata:
        r,outcome=item['region_code'],item['outcome']; y=cohort[outcome].to_numpy(float)
        train=eligible&(regions!=r)&np.isfinite(y); test=eligible&(regions==r)
        assert ((y[train]>=0)&(y[train]<=1)).all()
        assert item['training_ids']==ids[train].tolist() and item['test_eligible_ids']==ids[test].tolist()
        assert item['type_vocabulary']==sorted(set(cohort.municipal_district_type.to_numpy(str)[train]))
        art=json.loads((WAGE/'folds'/str(r)/'representation.json').read_text('utf-8'))
        assert item['expense_training_ids_sha256']==stable(art['training_entity_ids'])
        assert len(item['probes'])==4
        for name,probe in item['probes'].items():
            assert probe['spec']==spec['learner_spec'] and probe['iterations']==150
            assert probe['fitted_sha256']==stable({k:v for k,v in probe.items() if k!='fitted_sha256'})
        with np.load(WAGE/'folds'/str(r)/'profiles-and-labels.npz',allow_pickle=False) as cache:
            for name in ['kmeans4','huber75']:
                z=cache[name]
                for p in item['training_cluster_profiles'][name]:
                    ix=train&(z==p['label'])
                    assert p['observed_training_n']==int(ix.sum()) and p['observed_training_regions']==len(np.unique(regions[ix]))
                    assert abs(p['mean_share']-y[ix].mean())<1e-14
    bound_error=0.; observed_error=0.; missing_regions={}; h75_continuous={}
    for outcome in outcomes:
        part=predictions[predictions.outcome==outcome].set_index('entity_id').reindex(spec['eligible_entity_ids'])
        assert set(part.index)==set(ids[eligible]) and not part.index.duplicated().any()
        y=part.observed.to_numpy(float); rr=part.region_code.to_numpy(int); known=np.isfinite(y)
        np.testing.assert_allclose(y,cohort.reindex(part.index)[outcome],rtol=0,atol=1e-14,equal_nan=True)
        assert np.array_equal(part.target_observed,known)
        groups=[np.flatnonzero(rr==r) for r in np.unique(rr)]
        missing_regions[outcome]=[int(r) for r in np.unique(rr) if not known[rr==r].any()]
        for row in [r for r in result['observed_scores'] if r['outcome']==outcome]:
            error=y[known]-part[row['predictor']].to_numpy()[known]
            expect=[np.sqrt(np.mean(error**2)),np.mean([np.mean((y[known&(rr==r)]-part[row['predictor']].to_numpy()[known&(rr==r)])**2) for r in np.unique(rr[known])])]
            observed_error=max(observed_error,float(np.max(np.abs(np.array(expect)-[row['observed_municipality_rmse'],row['observed_equal_region_mse']]))))
            assert row['eligible_n']==1797 and row['eligible_regions']==72
        for row in [r for r in result['full_cohort_comparisons'] if r['outcome']==outcome]:
            a,b=part[row['candidate']].to_numpy(),part[row['reference']].to_numpy()
            # Evaluate the original squared losses at BOTH feasible endpoints;
            # each unknown entity can choose its own endpoint independently.
            d0=(0.-b)**2-(0.-a)**2; d1=(1.-b)**2-(1.-a)**2
            lo,hi=np.minimum(d0,d1),np.maximum(d0,d1)
            truth=(y[known]-b[known])**2-(y[known]-a[known])**2
            lo[known]=truth;hi[known]=truth
            expect=[np.mean([lo[g].mean() for g in groups]),np.mean([hi[g].mean() for g in groups]),lo.mean(),hi.mean()]
            stored=[row['equal_region_lower'],row['equal_region_upper'],row['municipality_lower'],row['municipality_upper']]
            bound_error=max(bound_error,float(np.max(np.abs(np.array(expect)-stored))))
            assert row['eligible_n']==1797 and row['eligible_regions']==72 and row['regions_with_no_observed_target']==missing_regions[outcome]
            for p,g in zip(row['per_region'],groups):
                assert p['eligible_n']==len(g) and p['observed_n']==int(known[g].sum())
                np.testing.assert_allclose([p['mean_loss_difference_lower'],p['mean_loss_difference_upper']],[lo[g].mean(),hi[g].mean()],rtol=0,atol=1e-14)
            if row['candidate']!=row['reference']: assert expect[0]<=0<=expect[1]
            if row['candidate']=='huber75' and row['reference']=='continuous5': h75_continuous[outcome]=stored[:2]
    assert bound_error<1e-14 and observed_error<1e-14
    report={'status':'PASS','scope':'Independent fixed actual fold/prediction/endpoint arithmetic, no fit','eligible_per_target':1797,'eligible_regions':72,
        'target_folds':len(metadata),'probes':len(metadata)*4,'observed_scores':12,'bound_comparisons':48,'nonself_bounds_cross_zero':36,
        'max_bound_error':bound_error,'max_observed_score_error':observed_error,'allmissing_target_regions_retained':missing_regions,
        'H75_vs_continuous5_equalregion_bounds':h75_continuous,'protocol_sha256':stable(spec),'results_sha256':digest(RUN/'results.json'),
        'predictions_sha256':digest(RUN/'all-eligible-oof-predictions.csv'),'oracle_sha256':digest(Path(__file__))}
    (OUT/'sector-evidence-review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__': main()
