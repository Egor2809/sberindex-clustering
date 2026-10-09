"""Independent real paired A/R, reference bounds/MMD and immutable I/O audit."""
import hashlib,json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np,pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from sbercluster.io import CATEGORIES,TOTAL
from scripts.predict_dual import read_pinned_json,read_panel,write_immutable_json
from verify_dual_contract import order_statistic_bounds
RUN=ROOT/'.local/core-round3-20261003/temporal/paired-v1';OUT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def seal_ok(x):assert x['content_sha256']==stable({k:v for k,v in x.items() if k!='content_sha256'})
def kernel(a,b,sigma):
    total=0.;bb=(b*b).sum(axis=1)
    for start in range(0,len(a),128):
        block=a[start:start+128];distance=np.maximum((block*block).sum(axis=1)[:,None]+bb[None]-2*block@b.T,0)
        total+=np.exp(-distance/(2*sigma**2)).sum()
    return total/(len(a)*len(b))
def main():
    summary=json.loads((RUN/'summary.json').read_text('utf-8'));provenance=json.loads((RUN/'provenance.json').read_text('utf-8'))
    lock=json.loads((RUN/'private-reference-lock.json').read_text('utf-8'));seal_ok(lock)
    assert provenance['reference_file_sha256']==digest(RUN/'private-reference-lock.json')
    panel=pd.read_csv(ROOT/'data/processed/panel.csv',dtype={'entity_id':str,'period':str,'region_code':str})
    assert digest(ROOT/'data/processed/panel.csv')==provenance['panel_sha256']
    ids=lock['entity_ids'];scaler=lock['frozen_artifact']['feature_scaler'];centers=np.asarray(lock['frozen_artifact']['rule']['centers'])
    tensor={}
    for year in [2023,2024]:
        blocks=[]
        for month in range(1,13):
            block=panel[panel.period==f'{year}-{month:02d}-01'].set_index('entity_id').reindex(ids)
            logs=np.log(block[CATEGORIES].to_numpy(float)/block[TOTAL].to_numpy(float)[:,None])
            blocks.append((logs-scaler['ratio_center'])/scaler['ratio_iqr']/np.sqrt(5))
        tensor[year]=np.asarray(blocks)
    np.testing.assert_array_equal(tensor[2023],lock['baseline_features'])
    manifest=json.loads((RUN/'packet_manifest.json').read_text('utf-8'))['packets'];shifts=[]
    for month,(period,item) in enumerate(manifest.items()):
        packet=json.loads((RUN/item['path']).read_text('utf-8'));seal_ok(packet)
        assert digest(RUN/item['path'])==item['sha256'] and packet['reference_lock_sha256']==lock['content_sha256']
        expected=np.median(tensor[2024][month]-tensor[2023][month],axis=0)
        np.testing.assert_allclose(packet['point_paired_median'],expected,rtol=0,atol=1e-14)
        assert packet['coverage']==1 and packet['observed_valid_n']==2016 and packet['status']=='complete_reference'
        shifts.append(expected)
    x0,xa=np.median(tensor[2023],axis=0),np.median(tensor[2024],axis=0)
    xr=np.median(tensor[2024]-np.asarray(shifts)[:,None],axis=0)
    nearest=lambda x:((x[:,None]-centers[None])**2).sum(axis=2).argmin(axis=1)
    z0,za,zr=[nearest(x) for x in [x0,xa,xr]]
    assignments=pd.read_csv(RUN/'assignments.csv')
    assert assignments.entity_id.tolist()==ids
    np.testing.assert_array_equal(assignments.absolute_label,za);np.testing.assert_array_equal(assignments.relative_label,zr)
    assert all(assignments.transition_status=='unresolved')
    assert int((zr!=z0).sum())==summary['paired_relative_changes']==218
    assert abs(adjusted_rand_score(z0,zr)-summary['paired_relative_ARI'])<1e-14
    assert np.bincount(zr).min()==69
    delta=tensor[2024][0]-tensor[2023][0];rng=np.random.default_rng(1729)
    orders={'random':rng.permutation(len(ids)),'upper_delta_tail':np.argsort(delta[:,0],kind='stable')[::-1]}
    controls=json.loads((RUN/'missing_reference_controls.json').read_text('utf-8'))
    for row in controls:
        name=row['scenario']
        if name.startswith('leave_region_'):
            keep=np.array(lock['regions'])!=name.removeprefix('leave_region_')
        else:
            kind,fraction=name.rsplit('_',1);keep=np.ones(len(ids),bool);keep[orders[kind][:int(np.ceil(len(ids)*float(fraction)))]]=False
        lo,hi=order_statistic_bounds(delta[keep],len(ids))
        stored_lo=[-np.inf if v is None else v for v in row['lower']];stored_hi=[np.inf if v is None else v for v in row['upper']]
        np.testing.assert_allclose(lo,stored_lo,rtol=0,atol=1e-14);np.testing.assert_allclose(hi,stored_hi,rtol=0,atol=1e-14)
        assert row['observed_n']==int(keep.sum()) and row['reference_n']==2016
        assert (np.median(delta,axis=0)>=lo).all() and (np.median(delta,axis=0)<=hi).all()
    sigma=float(np.median(pdist(x0)));assert abs(sigma-summary['mmd_bandwidth_fit2023'])<1e-14
    basekernel=kernel(x0,x0,sigma)
    mmd=[basekernel+kernel(x,x,sigma)-2*kernel(x0,x,sigma) for x in [xa,xr]]
    np.testing.assert_allclose(mmd,[summary['absolute_mmd2'],summary['relative_mmd2']],rtol=0,atol=2e-14)
    for name,sha in json.loads((RUN/'manifest.json').read_text('utf-8'))['files_sha256'].items():assert digest(RUN/name)==sha
    # Exact-consumed-byte readers and exclusive writer; no subprocess/fit.
    work=OUT/'dual-cli-control';work.mkdir(exist_ok=False);j,p,o=work/'input.json',work/'panel.csv',work/'immutable.json'
    j.write_text('{"x":[1]}',encoding='utf-8');panel.iloc[:2].to_csv(p,index=False)
    pin=digest(j);input_pin=digest(p);counts={j:0,p:0};original=Path.read_bytes
    def intercepted(path):
        raw=original(path)
        if path in counts:counts[path]+=1;path.write_bytes(b'{"x":[9]}' if path==j else raw+b'\n')
        return raw
    with patch.object(Path,'read_bytes',intercepted):
        assert read_pinned_json(j,pin)=={'x':[1]};frame,sha=read_panel(p);assert sha==input_pin and len(frame)==2
    assert all(v==1 for v in counts.values())
    first=write_immutable_json(o,{'label':1})
    try:write_immutable_json(o,{'label':2})
    except FileExistsError:pass
    else:raise AssertionError('Immutable writer overwrote evidence')
    assert digest(o)==first
    report={'status':'PASS','scope':'Independent historical no-fit paired A/R replay; descriptive MMD and exact boxes, no causal/support/future guarantee',
        'absolute_labels':len(za),'paired_relative_labels':len(zr),'immutable_packets':len(manifest),'missing_reference_controls':len(controls),
        'paired_relative_ARI':float(adjusted_rand_score(z0,zr)),'paired_changes':int((zr!=z0).sum()),'minimum_relative_cluster':int(np.bincount(zr).min()),
        'mmd_bandwidth_2023_only':sigma,'MMD2_absolute_relative':mmd,'MMD_scope':'Biased Gaussian V-statistic, no pvalue/causal interpretation',
        'exact_byte_readers_exclusive_writes':'PASS','source_sha256':provenance['source_sha256'],'oracle_sha256':digest(Path(__file__))}
    (OUT/'paired-evidence-review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
