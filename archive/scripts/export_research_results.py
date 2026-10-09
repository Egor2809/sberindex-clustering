"""Export audited, completed research runs for publication without fitting models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main(args):
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    inputs={name:ROOT/getattr(args,name) for name in ('annual','months','stability')}
    if args.validation:inputs['validation']=ROOT/args.validation
    for name,run in inputs.items():
        status=json.loads((run/'status.json').read_text('utf-8'))
        if status['status']!='completed':raise ValueError(f'Incomplete run: {run.name}')
        provenance=json.loads((run/'provenance.json').read_text('utf-8'))
        dest=output/'provenance'/run.name;dest.mkdir(parents=True,exist_ok=True)
        for filename in ('provenance.json','status.json','graphs.json'):
            if (run/filename).exists():
                if filename == 'provenance.json':
                    public = json.loads((run/filename).read_text('utf-8'))
                    public.get('config', {}).get('execution', {}).pop('resource_profile', None)
                    dump(dest/filename, public)
                else:shutil.copy2(run/filename,dest/filename)
        # Resolve the exact per-module hash, including early runs made before snapshots were copied.
        for relative,expected in provenance['source_hashes'].items():
            if relative in ('scripts/run_bounded.py', 'sbercluster/resources.py'):continue
            payload=None
            for candidate in (run/'implementation_snapshot'/relative,ROOT/relative,dest/'implementation_snapshot'/relative):
                if candidate.exists() and digest(candidate)==expected:
                    payload=candidate.read_bytes();break
            if payload is None:
                result=subprocess.run(['git','show',f"{provenance['git_commit']}:{relative}"],cwd=ROOT,capture_output=True)
                if result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==expected:payload=result.stdout
            if payload is None:raise ValueError(f'Cannot reconstruct exact source: {run.name}/{relative}')
            target=dest/'implementation_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    rows=[]
    for name in ('annual','months'):
        rows.extend({**r,'source_run':inputs[name].name} for r in json.loads((inputs[name]/'metrics.json').read_text('utf-8')))
    if len(rows)!=104 or len({(r['representation'],r['candidate']) for r in rows})!=104:
        raise ValueError('Expected 104 distinct prespecified comparisons')
    dump(output/'screen_metrics.json',rows)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]).to_csv(output/'screen_metrics.csv',index=False)
    for filename in ('stability.json','reference_metrics.json','membership_stability.json'):
        shutil.copy2(inputs['stability']/filename,output/filename)
    stability=json.loads((output/'stability.json').read_text('utf-8'))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in stability]).to_csv(output/'stability.csv',index=False)
    ids=json.loads((output/'membership_stability.json').read_text('utf-8'))['ids']
    parts=[]
    for name in ('annual','months'):
        for row in json.loads((inputs[name]/'metrics.json').read_text('utf-8')):
            if row['status']!='completed':continue
            labels=np.load(inputs[name]/'labels'/f"{row['representation']}__{row['candidate']}.npy",allow_pickle=False)
            parts.extend({'entity_id':entity,'representation':row['representation'],'candidate':row['candidate'],'cluster':int(label)} for entity,label in zip(ids,labels))
    pd.DataFrame(parts).to_csv(output/'screen_partitions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    reference=np.load(inputs['stability']/'labels/kmeans_k4__reference.npy',allow_pickle=False)
    pd.DataFrame({'entity_id':ids,'cluster':reference}).to_csv(output/'reference_assignments.csv',index=False)
    if args.validation:
        dest=output/'validation';dest.mkdir(exist_ok=True)
        for source in inputs['validation'].iterdir():
            if source.is_file() and source.suffix in ('.json','.csv'):
                if source.name == 'provenance.json':
                    public=json.loads(source.read_text('utf-8'));public.get('config',{}).get('execution',{}).pop('resource_profile',None);dump(dest/source.name,public)
                else:shutil.copy2(source,dest/source.name)
    summary={'n_territories':len(ids),'screen_partitions':len(rows),'stability_perturbations':len(stability),
             'selected_candidate':'kmeans_k4','source_runs':{k:v.name for k,v in inputs.items()},
             'holdout_completed':bool(args.validation),'panel_sha256':json.loads((inputs['annual']/'provenance.json').read_text())['panel_sha256']}
    dump(output/'summary.json',summary)
    dump(output/'SHA256.json',{str(p.relative_to(output)).replace('\\','/'):digest(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annual',default='runs/research-20260923T123427735210Z')
    parser.add_argument('--months',default='runs/research-20260923T123523157700Z')
    parser.add_argument('--stability',default='runs/research-20260923T123755575253Z')
    parser.add_argument('--validation')
    parser.add_argument('--output',default='reports/experiments/2026-09-23-v2')
    main(parser.parse_args())
