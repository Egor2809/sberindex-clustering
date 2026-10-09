"""Compare two edge-rule result directories, ignoring timings and provenance."""
import argparse
import json
import math
from pathlib import Path
import sys

FILES=['edge_rules.json','k_sensitivity.json','icvi_table.json','icvi_agreement.json','cluster_events.json']


def close(a,b,tolerance,path):
    if isinstance(a,dict) and isinstance(b,dict):
        if a.keys()!=b.keys():return [path+': keys differ']
        return [d for key in a if key!='timings_seconds' for d in close(a[key],b[key],tolerance,path+'.'+key)]
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [path+': lengths differ']
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in close(x,y,tolerance,f'{path}[{i}]')]
    if isinstance(a,float) or isinstance(b,float):
        return [] if a is not None and b is not None and math.isclose(a,b,rel_tol=tolerance,abs_tol=tolerance) else [f'{path}: {a} != {b}']
    return [] if a==b else [f'{path}: {a} != {b}']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('published',type=Path)
    p.add_argument('repeated',type=Path)
    p.add_argument('--tolerance',type=float,default=1e-4)
    p.add_argument('--min-ari',type=float,default=0.99)
    args=p.parse_args()
    differences=[d for name in FILES for d in close(json.loads((args.published/name).read_text('utf-8')),
                 json.loads((args.repeated/name).read_text('utf-8')),args.tolerance,name)]
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    a,b=(pd.read_csv(d/'edge_rule_partitions.csv.gz').set_index('entity_id') for d in (args.published,args.repeated))
    if list(a.columns)!=list(b.columns) or list(a.index)!=list(b.index):differences.append('edge_rule_partitions.csv.gz: layout differs')
    else:differences+=[f'{c}: ARI {adjusted_rand_score(a[c],b[c]):.4f}' for c in a.columns if adjusted_rand_score(a[c],b[c])<args.min_ari]
    print('\n'.join(differences[:20]) or 'Edge-rule results match the published files')
    sys.exit(1 if differences else 0)
