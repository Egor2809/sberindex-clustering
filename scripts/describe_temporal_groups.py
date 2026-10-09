"""Describe stable temporal Leiden groups from stored labels and lifecycle events."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/temporal_groups.json')
    p.add_argument('--output',help='Write to this directory instead of the configured output')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    from sbercluster.temporal_groups import describe,readme
    summary=describe(cfg,ROOT)
    types=json.loads((ROOT/cfg['type_names']).read_text('utf-8'))['cluster_names']
    out=ROOT/(args.output or cfg['output']);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n','utf-8',newline='\n')
    (out/'README.md').write_text(readme(summary,types),'utf-8',newline='\n')
    print(json.dumps({'groups':len(summary['groups']),'ari_median':summary['ari_reference']['median'],'multi_group_share':summary['multi_group_share']}))
