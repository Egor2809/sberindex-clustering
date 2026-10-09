"""Compare edge rules, tabulate ICVI for published partitions and track cluster events."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/edge_rules.json')
    p.add_argument('--output',help='Write to this directory instead of edge_rules.output')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    if args.output:cfg['edge_rules']['output']=args.output
    from sbercluster.edge_study import run
    print(run(cfg,ROOT))
