"""Build the network-core typology: Leiden consensus on a profile plus co-movement graph, judged out of time."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/network_core.json')
    p.add_argument('--output',help='Write to this directory instead of network_core.output')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    if args.output:cfg['network_core']['output']=args.output
    from sbercluster.network_core import run
    print(run(cfg,ROOT))
