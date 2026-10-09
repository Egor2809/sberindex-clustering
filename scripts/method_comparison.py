"""Score every stored annual partition on shared axes, marking circular axes; summarise temporal Leiden across seeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/method_comparison.json')
    p.add_argument('--output',help='Write to this directory instead of method_comparison.output')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    if args.output:cfg['method_comparison']['output']=args.output
    from sbercluster.method_comparison import run
    print(run(cfg,ROOT))
