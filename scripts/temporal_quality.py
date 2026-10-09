"""Score monthly partitions with six ICVI, check them on the next month's graph, choose omega and explain fixed-centre drift."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/temporal_quality.json')
    p.add_argument('--output',help='Write to this directory instead of temporal_quality.output')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text('utf-8'))
    if args.output:cfg['temporal_quality']['output']=args.output
    from sbercluster.temporal_quality import run
    print(run(cfg,ROOT))
