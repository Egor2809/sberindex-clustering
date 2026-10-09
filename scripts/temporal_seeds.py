"""Refit temporal Leiden of the main model with 20 seeds and measure agreement and the stability of its seven groups."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = [1729, 2718, 3141, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/round2_temporal_grid.json', help='Configuration of the main temporal Leiden run')
    p.add_argument('--groups', default='configs/temporal_groups.json', help='Main model: omega')
    p.add_argument('--comparison', default='configs/method_comparison.json', help='Tracking settings and published seeds')
    p.add_argument('--profiles', default='reports/temporal-groups/profiles.json')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', default='reports/temporal-seeds')
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text('utf-8'))
    cfg['clustering']['temporal_coupling_relative'] = json.loads((ROOT / args.groups).read_text('utf-8'))['omega']
    mc = json.loads((ROOT / args.comparison).read_text('utf-8'))['method_comparison']
    profiles = json.loads((ROOT / args.profiles).read_text('utf-8'))
    names = {g['cells']: g['name'] for g in profiles['groups']}
    from sbercluster.temporal_seeds import run
    print(run(cfg, ROOT, mc, args.seeds, args.output, args.workers, names))
