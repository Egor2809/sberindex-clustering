"""Synthetic transition tracking: planted groups and switches, temporal Leiden against monthly and frozen KMeans."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/round2_temporal_grid.json', help='Configuration of the main temporal Leiden run')
    p.add_argument('--groups', default='configs/temporal_groups.json', help='Main model: omega, seed and labels')
    p.add_argument('--profiles', default='reports/temporal-groups/profiles.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103, 104, 105])
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--output', default='reports/synthetic-transitions')
    args = p.parse_args()
    cfg = json.loads((ROOT / args.config).read_text('utf-8'))
    main = json.loads((ROOT / args.groups).read_text('utf-8'))
    cfg['seed'] = main['seed']
    cfg['clustering']['temporal_coupling_relative'] = main['omega']
    from sbercluster.synthetic_transitions import run
    print(run(cfg, ROOT, args.output, args.seeds, args.profiles, main['temporal_labels'], args.workers))
