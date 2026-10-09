"""Run one historical research_v2 stage or compare its KMeans4 labels."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS')


def partition_equivalent(expected, actual):
    """Return whether two label vectors differ only by a one-to-one renaming."""
    if len(expected) != len(actual):
        return False
    forward, reverse = {}, {}
    for left, right in zip(expected, actual):
        if (left in forward and forward[left] != right) or (right in reverse and reverse[right] != left):
            return False
        forward[left] = right
        reverse[right] = left
    return True


def compare_labels(run, published):
    import numpy as np

    run = Path(run)
    state = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if state.get('status') != 'completed' or state.get('phase') != 'screen':
        raise ValueError('Completed research screen required for label comparison')
    ids = json.loads((run / 'ids.json').read_text(encoding='utf-8'))
    labels = np.load(run / 'labels/annual_2023__kmeans_k4.npy', allow_pickle=False)
    if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Run identifiers must be unique nonempty strings')
    if labels.shape != (len(ids),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Run labels must contain one integer per entity')
    labels = labels.tolist()
    with Path(published).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected_by_id = {row['entity_id']: int(row['cluster']) for row in rows}
    if len(expected_by_id) != len(rows) or set(ids) != set(expected_by_id):
        raise ValueError('Entity identifiers differ from the published assignment')
    expected = [expected_by_id[entity] for entity in ids]
    exact = expected == labels
    equivalent = partition_equivalent(expected, labels)
    result = {'run': str(run), 'candidate': 'annual_2023__kmeans_k4',
              'entities': len(ids), 'exact_label_match': exact,
              'partition_match_up_to_label_permutation': equivalent,
              'different_numeric_labels': sum(a != b for a, b in zip(expected, labels))}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if exact else 1


def validate_config(cfg):
    if cfg.get('execution', {}).get('task') != 'research_v2':
        raise ValueError('run_research.py accepts only execution.task=research_v2')
    if cfg['execution'].get('allow_clustering') is not True:
        raise PermissionError('The selected research configuration is disabled')
    threads = cfg['execution'].get('threads', 1)
    if type(threads) is not int or threads < 1:
        raise ValueError('execution.threads must be a positive integer')
    return threads


def run_direct(config):
    cfg = json.loads(config.read_text(encoding='utf-8'))
    threads = validate_config(cfg)
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(ROOT))
    from sbercluster.research import run
    print('Portable runner: use OS or container controls when whole-system supervision is required.', flush=True)
    run(cfg, ROOT)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--config', type=Path)
    action.add_argument('--compare-run', type=Path)
    parser.add_argument('--published', type=Path,
                        default=ROOT / 'reports/experiments/2026-09-23-v2/reference_assignments.csv')
    parser.add_argument('--execute-clustering', action='store_true')
    parser.add_argument('--supervisor', metavar='PROFILE',
                        help='Use the optional Windows supervisor with this JSON profile')
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args(argv)
    if args.compare_run:
        if args.execute_clustering or args.supervisor:
            parser.error('comparison does not execute clustering or use a supervisor')
        return compare_labels(args.compare_run, args.published)
    if not args.execute_clustering:
        parser.error('--execute-clustering is required for a research run')
    config = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.supervisor:
        validate_config(json.loads(config.read_text(encoding='utf-8')))
        if os.name != 'nt':
            parser.error('--supervisor is available only on Windows')
        return subprocess.call([sys.executable, str(ROOT / 'scripts/run_bounded.py'),
                                '--config', str(config), '--profile', args.supervisor,
                                '--timeout', str(args.timeout), '--wait-for-headroom', '30',
                                '--pause-for-headroom', '60'], cwd=ROOT)
    return run_direct(config)


if __name__ == '__main__':
    sys.exit(main())
