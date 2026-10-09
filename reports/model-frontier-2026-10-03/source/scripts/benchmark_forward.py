"""Diagnostic chronological 2023 transfers with genuinely training-only scaling.

Already-inspected 2023 windows are sensitivity checks, not untouched holdout.
Fit Jan-Jun -> evaluate Jul-Sep; fit Jan-Sep -> evaluate Oct-Dec. Each window
fits its own frozen IQR/centres using the training months and same finite specs.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster.features import make_slices, transform_frozen
from sbercluster.graph import knn_graph
from sbercluster.io import sha256, write_json
from sbercluster.metrics import all_metrics
from sbercluster.research import annual_profile, load_development
from sbercluster.selection_frontier import fit_frontier, predict_frontier

SOURCES = ['scripts/benchmark_forward.py', 'sbercluster/selection_frontier.py',
           'sbercluster/features.py', 'sbercluster/metrics.py', 'sbercluster/graph.py']


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / 'status.json', {'status': 'running'})
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8'))
    raw, _, _, ids, _, panel_manifest = load_development(ROOT, cfg)
    screen = ROOT / '.local/frontier-20261003/benchmark-constrained'
    previous = {r['id']: r for r in json.loads((screen / 'screen.json').read_text('utf-8'))}
    selected = json.loads((screen / 'shortlist.json').read_text('utf-8'))['ids']
    models, partitions, rows = {}, {}, []
    for train_end, test_months in [('2023-06-01', ['2023-07-01', '2023-08-01', '2023-09-01']),
                                    ('2023-09-01', ['2023-10-01', '2023-11-01', '2023-12-01'])]:
        local_cfg = deepcopy(cfg)
        local_cfg['features']['calibration_end'] = train_end
        local_cfg['validation']['development_end'] = train_end
        training = raw[raw.period <= train_end].copy()
        slices, scaler = make_slices(training, local_cfg)
        if any(s[1] != ids for s in slices):
            raise ValueError('Training IDs differ')
        train_monthly = np.stack([s[2] for s in slices])
        x_train = annual_profile(train_monthly)
        train_distances = cdist(x_train, x_train)
        train_graph, _ = knn_graph(x_train, 15, 'union')
        test_tensors = []
        graphs = []
        for month in test_months:
            frame = raw[raw.period == month].sort_values('entity_id')
            if frame.entity_id.tolist() != ids:
                raise ValueError('Test IDs differ')
            features = transform_frozen(frame, scaler)
            test_tensors.append(features)
            graphs.append(knn_graph(features, 15, 'union')[0])
        quarter = annual_profile(np.stack(test_tensors))
        quarter_graph, _ = knn_graph(quarter, 15, 'union')
        for name in selected:
            z_train, model, fit = fit_frontier(x_train, previous[name]['spec'], distances=train_distances)
            train_metrics = all_metrics(x_train, z_train, train_graph)
            quarterly_labels = predict_frontier(quarter, model)
            models[train_end + '__' + name] = {'scaler': scaler, 'rule': model}
            partitions[train_end + '__' + name + '__quarter'] = quarterly_labels
            row = {'train_end': train_end, 'id': name, 'train_months': [s[0] for s in slices],
                   'test_months': test_months, 'scaler': scaler, 'fit': fit,
                   'training_metrics': train_metrics,
                   'test_quarter_metrics': all_metrics(quarter, quarterly_labels, quarter_graph), 'monthly': []}
            for month, features, graph in zip(test_months, test_tensors, graphs):
                labels = predict_frontier(features, model)
                partitions[train_end + '__' + name + '__' + month] = labels
                row['monthly'].append({'period': month, 'metrics': all_metrics(features, labels, graph),
                                       'cluster_sizes': np.bincount(labels, minlength=4).tolist()})
            rows.append(row)
            write_json(output / 'results.json', rows)
            print(json.dumps({'train_end': train_end, 'id': name, 'train_SW': train_metrics['SW'],
                              'test_quarter_SW': row['test_quarter_metrics']['SW'],
                              'test_monthly_SW_mean': float(np.mean([r['metrics']['SW'] for r in row['monthly']]))}), flush=True)
    summary = []
    for train_end in sorted(set(r['train_end'] for r in rows)):
        cases = {r['id']: r for r in rows if r['train_end'] == train_end}
        baseline = cases[selected[0]]
        for name in selected:
            case = cases[name]
            monthly_gains = [a['metrics']['SW'] - b['metrics']['SW'] for a, b in zip(case['monthly'], baseline['monthly'])]
            summary.append({'train_end': train_end, 'id': name,
                            'quarter_SW_gain': case['test_quarter_metrics']['SW'] - baseline['test_quarter_metrics']['SW'],
                            'monthly_SW_gains': monthly_gains, 'monthly_SW_gain_mean': float(np.mean(monthly_gains)),
                            'CH_ratios_monthly': [a['metrics']['CH'] / b['metrics']['CH'] for a, b in zip(case['monthly'], baseline['monthly'])],
                            'S_Dbw_monthly': [a['metrics']['S_Dbw'] for a in case['monthly']],
                            'min_occupancy_monthly': [a['metrics']['min_cluster_size'] for a in case['monthly']]})
    write_json(output / 'summary.json', summary)
    write_json(output / 'models.json', models)
    np.savez_compressed(output / 'labels.npz', **partitions)
    write_json(output / 'provenance.json', {'ids': ids, 'panel_sha256': panel_manifest['panel_sha256'],
                  'screen_manifest_sha256': sha256(screen / 'manifest.json'),
                  'selection': selected, 'calibration': 'each chronological training window only',
                  'scope': 'already_inspected_2023_diagnostic_forward_window_sensitivity',
                  'cohort': 'retrospectively_complete_2023_2024_panel',
                  'source_sha256': {name: sha256(ROOT / name) for name in SOURCES}})
    for name in SOURCES:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    write_json(output / 'status.json', {'status': 'completed', 'elapsed_seconds': time.perf_counter() - started})
    write_json(output / 'manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): sha256(p)
                                                         for p in sorted(output.rglob('*')) if p.is_file()}})


def main():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot set BelowNormal worker priority')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            run(args)
    except BaseException as exc:
        path = Path(args.output) / 'status.json'
        if path.exists():
            write_json(path, {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
