"""Bounded prototype search and 2023-only paired sensitivity benchmark.

python -m scripts.benchmark_frontier --output runs/frontier --stage screen
python -m scripts.benchmark_frontier --output runs/frontier-validation --stage validate --screen runs/frontier
Use new destinations. Historical archives and the production frozen model are
never written. The complete output includes negative outcomes and model rules.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster.graph import knn_graph
from sbercluster.io import sha256, write_json
from sbercluster.metrics import all_metrics
from sbercluster.published_inputs import read_archive
from sbercluster.research import annual_profile, block_month_indices, load_development
from sbercluster.selection_frontier import REVISION, fit_frontier, predict_frontier, silhouette_from_distances

SOURCE_FILES = ['sbercluster/selection_frontier.py', 'scripts/benchmark_frontier.py',
                'sbercluster/research.py', 'sbercluster/features.py', 'sbercluster/metrics.py',
                'sbercluster/graph.py', 'sbercluster/published_inputs.py']
POLICY = {'min_fraction': .03, 'full_SW_gain': .005, 'CH_ratio': .95,
          'AVI_loss': .03, 'AVU_gain': .03, 'S_Dbw_ratio': 1.10,
          'transfer_SW_gain': .002, 'paired_region_CI_lower': 0., 'stability_ARI_loss': .02}
LIMITATIONS = [
    'All selection and sensitivity use already-inspected 2023, frozen 2023 scaling and retrospective full-panel cohort.',
    'Quarterly transfers and month bootstraps are sensitivity checks, not untouched holdout or inferential certification after selection.',
    'Regional bootstrap conditions on selected methods and within-period scores; it does not establish independent economic validity.',
    'SW tuning is a bounded coordinate search of deployable decision rules, not proof of global optimum.',
    'Minimum 3% cluster occupancy and acceptance tolerances are practical continuation policies, not contest rules.',
    'MQ remains undefined by organizer; TurboMQ and NewmanQ retain their distinct names.',
]


def specs():
    common = {'k': 4, 'min_fraction': .03}
    result = [{'id': 'historical_kmeans4', **common}]
    result += [{'id': f'huber_q{int(q*100)}', **common, 'huber_quantile': q} for q in (.25, .5, .75)]
    result += [{'id': f'covariance_s{int(s*100)}', **common, 'shrinkage': s} for s in (.25, .5, .75)]
    result += [{'id': f'sw_tuned_{name}', **common, 'tune_silhouette': True, 'max_passes': 6,
                **extra} for name, extra in [('ordinary', {}), ('huber50', {'huber_quantile': .5}),
                                            ('covariance50', {'shrinkage': .5})]]
    return result


def full_gates(metrics, baseline):
    gates = {
        'occupancy': metrics['min_cluster_size'] >= int(np.ceil(metrics['n'] * POLICY['min_fraction'])),
        'SW': metrics['SW'] - baseline['SW'] >= POLICY['full_SW_gain'],
        'CH': metrics['CH'] >= baseline['CH'] * POLICY['CH_ratio'],
        'AVI': metrics['AVI'] >= baseline['AVI'] - POLICY['AVI_loss'],
        'AVU': metrics['AVU'] <= baseline['AVU'] + POLICY['AVU_gain'],
        'S_Dbw': (metrics['S_Dbw'] is not None and baseline['S_Dbw'] is not None
                  and metrics['S_Dbw'] <= baseline['S_Dbw'] * POLICY['S_Dbw_ratio']),
    }
    return gates


def baseline_model(x, labels):
    model = {'revision': REVISION, 'spec': specs()[0], 'transform': np.eye(x.shape[1]).tolist(),
             'centers': [x[labels == c].mean(axis=0).tolist() for c in range(4)], 'biases': [0.] * 4}
    if not np.array_equal(labels, predict_frontier(x, model)):
        raise ValueError('Archived KMeans labels are not reproduced by frozen centroid prediction')
    return model


def region_interval(samples, regions, seed=1903, repetitions=500):
    unique, code = np.unique(regions, return_inverse=True)
    counts = np.bincount(code)
    sums = np.bincount(code, weights=samples)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'difference': float(samples.mean()), 'CI95': np.quantile(values, [.025, .975]).tolist(),
            'regions': len(unique), 'repetitions': repetitions, 'seed': seed}


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    start = time.perf_counter()
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    write_json(output / 'status.json', {'status': 'running', 'stage': args.stage})
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8'))
    development, _, monthly, ids, scaler, manifest = load_development(ROOT, cfg)
    x = annual_profile(monthly)
    regions = development.drop_duplicates('entity_id').set_index('entity_id').loc[ids, 'region_code'].to_numpy()
    historical = read_archive(ROOT, 'review-20260924-joint', 'labels.json.gz')
    if historical['ids'] != ids:
        raise ValueError('Archived baseline IDs differ')
    labels0 = np.asarray(historical['labels']['joint_alpha0_k4'], dtype=np.int32)
    distance = cdist(x, x)
    graph, graph_info = knn_graph(x, 15, 'union')
    provenance = {'started_at': datetime.now(timezone.utc).isoformat(), 'stage': args.stage,
                  'revision': REVISION, 'source_sha256': sources, 'panel_sha256': manifest['panel_sha256'],
                  'config': cfg, 'scaler': scaler, 'ids': ids, 'policy': POLICY,
                  'limitations': LIMITATIONS, 'graph': graph_info,
                  'environment': {name: importlib.metadata.version(name) for name in
                                  ('numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl')}}
    provenance['python_executable'] = sys.executable
    provenance['resources'] = {'threads': 1, 'priority': 'BelowNormal' if os.name == 'nt' else 'default',
                               'pid': os.getpid(), 'gpu_used': False}
    write_json(output / 'provenance.json', provenance)
    np.save(output / 'features.npy', x, allow_pickle=False)
    rows, models, labels = [], {}, {}
    try:
        if args.stage == 'screen':
            baseline = all_metrics(x, labels0, graph)
            for spec in specs():
                t = time.perf_counter()
                if spec['id'] == 'historical_kmeans4':
                    z, model, info = labels0, baseline_model(x, labels0), {'historical_labels_reused': True}
                else:
                    z, model, info = fit_frontier(x, spec, initial_labels=labels0, distances=distance)
                metrics = all_metrics(x, z, graph)
                row = {'id': spec['id'], 'spec': spec, 'metrics': metrics, 'fit': info,
                       'ARI_to_historical': float(adjusted_rand_score(labels0, z)),
                       'full_gates': full_gates(metrics, baseline), 'elapsed_seconds': time.perf_counter() - t}
                rows.append(row); models[spec['id']] = model; labels[spec['id']] = z.tolist()
                write_json(output / 'screen.json', rows)
                write_json(output / 'models.json', models)
                write_json(output / 'labels.json', {'ids': ids, 'labels': labels})
                print(json.dumps({'id': spec['id'], 'SW': metrics['SW'], 'CH': metrics['CH'],
                                  'sizes': np.bincount(z).tolist(), 'gates': row['full_gates'],
                                  'seconds': row['elapsed_seconds']}), flush=True)
            eligible = [r for r in rows[1:] if r['full_gates']['occupancy']]
            # Advance up to three full-gate passes; if none pass, keep the two
            # best-SW challengers as explicit negative/diagnostic comparisons.
            winners = sorted([r for r in eligible if all(r['full_gates'].values())],
                             key=lambda r: r['metrics']['SW'], reverse=True)
            shortlist = (winners[:3] or sorted(eligible, key=lambda r: r['metrics']['SW'], reverse=True)[:2])
            write_json(output / 'shortlist.json', {'ids': ['historical_kmeans4'] + [r['id'] for r in shortlist],
                                                  'has_full_gate_pass': bool(winners)})
        else:
            if args.screen is None:
                raise ValueError('--screen required for validation')
            screen = Path(args.screen).resolve()
            previous_manifest = json.loads((screen / 'manifest.json').read_text('utf-8'))
            for name in ('screen.json', 'shortlist.json', 'labels.json', 'provenance.json'):
                if sha256(screen / name) != previous_manifest['files_sha256'][name]:
                    raise ValueError('Screen fingerprint differs: ' + name)
            prev = json.loads((screen / 'provenance.json').read_text('utf-8'))
            if prev['source_sha256'] != sources or prev['panel_sha256'] != manifest['panel_sha256']:
                raise ValueError('Screen source/input snapshot differs')
            provenance['screen_manifest_sha256'] = sha256(screen / 'manifest.json')
            write_json(output / 'provenance.json', provenance)
            previous = {r['id']: r for r in json.loads((screen / 'screen.json').read_text('utf-8'))}
            selected = json.loads((screen / 'shortlist.json').read_text('utf-8'))['ids']
            reference_labels = json.loads((screen / 'labels.json').read_text('utf-8'))['labels']
            transfer = {name: [] for name in selected}
            stability = {name: [] for name in selected}
            transfer_samples = {name: [] for name in selected}
            for fold in range(4):
                test = np.arange(fold * 3, fold * 3 + 3)
                train = np.setdiff1d(np.arange(12), test)
                x_train, x_test = annual_profile(monthly[train]), annual_profile(monthly[test])
                train_distance, test_distance = cdist(x_train, x_train), cdist(x_test, x_test)
                for name in selected:
                    t = time.perf_counter()
                    _, model, info = fit_frontier(x_train, previous[name]['spec'], distances=train_distance)
                    z_test = predict_frontier(x_test, model)
                    sw, sample = silhouette_from_distances(test_distance, z_test)
                    transfer[name].append({'fold': fold, 'train_month_indices': train.tolist(),
                                           'test_month_indices': test.tolist(), 'SW': sw,
                                           'cluster_sizes': np.bincount(z_test, minlength=4).tolist(),
                                           'fit': info, 'seconds': time.perf_counter() - t})
                    transfer_samples[name].append(sample)
                    np.save(output / f'{name}__fold{fold}.npy', z_test, allow_pickle=False)
                    print(json.dumps({'id': name, 'fold': fold, 'transfer_SW': sw,
                                      'seconds': time.perf_counter() - t}), flush=True)
                write_json(output / 'transfer.json', transfer)
            for draw in range(args.bootstrap):
                indices = block_month_indices(2200 + draw)
                xb = annual_profile(monthly[indices])
                db = cdist(xb, xb)
                for name in selected:
                    z, _, info = fit_frontier(xb, previous[name]['spec'], distances=db)
                    stability[name].append({'draw': draw, 'seed': 2200 + draw, 'month_indices': indices.tolist(),
                                            'ARI': float(adjusted_rand_score(reference_labels[name], z)),
                                            'cluster_sizes': np.bincount(z, minlength=4).tolist(),
                                            'fit': info})
                    np.save(output / f'{name}__month_draw{draw}.npy', z, allow_pickle=False)
                write_json(output / 'stability.json', stability)
                print(json.dumps({'bootstrap_draw': draw, 'ARI': {name: stability[name][-1]['ARI']
                                                                 for name in selected}}), flush=True)
            reference = selected[0]
            baseline_ari = float(np.median([r['ARI'] for r in stability[reference]]))
            for name in selected:
                diffs = np.mean(transfer_samples[name], axis=0) - np.mean(transfer_samples[reference], axis=0)
                interval = region_interval(diffs, regions)
                median_ari = float(np.median([r['ARI'] for r in stability[name]]))
                gates = {**previous[name]['full_gates'],
                         'transfer_gain': interval['difference'] >= POLICY['transfer_SW_gain'],
                         'transfer_region_CI': interval['CI95'][0] > POLICY['paired_region_CI_lower'],
                         'bootstrap_ARI': median_ari >= baseline_ari - POLICY['stability_ARI_loss']}
                rows.append({'id': name, 'full_metrics': previous[name]['metrics'],
                             'transfer_SW_mean': float(np.mean([r['SW'] for r in transfer[name]])),
                             'paired_transfer': interval, 'bootstrap_ARI_median': median_ari,
                             'bootstrap_ARI_range': [float(min(r['ARI'] for r in stability[name])),
                                                     float(max(r['ARI'] for r in stability[name]))],
                             'gates': gates, 'accepted_continuation': all(gates.values())})
            write_json(output / 'validation.json', rows)
        elapsed = time.perf_counter() - start
        write_json(output / 'status.json', {'status': 'completed', 'stage': args.stage, 'elapsed_seconds': elapsed})
        paths = sorted(p for p in output.rglob('*') if p.is_file())
        write_json(output / 'manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in paths}})
        print(json.dumps({'status': 'completed', 'output': str(output), 'seconds': elapsed}), flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(exc, (SystemExit, KeyboardInterrupt))
                                           else 'failed', 'stage': args.stage, 'error': str(exc)})
        raise


def main():
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot set BelowNormal worker priority')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--stage', choices=('screen', 'validate'), default='screen')
    p.add_argument('--screen')
    p.add_argument('--bootstrap', type=int, default=12)
    args = p.parse_args()
    if args.bootstrap < 1:
        p.error('--bootstrap must be positive')
    with threadpool_limits(limits=1):
        run(args)


if __name__ == '__main__':
    main()
