"""Reproduce exact-centroid joint refinement against immutable published fits.

Example: python -m scripts.verify_joint_refinement --output runs/joint-refinement --bootstrap 500
The destination must be new. This runs eight bounded joint fits and optionally
refreshes the existing exploratory market-access comparison; it does not fit DMoN.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster.graph import knn_graph
from sbercluster.io import sha256, write_json
from sbercluster.joint import ALGORITHM_REVISION, fit_graph_regularized_kmeans, joint_objective
from sbercluster.metrics import network_indices, network_quality_candidates
from sbercluster.research import annual_profile, load_development
from sbercluster.round2 import external_comparison, same_region_graph

JOINT = 'reports/review-2026-09-24/review-20260924-joint'
REGION = 'reports/round2-2026-09-24/round2-20260924-region_control'
SOURCE_FILES = ['sbercluster/' + name + '.py' for name in (
    'joint', 'metrics', 'graph', 'contest_graphs', 'research', 'features',
    'input_contracts', 'round2', 'followup', 'io', 'published_inputs', 'models', 'dynamics')]
SOURCE_FILES += ['scripts/verify_joint_refinement.py', 'scripts/external_validation_v4.py']
LIMITATIONS = [
    'Exact single-vertex stationarity is local; no global optimum is established.',
    'A supplied KMeans start and an archived endpoint can reach distinct local optima.',
    'All comparisons are retrospective on the frozen complete cohort and already-inspected data.',
    'Roads represent accessibility as of 2024-12-31, not observed trips or financial flows.',
    'Market access is exploratory and shares infrastructure content with roads; measurement independence is limited.',
    'Bootstrap intervals condition on fixed partitions and resample regions; they do not validate production types.',
    'Published historical labels, checkpoints, metrics and manifests are preserved.',
]


def read_checked(root, directory, filename):
    path = root / directory / filename
    manifest = json.loads((path.parent / 'manifest.json').read_text('utf-8'))
    if sha256(path) != manifest['files_sha256'][filename]:
        raise ValueError('Historical artifact fingerprint differs: ' + directory + '/' + filename)
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)


def exact_stationarity(x, adjacency, labels, ids=None, alpha=.25, tolerance=1e-12):
    """Enumerate valid vertex transfers and verify negative deltas by direct refit."""
    x = np.asarray(x, dtype=float)
    adjacency = csr_matrix(adjacency, dtype=float, copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Stationarity tolerance must be finite and nonnegative')
    if ids is not None and len(ids) != len(x):
        raise ValueError('Stationarity IDs must match attribute rows')
    # The public evaluator validates finite inputs, graph conventions and labels.
    original = joint_objective(x, adjacency, labels, alpha)
    _, labels = np.unique(np.asarray(labels), return_inverse=True)
    k = len(np.unique(labels))
    counts = np.bincount(labels, minlength=k)
    centers = np.array([x[labels == c].mean(axis=0) for c in range(k)])
    best, moves = 0., []
    for i in range(len(x)):
        old = labels[i]
        if counts[old] == 1:
            continue
        start, end = adjacency.indptr[i:i+2]
        neighbors, weights = adjacency.indices[start:end], adjacency.data[start:end]
        same = np.bincount(labels[neighbors], weights=weights, minlength=k)
        squared = np.square(x[i] - centers).sum(axis=1)
        removal = counts[old] / (counts[old] - 1) * squared[old]
        delta = ((counts / (counts + 1) * squared - removal) / original['TSS']
                 + alpha * (same[old] - same) / original['undirected_edge_mass'])
        delta[old] = 0.
        recipient = int(np.argmin(delta))
        value = float(delta[recipient])
        best = min(best, value)
        if value < -tolerance:
            candidate = labels.copy()
            candidate[i] = recipient
            actual = joint_objective(x, adjacency, candidate, alpha)['objective'] - original['objective']
            if abs(actual - value) > 2e-12:
                raise ArithmeticError('Move delta and directly refitted objective disagree')
            moves.append({'index': i, 'entity_id': ids[i] if ids is not None else None,
                          'from': int(old), 'to': recipient, 'delta': value,
                          'direct_refit_delta': actual})
    return {'tolerance': tolerance, 'improving_vertices': len(moves), 'best_move_delta': best,
            'coordinate_stationary': not moves, 'moves': moves}


def matched_changes(old, new):
    _, a = np.unique(old, return_inverse=True)
    _, b = np.unique(new, return_inverse=True)
    table = np.zeros((len(np.unique(a)), len(np.unique(b))), int)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    return int(len(a) - table[rows, cols].sum())


def graph_metrics(graph, labels):
    return {**network_indices(graph, labels), **network_quality_candidates(graph, labels)}


def verify(root, output, *, bootstrap=0, threads=1):
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('threads must be a positive integer')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    source_bytes = {name: (root / name).read_bytes() for name in SOURCE_FILES}
    started = time.perf_counter()
    historical = {name: read_checked(root, directory, 'labels.json.gz')
                  for name, directory in [('joint', JOINT), ('region', REGION)]}
    prior = {name: read_checked(root, directory, 'provenance.json')
             for name, directory in [('joint', JOINT), ('region', REGION)]}
    cfg = json.loads((root / 'configs/research_validation.json').read_text('utf-8'))
    with threadpool_limits(limits=threads):
        development, _, monthly, ids, scaler, manifest = load_development(root, cfg)
        x = annual_profile(monthly)
        regions = development.drop_duplicates('entity_id').set_index('entity_id').loc[ids, 'region_code'].to_numpy()
        for name in historical:
            if historical[name]['ids'] != ids or prior[name]['ids'] != ids:
                raise ValueError('Historical ordered IDs differ')
            if prior[name]['panel_sha256'] != manifest['panel_sha256']:
                raise ValueError('Historical panel fingerprint differs')
            if any(prior[name]['config'][key] != cfg[key] for key in ['features', 'graph', 'seed']):
                raise ValueError('Historical preprocessing or seed differs')
        road_path = root / 'reports/graphs/road_2024_2016.npz'
        if sha256(road_path) != prior['region']['transport_graph_sha256']:
            raise ValueError('Historical road graph fingerprint differs')
        road, regional = load_npz(road_path).tocsr(), same_region_graph(regions)
        attribute, attribute_info = knn_graph(x, cfg['graph']['k'], cfg['graph']['symmetry'])
        old = historical['joint']['labels']
        cases = [(f'joint_road_k{k}', road, np.array(old[f'joint_alpha0_k{k}']),
                  np.array(old[f'joint_road_k{k}'])) for k in (2, 3, 4)]
        cases += [('joint_same_region_k4', regional, np.array(old['joint_alpha0_k4']),
                   np.array(historical['region']['labels']['joint_same_region_k4']))]
        output.mkdir(parents=True, exist_ok=False)
        rows, partitions = [], {}
        for name, graph, initial, archived in cases:
            old_score = joint_objective(x, graph, archived, .25)
            old_stationarity = exact_stationarity(x, graph, archived, ids)
            old_sw = float(silhouette_score(x, archived))
            partitions[name + '__archived'] = archived
            for start_name, labels0 in [('initial', initial), ('endpoint', archived)]:
                labels, info = fit_graph_regularized_kmeans(x, graph, labels0, alpha=.25,
                                                            seed=1729, max_sweeps=50)
                audit = exact_stationarity(x, graph, labels, ids)
                if not info['converged'] or not audit['coordinate_stationary']:
                    raise ArithmeticError('Corrected fit did not reach exact coordinate stationarity')
                if any(a['objective'] + 1e-12 < b['objective'] for a, b in zip(info['trace'], info['trace'][1:])):
                    raise ArithmeticError('True objective increased')
                key = name + '__corrected_from_' + start_name
                partitions[key] = labels
                row = {'id': key, 'candidate': name, 'start': start_name,
                       'archived_objective': old_score, 'archived_stationarity': old_stationarity,
                       'archived_SW': old_sw, 'corrected_SW': float(silhouette_score(x, labels)),
                       'ARI_to_archived': float(adjusted_rand_score(archived, labels)),
                       'changed_after_label_matching': matched_changes(archived, labels),
                       'corrected_stationarity': audit, 'fit': info,
                       'on_real_road': joint_objective(x, road, labels, .25),
                       'attribute_graph_metrics': graph_metrics(attribute, labels),
                       'road_graph_metrics': graph_metrics(road, labels)}
                rows.append(row)
                print(json.dumps({'candidate': key, 'objective': info['objective'],
                                  'changed': row['changed_after_label_matching'], 'SW': row['corrected_SW']}), flush=True)
        for k in (2, 3, 4):
            partitions[f'kmeans_k{k}'] = np.array(old[f'joint_alpha0_k{k}'])
        external = None
        if bootstrap:
            bases = {k: f'kmeans_k{k}' for k in (2, 3, 4)}
            external = external_comparison(root, ids, partitions, bases, bootstrap)
            # Same paired draws, now with regional K4 as the comparison baseline.
            regional_names = {name: labels for name, labels in partitions.items() if '_k4' in name}
            external['road_vs_region'] = external_comparison(
                root, ids, regional_names, {4: 'joint_same_region_k4__corrected_from_initial'}, bootstrap)['results']
            write_json(output / 'external_comparison.json', external)
    write_json(output / 'comparisons.json', rows)
    write_json(output / 'labels.json', {'ids': ids, 'labels': {name: z.tolist() for name, z in partitions.items()}})
    write_json(output / 'regions.json', {'ids': ids, 'region_codes': regions.tolist()})
    np.save(output / 'features.npy', x, allow_pickle=False)
    source_objects = {}
    for name, content in source_bytes.items():
        if (root / name).read_bytes() != content:
            raise RuntimeError('Scientific source changed during verification: ' + name)
        digest = hashlib.sha256(content).hexdigest()
        source_objects[name] = digest
        target = output / 'source_objects' / (digest + '.py')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
    write_json(output / 'provenance.json', {
        'schema_version': 1, 'date_utc': datetime.now(timezone.utc).isoformat(),
        'algorithm_revision': ALGORITHM_REVISION, 'seed': 1729, 'alpha': .25, 'max_sweeps': 50,
        'threads': threads, 'bootstrap': bootstrap, 'n': len(ids),
        'features_shape': list(x.shape), 'features_array_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'panel_sha256': manifest['panel_sha256'], 'road_graph_sha256': sha256(road_path),
        'historical_manifests': {directory + '/manifest.json': sha256(root / directory / 'manifest.json')
                                 for directory in (JOINT, REGION)},
        'source_objects': source_objects, 'scaler': scaler,
        'feature_config': cfg['features'], 'attribute_graph': attribute_info,
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas']},
        'limitations': LIMITATIONS, 'total_seconds': time.perf_counter() - started})
    write_json(output / 'manifest.json', {'schema_version': 1,
        'files_sha256': {path.relative_to(output).as_posix(): sha256(path)
                         for path in sorted(output.rglob('*')) if path.is_file()}})
    return rows, external


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New result directory')
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root containing frozen inputs')
    parser.add_argument('--bootstrap', type=int, default=0, help='Optional paired regional draws, e.g. 500')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    verify(args.root, args.output, bootstrap=args.bootstrap, threads=args.threads)
