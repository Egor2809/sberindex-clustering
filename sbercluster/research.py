"""Bounded, staged comparisons. Development selection never uses 2024 scores."""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import json
import subprocess
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from .features import make_slices
from .input_contracts import validate_prepared_panel
from .graph import knn_graph
from .io import sha256, write_json, code_revision
from .metrics import all_metrics
from .models import require_execution, _igraph


@contextmanager
def tracked_run(out, **identity):
    """Keep interrupted/failed runs distinguishable from active workers."""
    started = perf_counter()
    write_json(out / 'status.json', {'status': 'running', **identity})
    try:
        yield
    except BaseException as exc:
        write_json(out / 'status.json', {
            'status': 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
            **identity, 'elapsed_seconds': perf_counter() - started,
            'error_type': type(exc).__name__, 'error': str(exc),
        })
        raise


def local_scale_graph(x, k=15, scale_neighbor=7):
    """Zelnik-Manor/Perona affinity on a sparse union-kNN support.

    This borrows their local bandwidth, not their full cluster-count algorithm.
    Distinct samples may coincide; zero local scale is rejected explicitly.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= scale_neighbor <= k < len(x):
        raise ValueError('Invalid local-scale graph input')
    distances = cdist(x, x)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind='stable')[:, :k]
    selected = np.take_along_axis(distances, order, axis=1)
    scales = selected[:, scale_neighbor - 1]
    if np.any(scales <= 0):
        raise ValueError('Local bandwidth is zero; duplicate profiles require explicit handling')
    weights = np.exp(-(selected ** 2) / (scales[:, None] * scales[order]))
    directed = csr_matrix((weights.ravel(), (np.repeat(np.arange(len(x)), k), order.ravel())),
                          shape=(len(x), len(x)))
    graph = directed.maximum(directed.T)
    graph.setdiag(0); graph.eliminate_zeros()
    components, _ = connected_components(graph, directed=False)
    return graph, {'kernel': 'exp(-d2/(sigma_i*sigma_j))', 'scale_neighbor': scale_neighbor,
                   'k': k, 'components': int(components), 'edges': graph.nnz // 2,
                   'isolates': int((graph.getnnz(axis=1) == 0).sum())}


def annual_profile(monthly, reducer='median'):
    monthly = np.asarray(monthly, dtype=float)
    if monthly.ndim != 3 or not np.isfinite(monthly).all():
        raise ValueError('Expected finite month-by-territory-by-feature tensor')
    if reducer != 'median':
        raise ValueError('Only the prespecified monthly median is supported')
    return np.median(monthly, axis=0)


def block_month_indices(seed, months=12, block=3):
    if months % block:
        raise ValueError('Month count must be divisible by block length')
    starts = np.random.default_rng(seed).integers(0, months, size=months // block)
    return np.concatenate([(start + np.arange(block)) % months for start in starts])


def fit_candidate(x, graph, spec, seed, cfg):
    require_execution(cfg, True)
    method = spec['method']
    info = {}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            estimator = KMeans(n_clusters=spec['k'], n_init=20, random_state=seed)
            labels = estimator.fit_predict(x)
            info.update(inertia=float(estimator.inertia_), n_iter=int(estimator.n_iter_))
        elif method == 'ward':
            from sklearn.cluster import AgglomerativeClustering
            labels = AgglomerativeClustering(n_clusters=spec['k'], linkage='ward').fit_predict(x)
        elif method == 'gmm_full':
            from sklearn.mixture import GaussianMixture
            estimator = GaussianMixture(n_components=spec['k'], covariance_type='full',
                n_init=3, reg_covar=1e-6, max_iter=300, random_state=seed)
            labels = estimator.fit_predict(x)
            posterior = estimator.predict_proba(x)
            info.update(BIC=float(estimator.bic(x)), converged=bool(estimator.converged_),
                        n_iter=int(estimator.n_iter_),
                        mean_max_posterior=float(posterior.max(axis=1).mean()))
        elif method in ('spectral_global', 'spectral_local'):
            from sklearn.cluster import SpectralClustering
            labels = SpectralClustering(n_clusters=spec['k'], affinity='precomputed',
                eigen_solver='arpack', assign_labels='cluster_qr', random_state=seed,
                n_jobs=1).fit_predict(graph)
        elif method in ('leiden_global', 'leiden_local'):
            import leidenalg as la
            partition = la.find_partition(_igraph(graph, list(range(len(x)))),
                la.RBConfigurationVertexPartition, weights='weight',
                resolution_parameter=spec['resolution'], seed=int(seed), n_iterations=4)
            labels = np.asarray(partition.membership)
        else:
            raise ValueError(f'Unknown candidate {method}')
    info['warnings'] = [str(w.message) for w in captured]
    _, labels = np.unique(labels, return_inverse=True)
    if not 2 <= len(np.unique(labels)) < len(x):
        raise ValueError('Degenerate partition')
    return labels.astype(np.int32), info


def load_development(root, cfg):
    path = root / 'data/processed/panel.csv'
    manifest = json.loads((path.parent / 'manifest.json').read_text('utf-8'))
    if sha256(path) != manifest['panel_sha256']:
        raise ValueError('Prepared panel hash changed')
    panel = pd.read_csv(path, dtype={'entity_id': str, 'period': str, 'territory_id': str})
    validate_prepared_panel(panel, manifest, cfg)
    # The cohort was selected earlier using full-period availability; this is retrospective.
    development = panel[panel.period <= '2023-12-01'].copy()
    if cfg['features']['calibration_end'] != '2023-12-01':
        raise ValueError('Development calibration is frozen at December 2023')
    slices, scaler = make_slices(development, cfg)
    expected = pd.date_range('2023-01-01','2023-12-01',freq='MS').strftime('%Y-%m-%d').tolist()
    if [s[0] for s in slices] != expected:
        raise ValueError('Expected all 12 development months')
    ids = slices[0][1]
    if any(s[1] != ids for s in slices):
        raise ValueError('Development IDs are not aligned')
    monthly = np.stack([s[2] for s in slices])
    return development, slices, monthly, ids, scaler, manifest


def source_snapshot(root):
    files = sorted((root / 'sbercluster').glob('*.py')) + [
        root / 'scripts/run_bounded.py', root / 'scripts/run_research.py']
    return {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in files}


def create_run(root, cfg, scaler, manifest):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = root / 'runs' / ('research-' + run_id)
    out.mkdir(parents=True, exist_ok=False)
    revision = code_revision(root)
    snapshot = source_snapshot(root)
    for relative in snapshot:
        target = out / 'implementation_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    write_json(out / 'provenance.json', {'config': cfg, 'scaler': scaler,
        'panel_sha256': manifest['panel_sha256'], **revision, 'source_hashes': snapshot,
        'started_at': run_id, 'selection_scope': '2023 only',
        'cohort_scope': 'balanced 2023-2024 cohort selected retrospectively'})
    write_json(out / 'status.json', {'status': 'running', 'phase': cfg['research']['phase']})
    print(json.dumps({'research_run': str(out), 'phase': cfg['research']['phase']}), flush=True)
    return out


def score(x, labels, reference_graph):
    result = all_metrics(x, labels, reference_graph)
    sizes = np.bincount(labels)
    silhouette = silhouette_samples(x, labels)
    result.update(max_cluster_size=int(sizes.max()),
        median_cluster_size=float(np.median(sizes)),
        negative_silhouette_fraction=float((silhouette < 0).mean()),
        below_20_territories_clusters=int((sizes < 20).sum()))
    return result


def screen(cfg, root, slices, monthly, ids, out):
    representations = {p: x for p, _, x in slices}
    representations['annual_2023'] = annual_profile(monthly)
    rows, graphs = [], []
    labels_dir = out / 'labels'; labels_dir.mkdir()
    for representation in cfg['research']['representations']:
        x = representations[representation]
        common, common_info = knn_graph(x, k=15)
        local, local_info = local_scale_graph(x, k=15, scale_neighbor=7)
        graphs.append({'representation': representation, 'reference': common_info, 'local': local_info})
        for spec in cfg['research']['candidates']:
            started = perf_counter()
            graph = local if spec['method'].endswith('_local') else common
            row = {'representation': representation, 'candidate': spec['id'],
                   'spec': spec, 'seed': cfg['seed']}
            try:
                labels, info = fit_candidate(x, graph, spec, cfg['seed'], cfg)
                row.update(score(x, labels, common), fit_info=info, status='completed')
                np.save(labels_dir / f'{representation}__{spec["id"]}.npy', labels, allow_pickle=False)
            except (ValueError, np.linalg.LinAlgError) as exc:
                row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            row['elapsed_seconds'] = perf_counter() - started
            rows.append(row)
            write_json(out / 'metrics.json', rows)
            print(json.dumps({key: row.get(key) for key in ('representation','candidate','status','SW','k','elapsed_seconds')}), flush=True)
    write_json(out / 'graphs.json', graphs)
    write_json(out / 'ids.json', ids)


def stability(cfg, root, monthly, ids, out):
    reference = annual_profile(monthly)
    common, _ = knn_graph(reference, k=15)
    local, _ = local_scale_graph(reference, k=15, scale_neighbor=7)
    rows, reference_metrics, consensus = [], [], {}
    labels_dir = out / 'labels'; labels_dir.mkdir()
    for spec in cfg['research']['candidates']:
        graph = local if spec['method'].endswith('_local') else common
        baseline, info = fit_candidate(reference, graph, spec, cfg['seed'], cfg)
        reference_metrics.append({'candidate': spec['id'], 'spec': spec,
                                  **score(reference, baseline, common), 'fit_info': info})
        np.save(labels_dir / f'{spec["id"]}__reference.npy', baseline, allow_pickle=False)
        variant_labels = []
        variants = []
        if spec['method'] in ('kmeans','gmm_full','leiden_global','leiden_local','spectral_global','spectral_local'):
            variants += [('seed', seed, reference, 15) for seed in cfg['research']['seeds'] if seed != cfg['seed']]
        variants += [('month_block', seed, annual_profile(monthly[block_month_indices(seed)]), 15)
                     for seed in cfg['research']['block_seeds']]
        for column in range(reference.shape[1]):
            variants.append(('drop_feature', column, np.delete(reference, column, axis=1), 15))
        if spec['method'] in ('spectral_global','spectral_local','leiden_global','leiden_local'):
            variants += [('graph_k', k, reference, k) for k in (10, 30)]
        for kind, parameter, x, neighbors in variants:
            started = perf_counter()
            seed = parameter if kind == 'seed' else cfg['seed']
            a, _ = local_scale_graph(x, neighbors, 7) if spec['method'].endswith('_local') else knn_graph(x, neighbors)
            labels, fit_info = fit_candidate(x, a, spec, seed, cfg)
            ari = float(adjusted_rand_score(baseline, labels))
            row = {'candidate': spec['id'], 'kind': kind, 'parameter': parameter,
                   'ARI': ari, 'k': len(np.unique(labels)), 'fit_info': fit_info,
                   'elapsed_seconds': perf_counter() - started}
            rows.append(row)
            if kind == 'month_block':
                variant_labels.append(labels)
            np.save(labels_dir / f'{spec["id"]}__{kind}_{parameter}.npy', labels, allow_pickle=False)
            write_json(out / 'stability.json', rows)
            print(json.dumps({k: row[k] for k in ('candidate','kind','parameter','ARI','elapsed_seconds')}), flush=True)
        # Per-object agreement after optimal label matching; bootstrap frequency, not a probability of truth.
        from scipy.optimize import linear_sum_assignment
        agreement = np.zeros(len(ids))
        for labels in variant_labels:
            table = np.zeros((baseline.max()+1, labels.max()+1), dtype=int)
            np.add.at(table, (baseline, labels), 1)
            row_indices, col_indices = linear_sum_assignment(-table)
            mapping = {int(c): int(r) for r,c in zip(row_indices,col_indices)}
            agreement += np.array([mapping.get(int(z), -1) for z in labels]) == baseline
        consensus[spec['id']] = (agreement / len(variant_labels)).tolist()
    write_json(out / 'reference_metrics.json', reference_metrics)
    write_json(out / 'membership_stability.json', {'ids': ids, 'agreement': consensus,
        'interpretation': 'fraction of month-block perturbations agreeing after Hungarian alignment; not correctness probability'})


def run(cfg, root):
    require_execution(cfg, True)
    root = Path(root)
    phase = cfg['research']['phase']
    if phase not in ('screen', 'stability', 'validation'):
        raise ValueError('Unknown research phase')
    started = perf_counter()
    _, slices, monthly, ids, scaler, manifest = load_development(root, cfg)
    out = create_run(root, cfg, scaler, manifest)
    with tracked_run(out, phase=phase):
        _run_created(cfg, root, slices, monthly, ids, scaler, out, started)
    return out


def _run_created(cfg, root, slices, monthly, ids, scaler, out, started):
    if cfg['research']['phase'] == 'screen':
        screen(cfg, root, slices, monthly, ids, out)
    elif cfg['research']['phase'] == 'stability':
        stability(cfg, root, monthly, ids, out)
    elif cfg['research']['phase'] == 'validation':
        from .research_validation import validate
        validate(cfg, root, monthly, ids, scaler, out)
    else:
        raise ValueError('Unknown research phase')
    write_json(out / 'status.json', {'status': 'completed', 'phase': cfg['research']['phase'],
                                   'elapsed_seconds': perf_counter() - started})
    print(json.dumps({'research_run': str(out), 'status': 'completed'}), flush=True)
