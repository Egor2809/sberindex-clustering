"""Edge-rule comparison, unified ICVI table and cluster lifecycle events on the 2 016-territory panel."""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from .contest_graphs import read_road_graph
from .dynamics import cluster_events
from .edge_rules import (cosine_similarity, distance_knn, dtw_distance_matrix, edge_jaccard, lagged_correlation,
                         residual_series, similarity_knn, structure)
from .features import make_slices
from .graph import knn_graph
from .io import CATEGORIES, TOTAL, sha256, write_json
from .metrics import all_metrics, network_indices, network_quality_candidates

ICVI = ['SW', 'CH', 'S_Dbw', 'S_Dbw_paper', 'AVI', 'AVU', 'MQ']


def load_panel(cfg, root):
    path = root / 'data/processed/panel.csv'
    manifest = json.loads((path.parent / 'manifest.json').read_text('utf-8'))
    if sha256(path) != manifest['panel_sha256']:
        raise ValueError('Prepared panel fingerprint differs')
    panel = pd.read_csv(path, dtype={'entity_id': str, 'period': str, 'oktmo': str})
    slices, _ = make_slices(panel, cfg)
    ids = slices[0][1]
    if len(slices) != 24 or any(s[1] != ids for s in slices):
        raise ValueError('Complete aligned 2023-2024 monthly cohort required')
    monthly = np.stack([s[2] for s in slices])
    wide = panel.set_index(['period', 'entity_id']).sort_index()
    raw = np.stack([wide.loc[s[0]].reindex(ids)[CATEGORIES + [TOTAL]].to_numpy(dtype=float) for s in slices])
    region = panel.drop_duplicates('entity_id').set_index('entity_id').reindex(ids)['region_code'].to_numpy()
    return ids, [s[0] for s in slices], monthly, raw, region


def reference_partition(root, ids, annual):
    vintage = root / 'reports/experiments/2026-09-23-v2/validation'
    frozen = json.loads((vintage / 'frozen_prototypes.json').read_text('utf-8'))
    if frozen['ids'] != ids:
        raise ValueError('Frozen model and panel identifiers differ')
    centers = np.asarray(frozen['centers'])
    labels = pd.read_csv(vintage / 'reference_assignments.csv').set_index('entity_id').reindex(ids).cluster.to_numpy(dtype=int)
    if not np.array_equal(cdist(annual, centers).argmin(axis=1), labels):
        raise ValueError('Frozen 2023 assignment no longer reproduces')
    return centers, labels


def leiden_partition(a, seed, iterations):
    import igraph as ig
    import leidenalg
    from scipy.sparse import triu
    upper = triu(a, k=1).tocoo()
    graph = ig.Graph(n=a.shape[0], edges=list(zip(upper.row.tolist(), upper.col.tolist())))
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, weights=upper.data.tolist(),
                                         n_iterations=iterations, seed=seed)
    return np.asarray(partition.membership)


def compact(row):
    return {key: (None if value is None else value if isinstance(value, (int, np.integer)) else round(float(value), 6))
            for key, value in row.items()}


def icvi(x, labels, graph):
    m = all_metrics(x, labels, graph)
    return {key: m.get(key) for key in ICVI} | {'k': int(m['k']), 'min_cluster_size': int(m['min_cluster_size'])}


def build_rules(cfg, root, ids, raw, annual, profile_months):
    k = cfg['graph']['k']
    window = cfg['edge_rules']['series_months']
    series = residual_series(raw[:window])
    base = raw[:profile_months]
    shares = np.median(base[:, :, :len(CATEGORIES)] / base[:, :, [len(CATEGORIES)]], axis=0)
    timings, graphs, extra = {}, {}, {}
    flat = series.transpose(1, 0, 2).reshape(series.shape[1], -1)
    spaces = {'euclid_profile': annual, 'cosine_shares': shares / np.linalg.norm(shares, axis=1, keepdims=True),
              'pearson_series': flat, 'lagged_correlation': flat, 'dtw_series': flat}
    started = time.perf_counter()
    graphs['euclid_profile'], info = knn_graph(annual, k)
    extra['euclid_profile'] = {'sigma': info['sigma']}
    timings['euclid_profile'] = time.perf_counter() - started
    started = time.perf_counter()
    graphs['cosine_shares'] = similarity_knn(cosine_similarity(shares), k)
    timings['cosine_shares'] = time.perf_counter() - started
    started = time.perf_counter()
    pearson, _ = lagged_correlation(series, 0)
    graphs['pearson_series'] = similarity_knn(pearson, k)
    timings['pearson_series'] = time.perf_counter() - started
    started = time.perf_counter()
    lagged, lag = lagged_correlation(series, cfg['edge_rules']['max_lag'])
    graphs['lagged_correlation'] = similarity_knn(lagged, k)
    rows, cols = graphs['lagged_correlation'].nonzero()
    upper = rows < cols
    edge_lags = np.abs(lag[rows[upper], cols[upper]])
    extra['lagged_correlation'] = {'edge_lag_counts': {str(v): int((edge_lags == v).sum()) for v in range(cfg['edge_rules']['max_lag'] + 1)}}
    timings['lagged_correlation'] = time.perf_counter() - started
    started = time.perf_counter()
    graphs['dtw_series'], sigma = distance_knn(dtw_distance_matrix(series, cfg['edge_rules']['dtw_window']), k)
    extra['dtw_series'] = {'sigma': sigma}
    timings['dtw_series'] = time.perf_counter() - started
    started = time.perf_counter()
    source = root / cfg['edge_rules']['source_dir']
    graphs['road_distance'], _, road = read_road_graph(source / 'hackathonlicence/connection.parquet', ids, k)
    extra['road_distance'] = {'bandwidth_km': road['bandwidth_km']}
    timings['road_distance'] = time.perf_counter() - started
    return graphs, spaces, extra, timings


def compare_rules(cfg, graphs, spaces, annual, reference, region, seed):
    rows, partitions = [], {}
    base = graphs['euclid_profile']
    k = cfg['edge_rules']['n_clusters']
    for name, a in graphs.items():
        spectral = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=seed,
                                      assign_labels='cluster_qr').fit_predict(a.toarray())
        leiden = leiden_partition(a, seed, cfg['edge_rules']['leiden_iterations'])
        partitions[name] = {'spectral': spectral, 'leiden': leiden}
        reference_on_graph = {**network_indices(a, reference), 'MQ': network_quality_candidates(a, reference)['NewmanQ']}
        rows.append({'rule': name, **structure(a, {'region': region, 'profile': reference}),
                     'edge_jaccard_euclid': edge_jaccard(a, base),
                     'spectral': compact(icvi(annual, spectral, a)) | {'SW_own_space': float(silhouette_score(spaces[name], spectral)) if name in spaces else None,
                                                                      'ARI_reference': adjusted_rand_score(reference, spectral),
                                                                      'ARI_region': adjusted_rand_score(region, spectral)},
                     'leiden': {'communities': int(leiden.max() + 1),
                                'MQ': network_quality_candidates(a, leiden)['NewmanQ'],
                                'ARI_reference': adjusted_rand_score(reference, leiden),
                                'ARI_region': adjusted_rand_score(region, leiden)},
                     'reference_on_graph': compact({key: reference_on_graph[key] for key in ('AVI', 'AVU', 'MQ')})})
    overlap = {a: {b: edge_jaccard(graphs[a], graphs[b]) for b in graphs} for a in graphs}
    return rows, partitions, overlap


def k_sensitivity(cfg, annual, raw, reference):
    series = residual_series(raw[:cfg['edge_rules']['series_months']])
    pearson, _ = lagged_correlation(series, 0)
    out = []
    for k in cfg['edge_rules']['k_grid']:
        for name, a in (('euclid_profile', knn_graph(annual, k)[0]), ('pearson_series', similarity_knn(pearson, k))):
            out.append({'rule': name, 'k': k, **structure(a, {'profile': reference}),
                        **compact({key: v for key, v in network_indices(a, reference).items() if key in ('AVI', 'AVU')}),
                        'MQ': network_quality_candidates(a, reference)['NewmanQ']})
    return out


def published_partitions(root, ids):
    out = {}
    screen = pd.read_csv(root / 'reports/experiments/2026-09-23-v2/screen_partitions.csv.gz')
    screen = screen[screen.representation == 'annual_2023']
    for candidate, frame in screen.groupby('candidate'):
        out[candidate] = frame.set_index('entity_id').reindex(ids).cluster.to_numpy()
    for run in ['reports/review-2026-09-24/review-20260924-small-k', 'reports/review-2026-09-24/review-20260924-joint',
                'reports/round2-2026-09-24/round2-20260924-region_control', 'reports/dmon-plateau-2026-09-25/dmon-plateau-20260925']:
        with gzip.open(root / run / 'labels.json.gz', 'rt', encoding='utf-8') as stream:
            stored = json.load(stream)
        if stored['ids'] != ids:
            raise ValueError('Stored labels are not aligned with the panel: ' + run)
        for name, labels in stored['labels'].items():
            if 'shuffled' not in name and 'alpha0' not in name and name != 'frozen_kmeans_k4':
                out.setdefault(name, np.asarray(labels))
    return out


def icvi_agreement(rows, min_size):
    from scipy.stats import spearmanr
    orient = {'SW': 1, 'CH': 1, 'S_Dbw': -1, 'S_Dbw_paper': -1, 'AVI': 1, 'AVU': -1, 'MQ': 1}
    out = {}
    for scope, subset in (('all', rows), ('k4', [r for r in rows if r['k'] == 4])):
        matrix = {}
        for a in orient:
            matrix[a] = {}
            for b in orient:
                pairs = [(r[a] * orient[a], r[b] * orient[b]) for r in subset if r[a] is not None and r[b] is not None]
                matrix[a][b] = round(float(spearmanr(*zip(*pairs))[0]), 3) if len(pairs) > 2 else None
        out[scope] = {'n': len(subset), 'spearman_oriented': matrix}
    candidates = [r for r in rows if r['k'] == 4 and r['min_cluster_size'] >= min_size and r['SW'] is not None and r['MQ'] is not None]
    front = [r['partition'] for r in candidates
             if not any(o['SW'] >= r['SW'] and o['MQ'] >= r['MQ'] and (o['SW'] > r['SW'] or o['MQ'] > r['MQ']) for o in candidates)]
    out['pareto_k4_sw_mq'] = {'min_cluster_size': min_size, 'candidates': len(candidates), 'front': front}
    return out


def icvi_table(root, ids, annual, graph, reference):
    rows = []
    for name, z in sorted(published_partitions(root, ids).items()):
        z = pd.factorize(z)[0]
        if len(np.unique(z)) < 2:
            continue
        rows.append({'partition': name, **compact(icvi(annual, z, graph)), 'ARI_reference': round(adjusted_rand_score(reference, z), 6)})
    return rows


def seasonal_profile(medians):
    """Classical decomposition: deviation from the centred 2x12 moving average, averaged by calendar month."""
    m = np.asarray(medians, dtype=float)
    if len(m) < 24:
        raise ValueError('At least 24 months are needed for a centred 12-month trend')
    weights = np.r_[0.5, np.ones(11), 0.5] / 12
    deviations = [[] for _ in range(12)]
    for t in range(6, len(m) - 6):
        deviations[t % 12].append(m[t] - weights @ m[t - 6:t + 7])
    season = np.array([np.mean(d, axis=0) for d in deviations])
    return season - season.mean(axis=0)


def lifecycle(cfg, root, ids, periods, monthly, centers):
    threshold, change = cfg['events']['jaccard_threshold'], cfg['events']['size_change']
    fixed = [cdist(x, centers).argmin(axis=1) for x in monthly]
    season = seasonal_profile(np.median(monthly, axis=1))
    deseasonalized = [cdist(x - season[index % 12], centers).argmin(axis=1) for index, x in enumerate(monthly)]
    temporal_path = root / cfg['events']['temporal_labels']
    with gzip.open(temporal_path, 'rt', encoding='utf-8') as stream:
        temporal = json.load(stream)
    if temporal['ids'] != ids or temporal['periods'] != periods:
        raise ValueError('Temporal labels do not match the panel cohort')
    out = {}
    for name, series in (('fixed_profiles', fixed), ('fixed_profiles_deseasonalized', deseasonalized),
                         ('temporal_leiden', [np.asarray(z) for z in temporal['labels']])):
        steps = []
        for index in range(1, len(series)):
            events = cluster_events(ids, series[index - 1], ids, series[index], threshold, change)
            steps.append({'from': periods[index - 1], 'to': periods[index], 'groups_before': int(len(np.unique(series[index - 1]))),
                          'groups_after': int(len(np.unique(series[index]))),
                          'counts': {kind: sum(e['event'] == kind for e in events) for kind in
                                     ('continue', 'grow', 'shrink', 'split', 'merge', 'birth', 'death')},
                          'events': events})
        sizes = [{str(label): int(count) for label, count in zip(*np.unique(z, return_counts=True))} for z in series]
        out[name] = {'steps': steps, 'sizes': sizes}
    return out


def run(cfg, root):
    root = Path(root)
    out = root / cfg['edge_rules']['output']
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ids, periods, monthly, raw, region = load_panel(cfg, root)
    profile_months = sum(period <= cfg['features']['calibration_end'] for period in periods)
    annual = np.median(monthly[:profile_months], axis=0)
    centers, reference = reference_partition(root, ids, annual)
    print(json.dumps({'phase': 'graphs', 'n': len(ids)}), flush=True)
    graphs, spaces, extra, timings = build_rules(cfg, root, ids, raw, annual, profile_months)
    print(json.dumps({'phase': 'compare', 'seconds': timings}), flush=True)
    rows, partitions, overlap = compare_rules(cfg, graphs, spaces, annual, reference, region, cfg['seed'])
    for row in rows:
        row.update(extra.get(row['rule'], {}))
    write_json(out / 'edge_rules.json', {'n': len(ids), 'k': cfg['graph']['k'], 'series_months': cfg['edge_rules']['series_months'],
                                         'rules': rows, 'edge_jaccard': overlap, 'timings_seconds': timings})
    write_json(out / 'k_sensitivity.json', k_sensitivity(cfg, annual, raw, reference))
    pd.DataFrame({'entity_id': ids, **{f'{rule}_{method}': labels for rule, value in partitions.items() for method, labels in value.items()}}).to_csv(
        out / 'edge_rule_partitions.csv.gz', index=False)
    print(json.dumps({'phase': 'icvi'}), flush=True)
    binary_reference = graphs['euclid_profile']
    table = icvi_table(root, ids, annual, binary_reference, reference)
    write_json(out / 'icvi_table.json', table)
    pd.DataFrame(table).to_csv(out / 'icvi_table.csv', index=False)
    write_json(out / 'icvi_agreement.json', icvi_agreement(table, cfg['edge_rules']['min_cluster_size']))
    print(json.dumps({'phase': 'events'}), flush=True)
    write_json(out / 'cluster_events.json', lifecycle(cfg, root, ids, periods, monthly, centers))
    write_json(out / 'provenance.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                                         'panel_sha256': sha256(root / 'data/processed/panel.csv'),
                                         'implementation': {p.name: sha256(p) for p in (root / 'sbercluster').glob('*.py')},
                                         'seconds': time.perf_counter() - started})
    return out
