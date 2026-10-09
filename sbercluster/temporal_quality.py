"""Monthly ICVI, next-month network checks, omega trade-off and fixed-centre drift for the 2023-2024 partitions."""
from __future__ import annotations
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from .contest_graphs import mix_layers
from .edge_study import load_panel, reference_partition, seasonal_profile
from .io import CATEGORIES, sha256, write_json
from .metrics import all_metrics, network_indices, network_quality_candidates
from .round2 import matched_change

SIX = ['SW', 'CH', 'S_Dbw', 'AVI', 'AVU', 'MQ']
SERIES = ['ARI_prev', 'churn_prev', 'size_shift_prev']
NEXT = ['MQ_next', 'AVI_next']
MIX = ['MQ_mix', 'AVI_mix', 'MQ_mix_next']
COLUMNS = SIX + NEXT + MIX + SERIES


def six_icvi(x, labels, graph):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {key: None for key in SIX} | {'k': 1, 'min_cluster_size': len(labels)}
    m = all_metrics(x, labels, graph)
    return {key: m[key] for key in SIX} | {'k': int(m['k']), 'min_cluster_size': int(m['min_cluster_size'])}


def next_month_network(labels, graph):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {'MQ_next': None, 'AVI_next': None}
    return {'MQ_next': network_quality_candidates(graph, labels)['NewmanQ'], 'AVI_next': network_indices(graph, labels)['AVI']}


def matched_size_shift(a, b):
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    if ia.shape != ib.shape:
        raise ValueError('Aligned label vectors required')
    table = np.zeros((ia.max() + 1, ib.max() + 1), int)
    np.add.at(table, (ia, ib), 1)
    r, c = linear_sum_assignment(-table)
    na, nb = table.sum(axis=1), table.sum(axis=0)
    moved = np.abs(na[r] - nb[c]).sum() + np.delete(na, r).sum() + np.delete(nb, c).sum()
    return float(moved / (2 * len(ia)))


def mixed_network(labels, graph, next_graph):
    labels = np.asarray(labels)
    if graph is None or len(np.unique(labels)) < 2:
        return dict.fromkeys(MIX)
    return {'MQ_mix': network_quality_candidates(graph, labels)['NewmanQ'], 'AVI_mix': network_indices(graph, labels)['AVI'],
            'MQ_mix_next': None if next_graph is None else network_quality_candidates(next_graph, labels)['NewmanQ']}


def series_rows(method, omega, seed, labels, monthly, graphs, periods, mixed=None, fit_mq=None):
    rows = []
    for t, z in enumerate(labels):
        row = {'method': method, 'omega': omega, 'seed': seed, 'period': periods[t], **six_icvi(monthly[t], z, graphs[t])}
        row |= next_month_network(z, graphs[t + 1]) if t + 1 < len(labels) else dict.fromkeys(NEXT)
        if t:
            row |= {'ARI_prev': float(adjusted_rand_score(labels[t - 1], z)), 'churn_prev': float(matched_change(labels[t - 1], z).mean()),
                    'size_shift_prev': matched_size_shift(labels[t - 1], z)}
        else:
            row |= dict.fromkeys(SERIES)
        row |= mixed_network(z, None if mixed is None else mixed[t], None if mixed is None or t + 1 == len(labels) else mixed[t + 1])
        if fit_mq is not None and fit_mq[t] is not None and abs(row['MQ_mix'] - fit_mq[t]) > 1e-9:
            raise ValueError('Mixed graph does not reproduce the stored temporal Leiden modularity: ' + periods[t])
        rows.append(row)
    return rows


def mean_valid(values):
    kept = [float(v) for v in values if v is not None and np.isfinite(v)]
    return (float(np.mean(kept)) if kept else None), len(kept)


def summarise(rows):
    out = {'months': len(rows), 'k_min': min(r['k'] for r in rows), 'k_max': max(r['k'] for r in rows)}
    for key in COLUMNS:
        out[key], out[key + '_valid'] = mean_valid([r[key] for r in rows])
    out['MQ_next_over_own_next'], _ = mean_valid([a['MQ_next'] / b['MQ'] if a['MQ_next'] is not None and b['MQ'] else None for a, b in zip(rows[:-1], rows[1:])])
    return out


def seed_agreement(series):
    pairs = [[adjusted_rand_score(series[i][t], series[j][t]) for i in range(len(series)) for j in range(i)] for t in range(len(series[0]))]
    flat = [v for month in pairs for v in month]
    return {'seed_ARI_mean': float(np.mean(flat)), 'seed_ARI_min': float(np.min(flat))} if flat else {'seed_ARI_mean': None, 'seed_ARI_min': None}


def select_omega(rows, max_drop):
    base = next(r['MQ_mean'] for r in rows if r['omega'] == 0)
    threshold = (1 - max_drop) * base
    eligible = [r['omega'] for r in rows if r['MQ_mean'] >= threshold]
    return {'baseline_MQ': base, 'threshold_MQ': threshold, 'eligible': eligible, 'selected': max(eligible)}


def omega_tradeoff(summaries, labels, omegas, seeds, main_seed, max_drop):
    rows = []
    for omega in omegas:
        runs = [summaries[('temporal_leiden', omega, seed)] for seed in seeds]
        per_seed_mq = [r['MQ'] for r in runs]
        rows.append({'omega': omega, 'MQ_mean': float(np.mean(per_seed_mq)), 'MQ_seed_sd': float(np.std(per_seed_mq)),
                     'MQ_main_seed': summaries[('temporal_leiden', omega, main_seed)]['MQ'],
                     'SW_mean': float(np.mean([r['SW'] for r in runs])), 'MQ_next_mean': float(np.mean([r['MQ_next'] for r in runs])),
                     'MQ_next_over_own_next': float(np.mean([r['MQ_next_over_own_next'] for r in runs])),
                     'MQ_mix_mean': float(np.mean([r['MQ_mix'] for r in runs])), 'MQ_mix_next_mean': float(np.mean([r['MQ_mix_next'] for r in runs])),
                     'ARI_adjacent_mean': float(np.mean([r['ARI_prev'] for r in runs])), 'churn_mean': float(np.mean([r['churn_prev'] for r in runs])),
                     **seed_agreement([labels[(omega, seed)] for seed in seeds]),
                     'k_min': min(r['k_min'] for r in runs), 'k_max': max(r['k_max'] for r in runs)})
    choice = select_omega(rows, max_drop)
    for row in rows:
        row['MQ_drop_vs_omega0'] = 1 - row['MQ_mean'] / choice['baseline_MQ']
        row['eligible'] = row['omega'] in choice['eligible']
        row['selected'] = row['omega'] == choice['selected']
    return rows, choice


def fixed_assignments(monthly, centers):
    season = seasonal_profile(np.median(monthly, axis=1))
    raw = [cdist(x, centers).argmin(axis=1) for x in monthly]
    deseasonalized = [cdist(x - season[t % 12], centers).argmin(axis=1) for t, x in enumerate(monthly)]
    return raw, deseasonalized, season


def refit_kmeans(monthly, k, n_init, seed):
    return [KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(x) for x in monthly]


def type_counts(x, centers):
    return np.bincount(cdist(x, centers).argmin(axis=1), minlength=len(centers))


def drift_decomposition(monthly, raw, periods, centers, season, annual, cfg):
    d = cfg['temporal_quality']['drift']
    window = [t for t, p in enumerate(periods) if d['start'] <= p <= d['end']]
    first, last, b = window[0], window[-1], d['balanced_type']
    xd = monthly - season[np.arange(len(monthly)) % 12][:, None, :]
    shift = np.median(xd, axis=1) - np.median(annual, axis=0)
    scenarios = {'deseasonalized': xd, 'common_shift_removed_all': xd - shift[:, None, :]}
    for c, name in enumerate(CATEGORIES):
        y = xd.copy()
        y[:, :, c] -= shift[:, None, c]
        scenarios['common_shift_removed_' + name] = y
    counts = {name: [type_counts(y[t], centers).tolist() for t in range(len(periods))] for name, y in scenarios.items()}
    base_drop = counts['deseasonalized'][first][b] - counts['deseasonalized'][last][b]
    effects = []
    for name, series in counts.items():
        drop = series[first][b] - series[last][b]
        effects.append({'scenario': name, 'balanced_start': series[first][b], 'balanced_end': series[last][b], 'drop': drop,
                        'share_of_drop_removed': 1 - drop / base_drop if base_drop else None})
    shares = raw[:, :, :len(CATEGORIES)] / raw[:, :, [len(CATEGORIES)]]
    medians = np.median(shares, axis=1)
    ja, jb = periods.index('2024-01-01'), periods.index('2024-12-01')
    raw_medians = [{'category': name, 'median_share_start': medians[first, c], 'median_share_end': medians[last, c],
                    'change_2023': medians[last, c] - medians[first, c], 'median_share_2024_01': medians[ja, c],
                    'median_share_2024_12': medians[jb, c], 'change_2024': medians[jb, c] - medians[ja, c],
                    'season_start': season[first % 12, c], 'season_end': season[last % 12, c],
                    'common_shift_start': shift[first, c], 'common_shift_end': shift[last, c],
                    'center_other_minus_balanced': {str(k): centers[k, c] - centers[b, c] for k in range(len(centers)) if k != b}}
                   for c, name in enumerate(CATEGORIES)]
    za, zb = cdist(xd[first], centers).argmin(axis=1), cdist(xd[last], centers).argmin(axis=1)
    transition = np.zeros((len(centers), len(centers)), int)
    np.add.at(transition, (za, zb), 1)
    change = xd[last] - xd[first]
    movers = {}
    for k in range(len(centers)):
        mask = (za == b) & (zb == k)
        if k != b and mask.any():
            movers[str(k)] = {'n': int(mask.sum()), 'median_change': dict(zip(CATEGORIES, np.median(change[mask], axis=0).tolist()))}
    return {'window': [periods[first], periods[last]], 'balanced_type': b, 'type_names': d['type_names'],
            'counts_by_scenario': counts, 'effects': effects, 'raw_and_feature_medians': raw_medians,
            'transition_start_to_end': transition.tolist(), 'balanced_movers': movers,
            'median_change_all': dict(zip(CATEGORIES, np.median(change, axis=0).tolist()))}


def size_profile(labels, periods, start, end, same_labels=False):
    window = [t for t, p in enumerate(periods) if start <= p <= end]
    sizes = [{str(label): int(count) for label, count in zip(*np.unique(labels[t], return_counts=True))} for t in range(len(periods))]
    first, last = sizes[window[0]], sizes[window[-1]]
    same = {'window_size_shift_same_labels': sum(abs(first.get(k, 0) - last.get(k, 0)) for k in first.keys() | last.keys()) / (2 * len(labels[0]))} if same_labels else {}
    return same | {'sizes': sizes, 'groups_in_window': [len(sizes[t]) for t in window], 'window_size_shift': matched_size_shift(labels[window[0]], labels[window[-1]]),
            'window_churn': float(matched_change(labels[window[0]], labels[window[-1]]).mean()),
            'window_mean_adjacent_size_shift': float(np.mean([matched_size_shift(labels[t - 1], labels[t]) for t in window[1:]]))}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6) if np.isfinite(value) else None
    return value


def load_inputs(cfg, root):
    s = cfg['temporal_quality']
    ids, periods, monthly, raw, region = load_panel(cfg, root)
    graph_dir = root / s['graphs_dir']
    graphs, inputs = [], {}
    for t, period in enumerate(periods):
        nodes = graph_dir / f'nodes_{period}.json'
        features = graph_dir / f'features_{period}.npy'
        path = graph_dir / f'{period}.npz'
        if json.loads(nodes.read_text('utf-8')) != ids or not np.allclose(np.load(features), monthly[t], rtol=0, atol=1e-12):
            raise ValueError('Monthly graph inputs differ from the panel features: ' + period)
        graphs.append(load_npz(path).tocsr())
        for p in (path, nodes, features):
            inputs[str(p.relative_to(root))] = sha256(p)
    temporal = {}
    for omega in s['omegas']:
        for seed in s['seeds']:
            path = root / s['temporal_dir'] / f'temporal_omega{omega:g}_seed{seed}.json.gz'
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                stored = json.load(stream)
            if stored['ids'] != ids or stored['periods'] != periods:
                raise ValueError('Temporal labels do not match the panel cohort: ' + path.name)
            temporal[(omega, seed)] = ([np.asarray(z) for z in stored['labels']], [m.get('NewmanQ') for m in stored['monthly_metrics']])
            inputs[str(path.relative_to(root))] = sha256(path)
    vintage = root / 'reports/experiments/2026-09-23-v2/validation'
    for name in ('frozen_prototypes.json', 'reference_assignments.csv'):
        inputs[str((vintage / name).relative_to(root))] = sha256(vintage / name)
    inputs['data/processed/panel.csv'] = sha256(root / 'data/processed/panel.csv')
    road_path = root / s['road_graph']
    if sha256(road_path) != s['road_graph_sha256']:
        raise ValueError('Road graph fingerprint differs')
    inputs[s['road_graph']] = s['road_graph_sha256']
    road = load_npz(road_path).tocsr()
    if road.shape != (len(ids), len(ids)):
        raise ValueError('Road graph shape differs')
    mixed = [mix_layers(g, road, s['attribute_graph_weight']) for g in graphs]
    return ids, periods, monthly, raw, graphs, mixed, temporal, inputs


def run(cfg, root):
    root = Path(root)
    s = cfg['temporal_quality']
    out = root / s['output']
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ids, periods, monthly, raw, graphs, mixed, temporal, inputs = load_inputs(cfg, root)
    profile_months = sum(period <= cfg['features']['calibration_end'] for period in periods)
    annual = np.median(monthly[:profile_months], axis=0)
    centers, _ = reference_partition(root, ids, annual)
    fixed_raw, fixed, season = fixed_assignments(monthly, centers)
    refit = refit_kmeans(monthly, s['kmeans']['n_clusters'], s['kmeans']['n_init'], cfg['seed'])
    runs = {('temporal_leiden', omega, seed): value for (omega, seed), value in temporal.items()}
    runs[('kmeans4_fixed_2023', None, None)] = (fixed, None)
    runs[('kmeans4_fixed_2023_no_season', None, None)] = (fixed_raw, None)
    runs[('kmeans4_monthly_refit', None, cfg['seed'])] = (refit, None)
    rows, summaries = [], {}
    for key, (labels, fit_mq) in runs.items():
        part = series_rows(*key, labels, monthly, graphs, periods, mixed, fit_mq)
        rows.extend(part)
        summaries[key] = summarise(part)
        print(json.dumps({'run': list(key), 'MQ': summaries[key]['MQ'], 'MQ_next': summaries[key]['MQ_next']}), flush=True)
    table = pd.DataFrame(rows)[['method', 'omega', 'seed', 'period', 'k', 'min_cluster_size'] + COLUMNS]
    table['seed'] = table['seed'].astype('Int64')
    table.round(6).to_csv(out / 'monthly_icvi.csv', index=False)
    tradeoff, choice = omega_tradeoff(summaries, {k: v[0] for k, v in temporal.items()}, s['omegas'], s['seeds'], s['main_seed'],
                                      s['omega_rule']['max_relative_drop'])
    pd.DataFrame(tradeoff).round(6).to_csv(out / 'omega_tradeoff.csv', index=False)
    retention = {f'{m}|{o}|{sd}': (v['MQ_next'] / v['MQ'] if v['MQ'] else None) for (m, o, sd), v in summaries.items()}
    start, end = s['drift']['start'], s['drift']['end']
    profiles = {'kmeans4_fixed_2023': size_profile(fixed, periods, start, end, True),
                'kmeans4_fixed_2023_no_season': size_profile(fixed_raw, periods, start, end, True),
                'kmeans4_monthly_refit': size_profile(refit, periods, start, end)}
    for omega in sorted({s['current_omega'], choice['selected'], 0}):
        profiles[f'temporal_leiden_omega{omega:g}'] = size_profile(temporal[(omega, s['main_seed'])][0], periods, start, end)
    summary = {'n': len(ids), 'months': len(periods), 'evaluation_graph': 'monthly attribute kNN graph data/processed/graphs/{period}.npz, binary convention of metrics.all_metrics',
               'mixed_graph': 'attribute_graph_weight * attribute + (1 - weight) * road, unit mean strength per layer, binary convention; the graph temporal Leiden optimised; *_mix columns',
               'methods': [{'method': m, 'omega': o, 'seed': sd, **v} for (m, o, sd), v in summaries.items()],
               'MQ_next_over_MQ': retention, 'omega_rule': s['omega_rule'], 'omega_choice': choice | {'current_omega': s['current_omega'],
               'matches_current': choice['selected'] == s['current_omega']},
               'omega_choice_sensitivity_mixed_graph': select_omega([{'omega': r['omega'], 'MQ_mean': r['MQ_mix_mean']} for r in tradeoff], s['omega_rule']['max_relative_drop']),
               'omega_tradeoff': tradeoff, 'size_profiles': profiles,
               'drift': drift_decomposition(monthly, raw, periods, centers, season, annual, cfg)}
    write_json(out / 'summary.json', clean(summary))
    modules = ['sbercluster/temporal_quality.py', 'sbercluster/edge_study.py', 'sbercluster/metrics.py', 'sbercluster/round2.py',
               'sbercluster/features.py', 'sbercluster/graph.py', 'sbercluster/contest_graphs.py', 'scripts/temporal_quality.py']
    write_json(out / 'provenance.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg, 'inputs_sha256': inputs,
                                         'implementation_sha256': {m: sha256(root / m) for m in modules},
                                         'outputs_sha256': {name: sha256(out / name) for name in ('summary.json', 'monthly_icvi.csv', 'omega_tradeoff.csv')},
                                         'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas']},
                                         'seconds': time.perf_counter() - started})
    return out
