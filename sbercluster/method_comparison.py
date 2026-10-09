"""Compare every stored annual partition on shared axes and mark which axes each method used for fitting.

No partition is refitted. Each method is scored on the 2023 features and graphs, on the same axes in 2024 with its
2023 labels kept fixed, and on graphs it never saw (roads R, co-movement C). Only axes a method did not optimise
are used to name winners. Axes that entered an earlier version of a method's selection rule are marked separately and
are also left out of that method's ranking. A null baseline without any training on spending (registry municipal type) is
scored on the same axes. Temporal Leiden is summarised separately across its three seeds.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .contest_graphs import read_road_graph
from .dynamics import cluster_events
from .edge_study import load_panel, published_partitions, reference_partition
from .io import sha256, write_json
from .network_core import build_layers, canonical, evaluate, select_variant
from .temporal_groups import event_owners, rejoin, track_labels

FEATURE = ('SW', 'CH', 'S_Dbw')
NETWORK = ('AVI', 'AVU', 'MQ')
DIRECTION = {'SW': 1, 'CH': 1, 'S_Dbw': -1, 'AVI': 1, 'AVU': -1, 'MQ': 1}
AXES = ('X23', 'P23', 'C23', 'R', 'X24', 'P24', 'C24')
YEAR_2024 = ('X24', 'P24', 'C24')
EXTERNAL = ('R', 'C23', 'C24')
UNSEEN = ('year_2024', 'external')
# Until 2026-10-07 the network-core rule (archive/configs/network_core_2026-10-06.json) built its Pareto front on oot_SW
# (X24) and oot_C_MQ (C24); that front held one variant, so these two axes decided the first selection. The rank sum also
# had half-weighted oot_P_* indices (P24); WITH_P24 is the stricter sensitivity that excludes P24 as well.
FIRST_SELECTION = {'network_typology': {'X24', 'C24'}, 'network_two_level': {'X24', 'C24'}}
WITH_P24 = {name: axes | {'P24'} for name, axes in FIRST_SELECTION.items()}
REGISTRY_MERGE = {'муниципальный округ': 'муниципальный район'}


def family(name):
    """Method family and the axes it used for fitting (P is the kNN graph of the same annual features as X)."""
    if name.startswith('temporal_leiden'):
        return 'временной Leiden (24 месяца, профиль и дороги)', {'X23', 'P23', 'R', 'X24', 'P24'}
    if name in ('network_typology', 'network_two_level'):
        return 'сетевая типология Leiden (профиль и совместное движение)', {'X23', 'P23', 'C23'}
    if name == 'leiden_comovement_only':
        return 'Leiden только на совместном движении', {'C23'}
    if name == 'spectral3_profile_comovement':
        return 'спектральная на графе профиля и совместного движения', {'X23', 'P23', 'C23'}
    if name.startswith('registry_type'):
        return 'тип МО по реестру (без обучения)', set()
    if name.startswith('joint_road'):
        return 'совместная модель: профиль и дороги', {'X23', 'P23', 'R'}
    if name.startswith('dmon'):
        return 'DMoN (признаки на дорожном графе)', {'X23', 'P23', 'R'}
    if name == 'spectral_transport_k4':
        return 'спектральная на дорожном графе', {'R'}
    prefixes = (('kmeans', 'KMeans'), ('ward', 'Ward'), ('gmm', 'Gaussian mixture'), ('huber75', 'Huber75'),
                ('spectral', 'спектральная на графе профилей'), ('leiden', 'Leiden на графе профилей'),
                ('joint_same_region', 'совместная модель: профиль и регион'))
    for prefix, label in prefixes:
        if name.startswith(prefix):
            return label, {'X23', 'P23'}
    raise ValueError('Unknown method family: ' + name)


def axis_role(axis, fitted, selected=frozenset()):
    """fit: used for fitting; selection: used by an earlier selection rule; otherwise unseen (year_2024 or external)."""
    if axis in fitted:
        return 'fit'
    if axis in selected:
        return 'selection'
    return 'year_2024' if axis in YEAR_2024 else 'external'


def reassign_roles(rows, selection):
    """Copy of rows with axis roles recomputed under another first-selection mapping (for sensitivity)."""
    out = []
    for r in rows:
        fitted = set(r['fitted_axes'].split()) if r['fitted_axes'] else set()
        out.append({**r, **{f'{axis}_role': axis_role(axis, fitted, selection.get(r['partition'], set())) for axis in AXES}})
    return out


def baseline_partitions(root, ids):
    """Null partitions without training on spending: registry municipal type (4 types, and 3 with municipal okrugs merged
    into districts)."""
    frame = pd.read_csv(root / 'reports/external-municipal/municipal_indicators.csv', dtype={'entity_id': str}).set_index('entity_id').reindex(ids)
    kind = frame['municipal_district_type']
    if kind.isna().any():
        raise ValueError('Municipal type does not cover the cohort')
    parts = {'registry_type4': pd.factorize(kind, sort=True)[0], 'registry_type3': pd.factorize(kind.replace(REGISTRY_MERGE), sort=True)[0]}
    return parts, {'registry_types': {str(t): int(n) for t, n in kind.value_counts().sort_index().items()}}


def collect_partitions(root, ids, reference):
    parts = {name: z for name, z in published_partitions(root, ids).items()}
    core = pd.read_csv(root / 'reports/network-core/assignments.csv', dtype={'entity_id': str}).set_index('entity_id').reindex(ids)
    if core.isna().any().any():
        raise ValueError('Network-core assignments do not cover the cohort')
    parts['network_typology'] = core['group'].to_numpy()
    parts['network_two_level'] = pd.factorize(core['type'].astype(str))[0]
    parts['leiden_profile_only'] = core['leiden_profile_only'].to_numpy()
    parts['leiden_comovement_only'] = core['leiden_comovement_only'].to_numpy()
    parts['spectral3_profile_comovement'] = core['spectral3_a0.75'].to_numpy()
    frontier = json.loads((root / 'reports/model-frontier-2026-10-03/constrained/labels.json').read_text('utf-8'))
    if frontier['ids'] != ids:
        raise ValueError('Huber75 labels are not aligned with the panel')
    parts['huber75_k4'] = np.asarray(frontier['labels']['sw_constrained_huber75'])
    if not np.array_equal(canonical(parts['kmeans_k4']), canonical(reference)):
        raise ValueError('Published kmeans_k4 no longer equals the frozen reference')
    return {name: canonical(pd.factorize(np.asarray(z))[0]) for name, z in parts.items()}


def temporal_annual(root, path, ids, periods):
    with gzip.open(root / path, 'rt', encoding='utf-8') as stream:
        temporal = json.load(stream)
    if temporal['ids'] != ids or temporal['periods'] != periods:
        raise ValueError('Temporal labels do not match the panel cohort')
    months = [i for i, p in enumerate(periods) if p.startswith('2023')]
    labels = np.asarray(temporal['labels'])[months]
    return canonical(np.array([Counter(column).most_common(1)[0][0] for column in labels.T])), temporal


def score(name, z, x23, x24, g23, g24, reference, selection=None):
    label, fitted = family(name)
    selected = (FIRST_SELECTION if selection is None else selection).get(name, set())
    row = {'partition': name, 'family': label, 'k': int(z.max() + 1), 'min_size': int(np.bincount(z).min()),
           'ARI_kmeans4': round(float(adjusted_rand_score(reference, z)), 6), 'fitted_axes': ' '.join(sorted(fitted)),
           'selection_axes': ' '.join(sorted(selected))}
    raw = {**evaluate(z, x23, {'P': g23['P'], 'C': g23['C'], 'R': g23['R']}, 'y23'),
           **evaluate(z, x24, {'P': g24['P'], 'C': g24['C']}, 'y24')}
    for axis in AXES:
        year = '24' if axis.endswith('24') else '23'
        graph = None if axis.startswith('X') else axis[0]
        for metric in (FEATURE if graph is None else NETWORK):
            value = raw[f'y{year}_{metric}' if graph is None else f'y{year}_{graph}_{metric}']
            row[f'{axis}_{metric}'] = None if value is None else round(float(value), 6)
        row[f'{axis}_role'] = axis_role(axis, fitted, selected)
    return row


def degenerate(rows, k, key, tol=1e-9):
    """True when every partition with K=k has the same value (binary AVU is 1 at K=2 and 2/3 at K=3 for any partition)."""
    values = [r[key] for r in rows if r['k'] == k and r.get(key) is not None]
    return len(values) > 1 and max(values) - min(values) < tol


def winners(rows, k, metrics, min_size):
    """Best value per (axis, metric) among methods with K=k that neither fitted nor selected on that axis.
    Metrics that are constant across all K=k partitions are skipped."""
    out = {}
    for axis in AXES:
        for metric in metrics.get(axis[0] if axis != 'R' else 'R', ()):
            if degenerate(rows, k, f'{axis}_{metric}'):
                continue
            pool = [r for r in rows if r['k'] == k and r['min_size'] >= min_size and r[f'{axis}_role'] in UNSEEN
                    and r.get(f'{axis}_{metric}') is not None]
            if len(pool) < 2:
                continue
            best = max(pool, key=lambda r: (DIRECTION[metric] * r[f'{axis}_{metric}'], r['partition']))
            out[f'{axis}_{metric}'] = {'partition': best['partition'], 'family': best['family'], 'value': best[f'{axis}_{metric}'],
                                       'candidates': len(pool)}
    return out


def neutral_rank(rows, k, keys, min_size):
    """Mean rank on the neutral axes. On each axis only K=k methods that neither fitted nor selected on it are ranked
    (1 is best); a method's mean is taken over the axes where it was ranked, and the count of those axes is reported."""
    base = [r for r in rows if r['k'] == k and r['min_size'] >= min_size]
    places = {r['partition']: {} for r in base}
    for key in keys:
        axis, metric = key.split('_', 1)
        pool = [r for r in base if r[f'{axis}_role'] in UNSEEN and r.get(key) is not None]
        if len(pool) < 2:
            continue
        for position, r in enumerate(sorted(pool, key=lambda r: (-DIRECTION[metric] * r[key], r['partition'])), 1):
            places[r['partition']][key] = [position, len(pool)]
    by = {r['partition']: r for r in base}
    table = [{'partition': name, 'family': by[name]['family'], 'mean_rank': round(float(np.mean([p[0] for p in own.values()])), 3),
              'axes': len(own), 'places': own, **{key: by[name][key] for key in keys}} for name, own in places.items() if own]
    if len(table) < 2:
        return []
    return sorted(table, key=lambda t: (t['mean_rank'], -t['axes'], t['partition']))


def agreement(parts, baselines, monthly_labels, months):
    """ARI and NMI of each null baseline with the network typology, KMeans4, the annual projection of temporal Leiden
    and (median over the 24 months) the monthly temporal Leiden labels."""
    out = {}
    for name in baselines:
        z = parts[name]
        row = {}
        for other in ('network_typology', 'kmeans_k4', 'temporal_leiden_seed1729'):
            row[other] = {'ARI': round(float(adjusted_rand_score(parts[other], z)), 4),
                          'NMI': round(float(normalized_mutual_info_score(parts[other], z)), 4)}
        row['temporal_leiden_seed1729_monthly_median'] = {
            'ARI': round(float(np.median([adjusted_rand_score(m, z) for m in monthly_labels])), 4),
            'NMI': round(float(np.median([normalized_mutual_info_score(m, z) for m in monthly_labels])), 4), 'months': months}
        out[name] = row
    return out


def selection_audit(root, path):
    """Network-core level-1 rule recomputed from candidates.csv (6 decimals): as published, without AVU, and with the
    first (pre-2026-10-07) version that used 2024 indices (oot_*)."""
    rule = json.loads((root / 'configs/network_core.json').read_text('utf-8'))['network_core']['selection']
    frame = pd.read_csv(root / path)
    rows = [{key: (None if isinstance(v, float) and np.isnan(v) else v) for key, v in r.items()} for r in frame[frame.scope == 'all'].to_dict('records')]

    def run_rule(r):
        picked, info = select_variant(rows, r)
        top = list(info['rank_sums'].items())[:2]
        return {'selected': picked, 'front': info['front'], 'top_rank_sums': {v: round(s, 2) for v, s in top}}
    no_avu = {**rule, 'rank': [item for item in rule['rank'] if 'AVU' not in item['key']]}
    first = {**rule, 'pareto': [{**p, 'key': p['key'].replace('in_', 'oot_', 1)} for p in rule['pareto']],
             'rank': [{**p, 'key': p['key'].replace('in_', 'oot_', 1)} for p in rule['rank']]}
    return {'source': path, 'current_rule': run_rule(rule), 'current_rule_without_AVU': run_rule(no_avu),
            'first_rule_oot': run_rule(first)}


def temporal_tracks(root, path, ids, periods, threshold, change, min_months, rejoin_jaccard):
    with gzip.open(root / path, 'rt', encoding='utf-8') as stream:
        temporal = json.load(stream)
    if temporal['ids'] != ids or temporal['periods'] != periods:
        raise ValueError('Temporal labels do not match the panel cohort')
    labels = [list(map(int, row)) for row in temporal['labels']]
    series = [np.asarray(row) for row in labels]
    steps = [{'events': cluster_events(ids, series[t - 1], ids, series[t], threshold, change)} for t in range(1, len(series))]
    sizes = [{str(label): int(count) for label, count in zip(*np.unique(z, return_counts=True))} for z in series]
    owners, _ = event_owners(sizes, steps)
    tracks = rejoin(track_labels(labels, owners), rejoin_jaccard)
    alive = Counter(track for row in tracks for track in set(row))
    stable = sorted(k for k, n in alive.items() if n >= min_months)
    cells = {k: {(i, t) for t, row in enumerate(tracks) for i, track in enumerate(row) if track == k} for k in stable}
    return {'tracks': len(alive), 'stable': len(stable), 'groups_per_month': [len(set(row)) for row in labels],
            'stable_cover': round(sum(len(c) for c in cells.values()) / (len(ids) * len(periods)), 4)}, cells


def temporal_seeds(root, cfg, ids, periods):
    t = cfg['temporal']
    per_seed, cells = {}, {}
    for seed, path in t['labels'].items():
        per_seed[seed], cells[seed] = temporal_tracks(root, path, ids, periods, t['jaccard_threshold'], t['size_change'],
                                                      t['min_months'], t['rejoin_jaccard'])
    base = t['base_seed']
    matches = []
    for group, own in sorted(cells[base].items(), key=lambda item: -len(item[1])):
        row = {'size_cells': len(own)}
        for seed in cells:
            if seed == base:
                continue
            best = max((len(own & other) / len(own | other) for other in cells[seed].values()), default=0.0)
            row[seed] = round(best, 4)
        matches.append(row)
    others = [s for s in cells if s != base]
    shared = sum(all(m[s] >= t['match_jaccard'] for s in others) for m in matches)
    return {'base_seed': base, 'match_jaccard': t['match_jaccard'], 'per_seed': per_seed, 'base_group_best_jaccard': matches,
            'base_groups_found_in_all_seeds': shared}


def run(cfg, root):
    root = Path(root)
    mc = cfg['method_comparison']
    out = root / mc['output']
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ids, periods, monthly, raw, region = load_panel(cfg, root)
    fit = slice(periods.index('2023-01-01'), periods.index('2023-12-01') + 1)
    test = slice(periods.index('2024-01-01'), periods.index('2024-12-01') + 1)
    x23, l23, _ = build_layers(raw, monthly, fit, mc['k'], mc['max_lag'])
    x24, l24, _ = build_layers(raw, monthly, test, mc['k'], mc['max_lag'])
    _, reference = reference_partition(root, ids, x23)
    road, _, _ = read_road_graph(root / mc['source_dir'] / 'hackathonlicence/connection.parquet', ids, mc['k'])
    g23, g24 = {**l23, 'R': road}, dict(l24)
    parts = collect_partitions(root, ids, reference)
    monthly_main = None
    for seed, path in mc['temporal']['labels'].items():
        parts[f'temporal_leiden_seed{seed}'], temporal = temporal_annual(root, path, ids, periods)
        if seed == mc['temporal']['base_seed']:
            monthly_main = [np.asarray(row) for row in temporal['labels']]
    nulls, null_info = baseline_partitions(root, ids)
    parts.update({name: canonical(pd.factorize(np.asarray(z))[0]) for name, z in nulls.items()})
    rows = [score(name, z, x23, x24, g23, g24, reference) for name, z in sorted(parts.items()) if z.max() >= 1]
    metrics = {'X': FEATURE, 'P': NETWORK, 'C': NETWORK, 'R': NETWORK}
    neutral_keys = mc['neutral_keys']
    strict_rows = reassign_roles(rows, WITH_P24)
    summary = {'n': len(ids), 'partitions': len(rows), 'axes': {
                   'X23': '2023 annual profile features', 'P23': '2023 kNN profile graph', 'C23': '2023 lagged co-movement graph',
                   'R': 'road graph (observed highway pairs)', 'X24': '2024 features, 2023 labels kept',
                   'P24': '2024 kNN profile graph, 2023 labels kept', 'C24': '2024 co-movement graph, 2023 labels kept'},
               'roles': {'fit': 'axis used by the method for fitting; circular',
                         'selection': 'axis used by the first (pre-2026-10-07) version of the rule that selected this partition; '
                                      'excluded from its winners and ranks like a fitted axis',
                         'year_2024': 'same kind of axis one year later', 'external': 'graph the method never saw'},
               'first_selection': {name: sorted(axes) for name, axes in FIRST_SELECTION.items()},
               'degenerate': {str(k): sorted(f'{axis}_{m}' for axis in AXES for m in NETWORK if axis[0] != 'X' and degenerate(rows, k, f'{axis}_{m}'))
                              for k in mc['k_values']},
               'neutral_keys': neutral_keys,
               'min_size': mc['min_size'],
               'winners': {str(k): winners(rows, k, metrics, mc['min_size']) for k in mc['k_values']},
               'neutral_rank': {str(k): neutral_rank(rows, k, neutral_keys, mc['min_size']) for k in mc['k_values']},
               'neutral_rank_excluding_P24_too': {str(k): neutral_rank(strict_rows, k, neutral_keys, mc['min_size']) for k in (3,)},
               'baselines': {'partitions': sorted(nulls), **null_info,
                             'agreement': agreement(parts, sorted(nulls), monthly_main, len(monthly_main))},
               'network_core_selection_audit': selection_audit(root, 'reports/network-core/candidates.csv'),
               'temporal_seeds': temporal_seeds(root, mc, ids, periods)}
    pd.DataFrame(rows).to_csv(out / 'table.csv', index=False)
    write_json(out / 'summary.json', summary)
    (out / 'README.md').write_text(readme(summary, rows), 'utf-8', newline='\n')
    if mc.get('fragment'):
        (root / mc['fragment']).write_text(fragment(summary, rows), 'utf-8', newline='\n')
    write_json(out / 'provenance.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg, 'panel_sha256': sha256(root / 'data/processed/panel.csv'),
        'inputs': {p: sha256(root / p) for p in sorted({'reports/network-core/assignments.csv', 'reports/network-core/candidates.csv',
                                                         'reports/external-municipal/municipal_indicators.csv', 'configs/network_core.json',
                                                         'reports/model-frontier-2026-10-03/constrained/labels.json',
                                                         *mc['temporal']['labels'].values()})},
        'implementation': {p.name: sha256(p) for p in [root / 'sbercluster/method_comparison.py', root / 'sbercluster/network_core.py',
                                                       root / 'sbercluster/metrics.py']},
        'seconds': round(time.perf_counter() - started, 1), 'refitting': False})
    return {'output': str(out), 'partitions': len(rows), 'seconds': round(time.perf_counter() - started, 1)}


NAMES = {'kmeans_k3': 'KMeans', 'ward_k3': 'Ward', 'spectral3_profile_comovement': 'Спектральная, профиль и движение',
         'network_typology': 'Каркас (сетевая типология)', 'joint_road_k3': 'Совместная модель с дорогами',
         'kmeans_k4': 'KMeans4 (базовая линия)', 'ward_k4': 'Ward', 'gmm_full_k4': 'Gaussian mixture', 'huber75_k4': 'Huber75',
         'spectral_global_k4': 'Спектральная, глобальный масштаб', 'spectral_local_k4': 'Спектральная, локальный масштаб',
         'leiden_profile_only': 'Leiden, только профиль', 'leiden_comovement_only': 'Leiden, только совместное движение',
         'joint_road_k4': 'Совместная модель с дорогами', 'joint_same_region_k4': 'Совместная модель с регионом',
         'joint_road_sensitivity_k4': 'Совместная модель с дорогами, другой вес', 'dmon_k4_seed1729': 'DMoN',
         'dmon_k4_seed2718': 'DMoN, seed 2718', 'dmon_k4_seed3141': 'DMoN, seed 3141', 'spectral_transport_k4': 'Спектральная на дорогах',
         'registry_type3': 'Тип МО по реестру, 3 типа (без обучения)', 'registry_type4': 'Тип МО по реестру, 4 типа (без обучения)'}
SHOWN = {3: ['network_typology', 'kmeans_k3', 'ward_k3', 'spectral3_profile_comovement', 'joint_road_k3', 'registry_type3'],
         4: ['kmeans_k4', 'ward_k4', 'gmm_full_k4', 'huber75_k4', 'spectral_global_k4', 'spectral_local_k4', 'leiden_profile_only',
             'leiden_comovement_only', 'joint_road_k4', 'joint_same_region_k4', 'dmon_k4_seed1729', 'spectral_transport_k4',
             'registry_type4']}
COLUMNS = [('X23_SW', 'SW 2023'), ('P23_MQ', 'MQ, профиль 2023'), ('C23_MQ', 'MQ, движение 2023'), ('R_MQ', 'MQ, дороги'),
           ('X24_SW', 'SW 2024'), ('P24_MQ', 'MQ, профиль 2024'), ('C24_MQ', 'MQ, движение 2024')]
AXIS_NAMES = {'R_MQ': 'MQ на дорожном графе', 'C24_MQ': 'MQ на графе совместного движения 2024 года',
              'P24_MQ': 'MQ на графе профилей 2024 года', 'X24_SW': 'силуэт 2024 года'}
MARK = {'fit': '*', 'selection': '†'}


def _num(value, digits=3):
    return 'н/д' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def _name(partition):
    return NAMES.get(partition, partition)


def _cell(row, key):
    return _num(row.get(key)) + MARK.get(row[f'{key.split("_")[0]}_role'], '')


def _verdict(summary, k):
    best = summary['winners'][str(k)]
    return [f'{AXIS_NAMES[key]}: {_name(best[key]["partition"])} ({_num(best[key]["value"])})' for key in summary['neutral_keys'] if key in best]


def _ranks(ranked):
    return ', '.join(f'{_name(r["partition"])} {_num(r["mean_rank"], 2)}' + ('' if r['axes'] == 4 else f' ({r["axes"]} из 4 осей)')
                     for r in ranked)


def _lead(summary, by):
    """Short plain-language findings computed from the summary."""
    r3 = summary['neutral_rank']['3']
    r4 = summary['neutral_rank']['4']
    net, reg, reg4, km = by['network_typology'], by['registry_type3'], by['registry_type4'], by['kmeans_k3']
    place = {r['partition']: r for r in r3}
    road = place['network_typology']['places']['R_MQ']
    agree = summary['baselines']['agreement']
    lines = [
        f'- При K = 3 на осях, которых метод не видел, первым идёт {_name(r3[0]["partition"])} (средний ранг {_num(r3[0]["mean_rank"], 2)} '
        f'по {r3[0]["axes"]} осям). У сетевой типологии {_num(place["network_typology"]["mean_rank"], 2)} по {place["network_typology"]["axes"]} '
        f'осям (дороги и граф профилей 2024 года): силуэт 2024 года и MQ совместного движения 2024 года решали первую версию её правила отбора, поэтому в её ранг '
        f'не входят. На дорогах она {road[0]}-я из {road[1]} (MQ {_num(net["R_MQ"])} против {_num(km["R_MQ"])} у KMeans3).',
        f'- Сеть против справочника. Тип МО по реестру из трёх групп (городские округа; районы и муниципальные округа; '
        f'внутригородские МО) на дорожном графе лучше сетевой типологии: MQ {_num(reg["R_MQ"])} против {_num(net["R_MQ"])}. '
        f'На осях расходов 2024 года сеть сильнее справочника (силуэт {_num(net["X24_SW"])} против {_num(reg["X24_SW"])}, '
        f'MQ профилей {_num(net["P24_MQ"])} против {_num(reg["P24_MQ"])}, MQ совместного движения {_num(net["C24_MQ"])} против '
        f'{_num(reg["C24_MQ"])}), но силуэт и совместное движение служат осями её первого отбора; справочник на них уступает и KMeans3.',
        f'- Каркас заметно совпадает со справочником: ARI {_num(agree["registry_type3"]["network_typology"]["ARI"])} '
        f'и NMI {_num(agree["registry_type3"]["network_typology"]["NMI"])} с тремя типами МО, '
        f'{_num(agree["registry_type4"]["network_typology"]["ARI"])} и {_num(agree["registry_type4"]["network_typology"]["NMI"])} с четырьмя. '
        f'Четыре типа по реестру при K = 4 на дорогах лучше KMeans4 ({_num(reg4["R_MQ"])} против {_num(by["kmeans_k4"]["R_MQ"])}), '
        f'на осях расходов хуже всех обученных методов.',
        f'- При K = 4 первой идёт {_name(r4[0]["partition"])} ({_num(r4[0]["mean_rank"], 2)}), '
        f'у KMeans4 {_num(next(r["mean_rank"] for r in r4 if r["partition"] == "kmeans_k4"), 2)}.']
    return lines


def readme(summary, rows):
    by = {r['partition']: r for r in rows}
    t = summary['temporal_seeds']
    seeds = t['per_seed']
    audit = summary['network_core_selection_audit']
    cur, no_avu = audit['current_rule']['top_rank_sums'], audit['current_rule_without_AVU']['top_rank_sums']
    strict = {r['partition']: r for r in summary['neutral_rank_excluding_P24_too']['3']}['network_typology']
    agree = summary['baselines']['agreement']
    out = ['# Сравнение методов на общих осях', '', 'Коротко:', '', *_lead(summary, by), '',
           '## Как я сравниваю', '',
           'Я оцениваю все сохранённые годовые разбиения 2023 года без переобучения на семи осях: признаки и граф профилей 2023 года, '
           'граф совместного движения 2023 года, дорожный граф, а также признаки, граф профилей и граф совместного движения 2024 года '
           'при неизменных метках 2023 года. Звёздочка (*) отмечает ось, на которой метод обучался: такая оценка круговая. '
           'Крестик (†) отмечает ось первого отбора: до 7 октября каркас выбирался по парето-фронту силуэта 2024 года (X24) и MQ на '
           'графе совместного движения 2024 года (C24) (`archive/configs/network_core_2026-10-06.json`). Обе пометки исключают ось из '
           'выбора победителя и из ранга этого метода.',
           '',
           f'Победителя я называю только среди разбиений с одинаковым K и минимальной группой не меньше {summary["min_size"]} МО. '
           'Средний ранг считаю по четырём осям: дороги и три оси 2024 года. На каждой оси ранжирую только методы, которые её не видели; '
           'если метод видел часть осей, его ранг берётся по остальным, и число осей указано в скобках. Модель с регионом дорог не видела, '
           'но регион и дорожная связность географически близки, поэтому на дорожном графе у неё косвенное преимущество. Спектральная '
           'кластеризация и однослойные Leiden берут K, α и γ у выбранного варианта каркаса, но их собственные метки по осям 2024 года '
           'не отбирались, поэтому крестика у них нет.', '',
           'Нулевые линии не обучаются на расходах. Тип МО взят из реестра (`reports/external-municipal/municipal_indicators.csv`): '
           + ', '.join(f'{name} {n}' for name, n in summary['baselines']['registry_types'].items())
           + '. В варианте из трёх типов муниципальные округа объединены с районами.', '',
           'AVU при K = 2 и K = 3 одинаков у любого разбиения (1 и 2/3: сумма по упорядоченным парам при K = 3 равна 2), поэтому при этих K '
           'я его не ранжирую, а между разными K в этом сравнении AVU не сравнивается вообще. Правило отбора каркаса '
           '(`configs/network_core.json`) складывает ранги AVU кандидатов с разным K. По `reports/network-core/candidates.csv` суммы '
           f'рангов двух лучших вариантов равны {_num(list(cur.values())[0], 1)} и {_num(list(cur.values())[1], 1)}, без AVU '
           f'{_num(list(no_avu.values())[0], 1)} и {_num(list(no_avu.values())[1], 1)}; выбран тот же вариант '
           f'`{audit["current_rule_without_AVU"]["selected"]}`. Первая версия правила по индексам 2024 года выбирает его же: парето-фронт '
           f'состоит из одного варианта.', '',
           f'Всего разбиений: {summary["partitions"]}, из них {len(summary["baselines"]["partitions"])} нулевые линии по реестру. '
           'Полная таблица: `table.csv`, сводка: `summary.json`.', '']
    for k in (3, 4):
        out += [f'## K = {k}', '', '| Метод | Мин. группа | ' + ' | '.join(c for _, c in COLUMNS) + ' |',
                '|---|---:|' + '---:|' * len(COLUMNS)]
        for name in SHOWN[k]:
            r = by[name]
            out.append(f'| {NAMES[name]} | {r["min_size"]} | ' + ' | '.join(_cell(r, key) for key, _ in COLUMNS) + ' |')
        out += ['', 'Лучшие на осях, которые метод не видел ни при обучении, ни при отборе: ' + '; '.join(_verdict(summary, k)) + '.',
                'Средний ранг: ' + _ranks(summary['neutral_rank'][str(k)]) + '.', '']
        if k == 3:
            out += [f'В сумму рангов первой версии правила с весом 0,5 входили и индексы на графе профилей 2024 года (P24), но решал '
                    f'парето-фронт из одного варианта. Если исключить у сетевой типологии и P24, её средний ранг '
                    f'{_num(strict["mean_rank"], 2)} по {strict["axes"]} оси.', '']
    out += ['## Совпадение нулевых линий с моделями', '',
            '| Нулевая линия | K | Сетевая типология, ARI / NMI | KMeans4, ARI / NMI | Временной Leiden по месяцам, медиана ARI / NMI |',
            '|---|---:|---|---|---|']
    for name in summary['baselines']['partitions']:
        a = agree[name]
        out.append(f'| {NAMES[name]} | {by[name]["k"]} | ' + ' | '.join(
            f'{_num(a[o]["ARI"])} / {_num(a[o]["NMI"])}' for o in ('network_typology', 'kmeans_k4', 'temporal_leiden_seed1729_monthly_median')) + ' |')
    out += ['', 'Временной Leiden взят с основным seed 1729 и сравнивается с нулевой линией в каждом из 24 месяцев.', '',
            '## Временной Leiden на трёх seed', '',
            '| Seed | Треков | Устойчивых групп (не меньше 6 месяцев) | Групп в месяц | Доля наблюдений в устойчивых группах |',
            '|---|---:|---:|---|---:|']
    for seed, s in seeds.items():
        g = s['groups_per_month']
        out.append(f'| {seed} | {s["tracks"]} | {s["stable"]} | от {min(g)} до {max(g)} | {_num(100 * s["stable_cover"], 1)} % |')
    shared = t['base_groups_found_in_all_seeds']
    out += ['', f'Из {len(t["base_group_best_jaccard"])} устойчивых групп seed {t["base_seed"]} в обоих других seed находятся {shared} '
            f'(Жаккар по парам «МО × месяц» не ниже {_num(t["match_jaccard"], 1)}). Лучшие совпадения по группам, от крупной к мелкой: '
            + '; '.join(' и '.join(_num(m[s], 2) for s in m if s != 'size_cells') for m in t['base_group_best_jaccard']) + '.', '',
            '## Как повторить', '', '```sh', 'python scripts/method_comparison.py', '```', '',
            'Расчёт читает сохранённые метки и не обучает модели; на ноутбуке он занимает около 10 секунд.', '']
    return '\n'.join(out)


def fragment(summary, rows):
    by = {r['partition']: r for r in rows}
    t = summary['temporal_seeds']

    def table(k):
        head = ''.join(f'<th scope="col" class="num">{c}</th>' for _, c in COLUMNS)
        body = ''.join(f'<tr><th scope="row">{NAMES[n]}</th><td class="num">{by[n]["min_size"]}</td>'
                       + ''.join(f'<td class="num">{_cell(by[n], key)}</td>' for key, _ in COLUMNS) + '</tr>' for n in SHOWN[k])
        return (f'<div class="table-scroll" tabindex="0" role="region" aria-label="Сравнение методов при K={k}"><table>'
                f'<caption>K = {k}: 2 016 МО, метки 2023 года без переобучения</caption><thead><tr><th scope="col">Метод</th>'
                f'<th scope="col" class="num">Мин. группа</th>{head}</tr></thead><tbody>{body}</tbody></table></div>')

    def verdict(k):
        return ('<p class="control-help">Лучшие на осях, которые метод не видел: ' + '; '.join(_verdict(summary, k)) + '. '
                'Средний ранг: ' + _ranks(summary['neutral_rank'][str(k)][:4]) + '.</p>')
    seeds = '; '.join(f'seed {s}: {v["stable"]}' for s, v in t['per_seed'].items())
    lead = ' '.join(line[2:] for line in _lead(summary, by)[:2])
    return (f'<section class="card result-block" aria-labelledby="compare-title">\n'
            f'<div class="section-head"><h2 id="compare-title">Сравнение методов на общих осях</h2>'
            f'<p>Я оцениваю все сохранённые разбиения 2023 года ({summary["partitions"]}) на одних и тех же признаках и графах, включая дорожный '
            f'граф и данные 2024 года при неизменных метках, и нулевую линию без обучения: тип МО по реестру. '
            f'Звёздочкой отмечены оси, на которых метод обучался, крестиком оси первой версии правила отбора каркаса; '
            f'в выборе лучших они не участвуют.</p></div>\n'
            f'<p class="control-help">{lead}</p>\n'
            f'{table(3)}\n{verdict(3)}\n{table(4)}\n{verdict(4)}\n'
            f'<p class="control-help">Временной Leiden на трёх seed даёт разное число устойчивых групп ({seeds}). '
            f'Из {len(t["base_group_best_jaccard"])} групп основного запуска {t["base_groups_found_in_all_seeds"]} находятся во всех трёх seed. '
            f'Расчёт: <code>scripts/method_comparison.py</code>, результаты в <code>reports/method-comparison/</code>.</p>\n</section>\n')
