"""Network-core typology: Leiden consensus on a profile plus co-movement multiplex, judged on the following year."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from scipy.stats import rankdata
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from .contest_graphs import read_road_graph
from .edge_rules import lagged_correlation, residual_series, similarity_knn
from .edge_study import load_panel, reference_partition
from .graph import knn_graph
from .io import CATEGORIES, TOTAL, sha256, write_json
from .metrics import calinski_harabasz_indices, network_indices, network_quality_candidates, s_dbw

FEATURE_ICVI = ('SW', 'CH', 'S_Dbw')
NETWORK_ICVI = ('AVI', 'AVU', 'MQ')
ORIENT = {'SW': 1, 'CH': 1, 'S_Dbw': -1, 'AVI': 1, 'AVU': -1, 'MQ': 1}


def normalize_layer(a):
    a = csr_matrix(a, dtype=float)
    total = float(a.sum())
    if total <= 0 or (a != a.T).nnz or np.any(a.data < 0):
        raise ValueError('Symmetric nonnegative layer with positive weight required')
    return a / total


def combine_layers(p, c, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must lie in [0, 1]')
    g = (alpha * normalize_layer(p) + (1 - alpha) * normalize_layer(c)).tocsr()
    g.setdiag(0)
    g.eliminate_zeros()
    return g


def canonical(labels):
    _, z = np.unique(np.asarray(labels), return_inverse=True)
    sizes = np.bincount(z)
    first = np.array([np.flatnonzero(z == c)[0] for c in range(len(sizes))])
    order = np.lexsort((first, -sizes))
    remap = np.empty(len(sizes), dtype=int)
    remap[order] = np.arange(len(sizes))
    return remap[z]


def leiden_resolution(a, gamma, seed, iterations=-1):
    import igraph as ig
    import leidenalg
    a = csr_matrix(a, dtype=float)
    upper = triu(a, k=1).tocoo()
    scale = upper.nnz / float(upper.data.sum())
    graph = ig.Graph(n=a.shape[0], edges=list(zip(upper.row.tolist(), upper.col.tolist())))
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, weights=(upper.data * scale).tolist(),
                                         resolution_parameter=gamma, n_iterations=iterations, seed=seed)
    return np.asarray(partition.membership)


def merge_small(a, labels, min_size):
    a = csr_matrix(a, dtype=float)
    z = canonical(labels)
    while True:
        sizes = np.bincount(z)
        if len(sizes) < 2 or sizes.min() >= min_size:
            return z
        small = int(np.lexsort((-np.arange(len(sizes)), sizes))[0])
        h = csr_matrix((np.ones(len(z)), (np.arange(len(z)), z)), shape=(len(z), len(sizes)))
        links = np.asarray((h[:, small].T @ a @ h).todense()).ravel()
        links[small] = -np.inf
        candidates = sizes.astype(float)
        candidates[small] = -np.inf
        target = int(np.argmax(links)) if links.max() > 0 else int(np.argmax(candidates))
        z = canonical(np.where(z == small, target, z))


def pairwise_ari(runs):
    n = len(runs)
    ari = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            ari[i, j] = ari[j, i] = adjusted_rand_score(runs[i], runs[j])
    return ari


def consensus(a, gamma, seeds, iterations, min_size):
    raw = [leiden_resolution(a, gamma, s, iterations) for s in seeds]
    runs = [merge_small(a, z, min_size) for z in raw]
    ari = pairwise_ari(runs)
    mean_other = (ari.sum(axis=1) - 1) / (len(runs) - 1)
    medoid = int(np.argmax(mean_other))
    upper = ari[np.triu_indices(len(runs), 1)]
    z = canonical(runs[medoid])
    raw_sizes = np.bincount(canonical(raw[medoid]))
    return z, {'seed_ari': float(upper.mean()), 'seed_ari_min': float(upper.min()), 'medoid_seed': int(seeds[medoid]),
               'medoid_mean_ari': float(mean_other[medoid]), 'raw_communities': int(len(np.unique(raw[medoid]))),
               'merged_nodes': int(raw_sizes[raw_sizes < min_size].sum()),
               'k_per_seed': [int(len(np.unique(r))) for r in runs]}


def build_layers(raw, features, months, k, max_lag):
    annual = np.median(features[months], axis=0)
    p, _ = knn_graph(annual, k)
    corr, _ = lagged_correlation(residual_series(raw[months]), max_lag)
    return annual, {'P': p, 'C': similarity_knn(corr, k)}, corr


def subgroup_layers(x, corr, members, k):
    p, _ = knn_graph(x[members], k)
    return {'P': p, 'C': similarity_knn(corr[np.ix_(members, members)], k)}


def evaluate(z, x, graphs, prefix):
    z = canonical(z)
    if z.max() < 1:
        return {f'{prefix}_{key}': None for key in (*FEATURE_ICVI, *[f'{g}_{m}' for g in graphs for m in NETWORK_ICVI])}
    out = {f'{prefix}_SW': float(silhouette_score(x, z)), f'{prefix}_CH': calinski_harabasz_indices(x, z)['CH']}
    try:
        out[f'{prefix}_S_Dbw'] = s_dbw(x, z)
    except ValueError:
        out[f'{prefix}_S_Dbw'] = None
    for name, a in graphs.items():
        ni = network_indices(a, z)
        out[f'{prefix}_{name}_AVI'] = ni['AVI']
        out[f'{prefix}_{name}_AVU'] = ni['AVU']
        out[f'{prefix}_{name}_MQ'] = network_quality_candidates(a, z)['NewmanQ']
    return out


def oriented(row, key, direction):
    value = row.get(key)
    return -np.inf if value is None or not np.isfinite(value) else direction * float(value)


def select_variant(rows, rule):
    admissible = [r for r in rows if r['seed_ari'] >= rule['min_seed_ari'] and r['min_size'] >= rule['min_size']
                  and rule['k_min'] <= r['k'] <= rule['k_max']]
    if not admissible:
        return None, {'admissible': [], 'front': [], 'rank_sums': {}}
    keys = [(p['key'], p['direction']) for p in rule['pareto']]

    def dominates(o, r):
        better = [oriented(o, k, d) >= oriented(r, k, d) for k, d in keys]
        strict = [oriented(o, k, d) > oriented(r, k, d) for k, d in keys]
        return all(better) and any(strict)

    front = [r for r in admissible if not any(dominates(o, r) for o in admissible)]
    sums = np.zeros(len(front))
    for item in rule['rank']:
        sums += item['weight'] * rankdata([oriented(r, item['key'], item['direction']) for r in front])
    order = sorted(range(len(front)), key=lambda i: (-sums[i], *[-oriented(front[i], t['key'], t['direction']) for t in rule['tiebreak']],
                                                    front[i]['variant']))
    return front[order[0]]['variant'], {'admissible': [r['variant'] for r in admissible], 'front': [r['variant'] for r in front],
                                        'rank_sums': {front[i]['variant']: float(sums[i]) for i in order}}


def design(n, covariates, fe, labels=None):
    cols = [np.ones(n), *covariates]
    for values in (fe, labels):
        if values is not None:
            levels, codes = np.unique(values, return_inverse=True)
            cols.extend((codes == c).astype(float) for c in range(1, len(levels)))
    return np.column_stack(cols)


def r_squared(y, x):
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    return 1 - float(residual @ residual) / float(np.sum((y - y.mean()) ** 2))


def r2_gain(y, covariates, fe, labels):
    base = r_squared(y, design(len(y), covariates, fe))
    full = r_squared(y, design(len(y), covariates, fe, labels))
    return base, full


def bootstrap_gain(y, covariates, fe, partitions, reps, seed):
    rng = np.random.default_rng(seed)
    regions = np.unique(fe)
    members = {r: np.flatnonzero(fe == r) for r in regions}
    draws = {name: [] for name in partitions}
    for _ in range(reps):
        picked = rng.choice(regions, size=len(regions), replace=True)
        index = np.concatenate([members[r] for r in picked])
        block = np.concatenate([np.full(len(members[r]), j) for j, r in enumerate(picked)])
        for name, z in partitions.items():
            base, full = r2_gain(y[index], [c[index] for c in covariates], block, z[index])
            draws[name].append(full - base)
    return {name: np.asarray(v) for name, v in draws.items()}


def external_validation(frame, partitions, targets, reps, seed):
    out = {'controls': 'region fixed effects + log(population)', 'bootstrap': f'{reps} region-cluster resamples'}
    names = list(partitions)
    for target in targets:
        values = frame[target['column']].to_numpy(dtype=float)
        keep = frame['population'].notna().to_numpy() & np.isfinite(values) & (values > 0 if target['transform'] == 'log' else True)
        y = np.log(values[keep]) if target['transform'] == 'log' else values[keep]
        fe = frame['region_code'].to_numpy()[keep]
        covariates = [np.log(frame['population'].to_numpy(dtype=float)[keep])]
        parts = {name: np.asarray(z)[keep] for name, z in partitions.items()}
        draws = bootstrap_gain(y, covariates, fe, parts, reps, seed)
        result = {'n': int(keep.sum())}
        for name, z in parts.items():
            base, full = r2_gain(y, covariates, fe, z)
            result[name] = {'R2_base': round(base, 6), 'R2_full': round(full, 6), 'delta_R2': round(full - base, 6),
                            'delta_R2_ci95': [round(float(q), 6) for q in np.quantile(draws[name], [0.025, 0.975])]}
        for other in names[1:]:
            diff = draws[names[0]] - draws[other]
            result[f'{names[0]}_minus_{other}'] = {'mean': round(float(diff.mean()), 6),
                                                   'ci95': [round(float(q), 6) for q in np.quantile(diff, [0.025, 0.975])]}
        out[target['name']] = result
    return out


def kmeans_runs(x, k, seeds):
    return [KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(x) for s in seeds]


def spectral(a, k, seed):
    return SpectralClustering(n_clusters=k, affinity='precomputed', random_state=seed, assign_labels='cluster_qr').fit_predict(a.toarray())


def variant_grid(l23, l24, x23, x24, e23, e24, grid, seeds, compare, log):
    rows, labels, reruns = [], {}, {}
    for alpha in grid['alphas']:
        g23 = combine_layers(l23['P'], l23['C'], alpha)
        g24 = combine_layers(l24['P'], l24['C'], alpha)
        for gamma in grid['gammas']:
            name = f'leiden_a{alpha:g}_g{gamma:g}'
            z, info = consensus(g23, gamma, seeds, grid['leiden_iterations'], grid['min_size'])
            z24, info24 = consensus(g24, gamma, seeds, grid['leiden_iterations'], grid['min_size'])
            labels[name], reruns[name] = z, z24
            rows.append({'variant': name, 'method': 'leiden_consensus', 'alpha': alpha, 'gamma': gamma, 'k': int(z.max() + 1),
                         'min_size': int(np.bincount(z).min()), **info, 'oot_rerun_ari': float(adjusted_rand_score(z, z24)),
                         'k_2024_rerun': int(z24.max() + 1), 'seed_ari_2024': info24['seed_ari'],
                         **{f'ARI_{key}': float(adjusted_rand_score(value, z)) for key, value in compare.items()},
                         **evaluate(z, x23, e23, 'in'), **evaluate(z, x24, e24, 'oot')})
            log({'variant': name, 'k': rows[-1]['k'], 'seed_ari': round(info['seed_ari'], 3)})
    return rows, labels, reruns


def compose_types(level1, splits):
    types = np.array([str(g) for g in level1], dtype=object)
    for g, sub in splits.items():
        members = np.flatnonzero(level1 == g)
        types[members] = [f'{g}.0'] * len(members) if sub is None else [f'{g}.{s + 1}' for s in canonical(sub)]
    return types


def central_members(x, members, count):
    d = np.sqrt(((x[members][:, None, :] - x[members][None, :, :]) ** 2).sum(axis=2))
    return members[np.argsort(d.mean(axis=1), kind='stable')[:count]]


GENITIVE = {'Здоровье': 'здоровья', 'Маркетплейсы': 'маркетплейсов', 'Общественное питание': 'общепита', 'Продовольствие': 'продовольствия',
            'Транспорт': 'транспорта'}


def describe(mask, shares, total, frame, reference):
    sub = frame[mask]
    return {'size': int(mask.sum()),
            'median_share': {c: round(float(np.median(shares[mask, i])), 4) for i, c in enumerate(CATEGORIES)},
            'median_total': round(float(np.median(total[mask])), 1),
            'municipal_type_share': {t: round(float(v), 4) for t, v in sub['municipal_district_type'].value_counts(normalize=True).items()},
            'top_regions': [{'region': r, 'count': int(c)} for r, c in sub['region_name'].value_counts().head(3).items()],
            'median_population': None if sub['population'].isna().all() else round(float(sub['population'].median()), 1),
            'median_wage_2023': round(float(sub['wage_2023'].median()), 1),
            'kmeans4_counts': {str(t): int(c) for t, c in zip(*np.unique(reference[mask], return_counts=True))}}


def profiles(z, raw, months, frame, reference):
    base = raw[months]
    shares = np.median(base[:, :, :len(CATEGORIES)] / base[:, :, [len(CATEGORIES)]], axis=0)
    total = np.median(base[:, :, len(CATEGORIES)], axis=0)
    return [{'group': g, **describe(z == g, shares, total, frame, reference)} for g in range(int(z.max()) + 1)]


def subtype_profiles(types, level1, x, raw, months, frame, reference, names, level_names, count, working_names):
    base = raw[months]
    shares = np.median(base[:, :, :len(CATEGORIES)] / base[:, :, [len(CATEGORIES)]], axis=0)
    total = np.median(base[:, :, len(CATEGORIES)], axis=0)
    out = []
    for label in sorted(set(types)):
        mask = types == label
        g = int(label.split('.')[0])
        group = level1 == g
        info = describe(mask, shares, total, frame, reference)
        relative = {c: float(np.median(shares[mask, i]) / np.median(shares[group, i]) - 1) for i, c in enumerate(CATEGORIES)}
        if '.' in label and not label.endswith('.0'):
            high, low = max(relative, key=relative.get), min(relative, key=relative.get)
            name = f'{level_names[str(g)]}: больше {GENITIVE[high]}, меньше {GENITIVE[low]}'
        else:
            name = level_names[str(g)]
        out.append({'type': label, 'group': g, 'working_name': working_names.get(label, name), 'auto_name': name, **info,
                    'share_vs_group_pp': {c: round(100 * (float(np.median(shares[mask, i])) - float(np.median(shares[group, i]))), 2)
                                          for i, c in enumerate(CATEGORIES)},
                    'share_vs_group_relative': {c: round(v, 4) for c, v in relative.items()},
                    'central_members': [names[i] for i in central_members(x, np.flatnonzero(mask), count)]})
    return out


def comparison_row(name, role, z, z24, x23, x24, e23, e24, reference, region, extra=None):
    z = canonical(z)
    return {'variant': name, 'role': role, 'k': int(z.max() + 1), 'min_size': int(np.bincount(z).min()), **(extra or {}),
            'oot_rerun_ari': float(adjusted_rand_score(z, z24)), 'ARI_kmeans4': float(adjusted_rand_score(reference, z)),
            'ARI_region': float(adjusted_rand_score(region, z)), **evaluate(z, x23, e23, 'in'), **evaluate(z, x24, e24, 'oot')}


def rounded(row):
    return {key: (round(v, 6) if isinstance(v, float) else v) for key, v in row.items()}


def run(cfg, root):
    root = Path(root)
    nc = cfg['network_core']
    l2 = nc['level2']
    out = root / nc['output']
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def log(event):
        print(json.dumps({**event, 'seconds': round(time.perf_counter() - started, 1)}, ensure_ascii=False), flush=True)

    ids, periods, monthly, raw, region = load_panel(cfg, root)
    fit = slice(periods.index(f"{nc['fit_year']}-01-01"), periods.index(f"{nc['fit_year']}-12-01") + 1)
    test = slice(periods.index(f"{nc['test_year']}-01-01"), periods.index(f"{nc['test_year']}-12-01") + 1)
    seeds = [cfg['seed'] + i for i in range(nc['seeds'])]
    x23, layers23, corr23 = build_layers(raw, monthly, fit, nc['k'], nc['max_lag'])
    x24, layers24, corr24 = build_layers(raw, monthly, test, nc['k'], nc['max_lag'])
    _, reference = reference_partition(root, ids, x23)
    road_path = root / nc['source_dir'] / 'hackathonlicence/connection.parquet'
    road, _, road_info = read_road_graph(road_path, ids, nc['k'])
    eval23, eval24 = {**layers23, 'R': road}, {**layers24, 'R': road}
    log({'phase': 'grid', 'n': len(ids)})
    rows, labels, reruns = variant_grid(layers23, layers24, x23, x24, eval23, eval24, nc, seeds, {'kmeans4': reference, 'region': region}, log)
    for row in rows:
        row['scope'] = 'all'
    selected, selection = select_variant(rows, nc['selection'])
    if selected is None:
        raise ValueError('No admissible network variant under the frozen selection rule')
    chosen = next(r for r in rows if r['variant'] == selected)
    k, alpha, gamma = chosen['k'], chosen['alpha'], chosen['gamma']
    log({'phase': 'comparators', 'selected': selected})
    g23 = combine_layers(layers23['P'], layers23['C'], alpha)
    g24 = combine_layers(layers24['P'], layers24['C'], alpha)
    comparators = {}
    km4_runs = kmeans_runs(x23, 4, seeds)
    km4_24 = KMeans(n_clusters=4, n_init=20, random_state=cfg['seed']).fit_predict(x24)
    comparators['kmeans4_reference'] = (reference, pairwise_ari(km4_runs), km4_24, {'method': 'frozen KMeans K=4 on the 2023 profile'})
    kk = KMeans(n_clusters=k, n_init=20, random_state=cfg['seed']).fit_predict(x23)
    kk24 = KMeans(n_clusters=k, n_init=20, random_state=cfg['seed']).fit_predict(x24)
    comparators[f'kmeans{k}'] = (kk, pairwise_ari(kmeans_runs(x23, k, seeds)), kk24, {'method': f'KMeans K={k} on the 2023 profile'})
    sp = [spectral(g23, k, s) for s in seeds[:5]]
    comparators[f'spectral{k}_a{alpha:g}'] = (canonical(sp[0]), pairwise_ari(sp), spectral(g24, k, cfg['seed']),
                                              {'method': f'spectral clustering on G_alpha={alpha:g}, K={k}, 5 seeds'})
    table = [{'variant': selected, 'role': 'selected', **{key: chosen[key] for key in chosen if key.startswith(('in_', 'oot_')) or key in
                                                         ('k', 'min_size', 'seed_ari', 'oot_rerun_ari', 'ARI_kmeans4', 'ARI_region')}}]
    for single in (f'leiden_a1_g{gamma:g}', f'leiden_a0_g{gamma:g}'):
        r = next(r for r in rows if r['variant'] == single)
        table.append({'variant': single, 'role': 'profile layer only' if 'a1_' in single else 'co-movement layer only',
                      **{key: r[key] for key in r if key.startswith(('in_', 'oot_')) or key in
                         ('k', 'min_size', 'seed_ari', 'oot_rerun_ari', 'ARI_kmeans4', 'ARI_region')}})
    for name, (z, ari, z24, meta) in comparators.items():
        z = canonical(z)
        labels[name] = z
        table.append(comparison_row(name, meta['method'], z, z24, x23, x24, eval23, eval24, reference, region,
                                    {'seed_ari': float(ari[np.triu_indices(len(ari), 1)].mean())}))
    level1 = labels[selected]
    level2, splits, split_reruns, sub_rows = {}, {}, {}, []
    for g in l2['groups']:
        members = np.flatnonzero(level1 == g)
        sub_ids = [ids[i] for i in members]
        s23, s24 = subgroup_layers(x23, corr23, members, l2['k']), subgroup_layers(x24, corr24, members, l2['k'])
        sub_road, _, _ = read_road_graph(road_path, sub_ids, l2['k'])
        log({'phase': 'level2', 'group': g, 'n': len(members)})
        grows, glabels, greruns = variant_grid(s23, s24, x23[members], x24[members], {**s23, 'R': sub_road}, {**s24, 'R': sub_road}, l2,
                                               [cfg['seed'] + i for i in range(l2['seeds'])],
                                               {'kmeans4': reference[members], 'region': region[members]}, log)
        for row in grows:
            row['scope'] = f'group{g}'
        sub_rows.extend(grows)
        picked, gselection = select_variant(grows, l2['selection'])
        robust, _ = select_variant([r for r in grows if r['oot_rerun_ari'] >= l2['sensitivity_min_oot_rerun_ari']], l2['selection'])
        robust_row = next((r for r in grows if r['variant'] == robust), None)
        sensitivity = {'min_oot_rerun_ari': l2['sensitivity_min_oot_rerun_ari'], 'variant': robust,
                       'k': None if robust_row is None else robust_row['k'], 'min_size': None if robust_row is None else robust_row['min_size'],
                       'seed_ari': None if robust_row is None else round(robust_row['seed_ari'], 6),
                       'oot_rerun_ari': None if robust_row is None else round(robust_row['oot_rerun_ari'], 6)}
        if picked is None:
            splits[g], split_reruns[g] = None, None
            level2[str(g)] = {'n': len(members), 'selection': gselection, 'selected': None, 'decision': 'kept whole: no admissible variant',
                              'post_hoc_sensitivity': sensitivity}
            continue
        row = next(r for r in grows if r['variant'] == picked)
        splits[g], split_reruns[g] = glabels[picked], greruns[picked]
        level2[str(g)] = {'n': len(members), 'selection': gselection, 'decision': 'split',
                          'selected': rounded({key: row[key] for key in row if key.startswith('oot_') or key in
                                               ('variant', 'alpha', 'gamma', 'k', 'min_size', 'seed_ari', 'seed_ari_min', 'oot_rerun_ari',
                                                'k_2024_rerun', 'ARI_kmeans4', 'ARI_region')}),
                          'sizes': np.bincount(glabels[picked]).tolist(), 'post_hoc_sensitivity': sensitivity}
    types = compose_types(level1, splits)
    types_rerun = compose_types(level1, {g: split_reruns[g] for g in splits})
    tz = np.unique(types, return_inverse=True)[1]
    tz24 = np.unique(types_rerun, return_inverse=True)[1]
    kt = int(tz.max() + 1)
    log({'phase': 'typology', 'types': kt})
    flat23 = KMeans(n_clusters=kt, n_init=20, random_state=cfg['seed']).fit_predict(x23)
    flat24 = KMeans(n_clusters=kt, n_init=20, random_state=cfg['seed']).fit_predict(x24)
    nested23, nested24 = level1.astype(str), level1.astype(str)
    for g, sub in splits.items():
        if sub is None:
            continue
        members = np.flatnonzero(level1 == g)
        ks = int(sub.max() + 1)
        nested23[members] = [f'{g}.{s}' for s in KMeans(n_clusters=ks, n_init=20, random_state=cfg['seed']).fit_predict(x23[members])]
        nested24[members] = [f'{g}.{s}' for s in KMeans(n_clusters=ks, n_init=20, random_state=cfg['seed']).fit_predict(x24[members])]
    nested23, nested24 = np.unique(nested23, return_inverse=True)[1], np.unique(nested24, return_inverse=True)[1]
    typology_table = [comparison_row('network_two_level', 'level 1 x network subtypes', tz, tz24, x23, x24, eval23, eval24, reference, region),
                      comparison_row(selected, 'level 1 only', level1, canonical(reruns[selected]), x23, x24, eval23, eval24, reference, region),
                      comparison_row('kmeans4_reference', 'frozen KMeans K=4', reference, km4_24, x23, x24, eval23, eval24, reference, region),
                      comparison_row(f'kmeans{kt}_flat', f'KMeans K={kt} on the 2023 profile', flat23, flat24, x23, x24, eval23, eval24, reference, region),
                      comparison_row('level1_kmeans_subtypes', 'level 1 x KMeans subtypes of the same sizes K', nested23, nested24, x23, x24, eval23,
                                     eval24, reference, region)]
    log({'phase': 'external'})
    ext_path = root / nc['external']
    frame = pd.read_csv(ext_path, dtype={'entity_id': str}).set_index('entity_id').reindex(ids)
    if frame['region_code'].isna().any():
        raise ValueError('External indicators do not cover every territory')
    frame['marketplace_share'] = np.median(raw[fit][:, :, CATEGORIES.index('Маркетплейсы')] / raw[fit][:, :, len(CATEGORIES)], axis=0)
    targets = l2['external_targets']
    external = external_validation(frame, {selected: level1, 'kmeans4_reference': reference}, targets[:2], nc['bootstrap'], cfg['seed'])
    typology_external = external_validation(frame, {'network_two_level': tz, selected: level1, 'kmeans4_reference': reference,
                                                    f'kmeans{kt}_flat': flat23, 'level1_kmeans_subtypes': nested23},
                                            targets, nc['bootstrap'], cfg['seed'])
    names = pd.read_csv(root / 'data/processed/panel.csv', usecols=['entity_id', 'municipal_district_name'], dtype=str).drop_duplicates(
        'entity_id').set_index('entity_id').reindex(ids)['municipal_district_name'].tolist()
    crosstab = pd.crosstab(pd.Series(level1, name='group'), pd.Series(reference, name='kmeans4'))
    type_crosstab = pd.crosstab(pd.Series(types, name='type'), pd.Series(reference, name='kmeans4'))
    write_json(out / 'profiles.json', {
        'variant': selected, 'year': nc['fit_year'], 'groups': profiles(level1, raw, fit, frame, reference),
        'crosstab_kmeans4': {str(g): {str(t): int(v) for t, v in row.items()} for g, row in crosstab.iterrows()},
        'subtypes': subtype_profiles(types, level1, x23, raw, fit, frame, reference, names, l2['level_names'], l2['central_members'],
                                     l2.get('working_names', {})),
        'type_crosstab_kmeans4': {str(g): {str(t): int(v) for t, v in row.items()} for g, row in type_crosstab.iterrows()},
        'kmeans4_names': {'0': 'сбалансированный', '1': 'продуктовый', '2': 'городской сервисный', '3': 'удалённый'}})
    candidates = pd.DataFrame(rows + sub_rows)
    candidates[['scope'] + [c for c in candidates.columns if c != 'scope']].round(6).to_csv(out / 'candidates.csv', index=False)
    pd.DataFrame({'entity_id': ids, 'group': level1, 'type': types, 'type_2024_rerun': types_rerun,
                  'group_2024_rerun': canonical(reruns[selected]), 'kmeans4_reference': reference,
                  **{name: labels[name] for name in comparators if name != 'kmeans4_reference'},
                  'leiden_profile_only': labels[f'leiden_a1_g{gamma:g}'], 'leiden_comovement_only': labels[f'leiden_a0_g{gamma:g}'],
                  f'kmeans{kt}_flat': canonical(flat23), 'level1_kmeans_subtypes': nested23}).to_csv(out / 'assignments.csv', index=False)
    write_json(out / 'summary.json', {
        'n': len(ids), 'fit_months': periods[fit], 'test_months': periods[test],
        'layers': {'P': 'euclid_profile kNN on the annual median log-ratio profile', 'C': 'lagged_correlation kNN on residual monthly series',
                   'R': 'road_distance kNN, external check only', 'k': nc['k'], 'max_lag': nc['max_lag'],
                   'edges': {f'{year}_{name}': int(triu(a, k=1).nnz) for year, layers in (('2023', layers23), ('2024', layers24)) for name, a in layers.items()} |
                            {'R': int(triu(road, k=1).nnz)}, 'road_bandwidth_km': road_info['bandwidth_km']},
        'grid': {'alphas': nc['alphas'], 'gammas': nc['gammas'], 'seeds': seeds, 'consensus': nc['consensus'], 'min_size': nc['min_size']},
        'selection_rule': nc['selection'], 'selection': selection,
        'selected': {'variant': selected, 'alpha': alpha, 'gamma': gamma, 'k': k, 'sizes': np.bincount(level1).tolist(),
                     'seed_ari': chosen['seed_ari'], 'seed_ari_min': chosen['seed_ari_min'], 'oot_rerun_ari': chosen['oot_rerun_ari'],
                     'k_2024_rerun': chosen['k_2024_rerun'], 'ARI_kmeans4': chosen['ARI_kmeans4'], 'ARI_region': chosen['ARI_region']},
        'comparison': [rounded(row) for row in table],
        'external_validation': external,
        'level2': {'rule': l2['selection'], 'groups': level2},
        'typology': {'types': {label: int(count) for label, count in zip(*np.unique(types, return_counts=True))},
                     'oot_rerun_note': 'type_2024_rerun keeps the 2023 level-1 groups and reruns the selected subtype variant on 2024 graphs inside them',
                     'comparison': [rounded(row) for row in typology_table], 'external_validation': typology_external},
        'icvi_convention': 'SW, CH, S_Dbw on the annual median log-ratio profile; AVI, AVU, MQ (Newman Q) on binary P, C and R graphs'})
    inputs = [root / 'data/processed/panel.csv', ext_path, road_path,
              root / 'reports/experiments/2026-09-23-v2/validation/frozen_prototypes.json',
              root / 'reports/experiments/2026-09-23-v2/validation/reference_assignments.csv']
    write_json(out / 'provenance.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                                         'inputs': {str(p.relative_to(root)): sha256(p) for p in inputs},
                                         'implementation': {p.name: sha256(p) for p in sorted((root / 'sbercluster').glob('*.py'))},
                                         'script_sha256': sha256(root / 'scripts/network_core.py'),
                                         'seconds': time.perf_counter() - started})
    return out
