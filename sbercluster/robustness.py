from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .graph import knn_graph
from .io import CATEGORIES, TOTAL
from .metrics import all_metrics


def kmeans(x, k, seed, n_init):
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(x)
    return model.labels_.astype(int), model.cluster_centers_, float(model.inertia_)


def assign(x, centers):
    return cdist(x, centers).argmin(axis=1)


def match_groups(reference, labels, k):
    reference, labels = np.asarray(reference, dtype=int), np.asarray(labels, dtype=int)
    if reference.shape != labels.shape or reference.min() < 0 or labels.min() < 0 or max(reference.max(), labels.max()) >= k:
        raise ValueError('Invalid aligned labels')
    table = np.zeros((k, k), dtype=int)
    np.add.at(table, (reference, labels), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = np.empty(k, dtype=int)
    mapping[cols] = rows
    return mapping, table


def linear_trend(values, start=0):
    values = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(np.arange(start, start + len(values)), values, 1)
    return float(slope), float(intercept)


def month_indices(periods, window):
    idx = [i for i, p in enumerate(periods) if window[0] <= p <= window[1]]
    if not idx or idx != list(range(idx[0], idx[-1] + 1)):
        raise ValueError(f'Window {window} is empty or not contiguous')
    return idx


def market_trajectory(panel, periods, category, trend_window):
    ratio = panel[category] / panel[TOTAL] * 100
    medians = ratio.groupby(panel.period).median().reindex(periods).to_numpy(dtype=float)
    if not np.isfinite(medians).all() or len(periods) != 24:
        raise ValueError('Expected 24 complete months')
    trend_idx = month_indices(periods, trend_window)
    slope, intercept = linear_trend(medians[trend_idx], trend_idx[0])
    predicted = intercept + slope * np.arange(12, 24)
    trends = {}
    for name, idx in {'2023': range(12), '2024': range(12, 24), 'h1_2023': range(6), 'h2_2023': range(6, 12),
                      'h1_2024': range(12, 18), 'h2_2024': range(18, 24)}.items():
        trends[name] = {'slope_pp_per_month': linear_trend(medians[list(idx)], idx[0])[0],
                        'start': float(medians[idx[0]]), 'end': float(medians[idx[-1]])}
    return {'periods': list(periods), 'median_share_pct': medians.tolist(), 'trends': trends,
            'jul_to_nov_pp': {'2023': float(medians[10] - medians[6]), '2024': float(medians[22] - medians[18])},
            'same_month_yoy_pp': (medians[12:] - medians[:12]).tolist(),
            'extrapolation': {'fit_window': list(trend_window), 'slope_pp_per_month': slope,
                              'predicted_2024_pct': predicted.tolist(), 'actual_2024_pct': medians[12:].tolist(),
                              'predicted_2024_mean': float(predicted.mean()), 'actual_2024_mean': float(medians[12:].mean()),
                              'predicted_dec_2024': float(predicted[-1]), 'actual_dec_2024': float(medians[23]),
                              'annual_2023_mean': float(medians[:12].mean())}}


def national_shifts(monthly, trend_idx):
    medians = np.median(monthly, axis=1)
    actual = medians[12:] - medians[:12]
    t = np.asarray(trend_idx)
    fits = [np.polyfit(t, medians[t, c], 1) for c in range(medians.shape[1])]
    predicted = np.column_stack([np.polyval(f, np.arange(12, 24)) for f in fits])
    return actual, predicted - medians[:12], medians


def group_medians(values, labels, k):
    return [np.median(values[labels == g], axis=0).tolist() if (labels == g).any() else None for g in range(k)]


def semantic_check(reference_centers, centers):
    corr = [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(reference_centers, centers)]
    extremes = all((np.argmax(reference_centers, axis=0) == np.argmax(centers, axis=0)).tolist()
                   + (np.argmin(reference_centers, axis=0) == np.argmin(centers, axis=0)).tolist())
    return {'center_correlation': corr, 'max_abs_center_difference': float(np.abs(reference_centers - centers).max()),
            'extreme_groups_per_feature_preserved': bool(extremes)}


def base_robustness(monthly, ratios, reference, reference_centers, bases, periods, trend_idx, remote, seed, n_init, k):
    actual_shift, trend_shift, _ = national_shifts(monthly, trend_idx)
    y24 = np.median(monthly[12:], axis=0)
    relative24 = np.median(monthly[12:] - actual_shift[:, None, :], axis=0)
    trend24 = np.median(monthly[12:] - trend_shift[:, None, :], axis=0)
    out = {}
    for name, window in bases.items():
        idx = month_indices(periods, window)
        profile = np.median(monthly[idx], axis=0)
        raw_labels, raw_centers, inertia = kmeans(profile, k, seed, n_init)
        mapping, table = match_groups(reference, raw_labels, k)
        labels = mapping[raw_labels]
        centers = np.empty_like(raw_centers)
        centers[mapping] = raw_centers
        base_ratio = np.median(ratios[idx], axis=0)
        aligned24 = y24 - (np.median(y24, axis=0) - np.median(profile, axis=0))
        variants = {'raw': y24, 'monthly_shift_removed': relative24, 'base_median_aligned': aligned24,
                    'h2_2023_trend_removed': trend24}
        transitions = {}
        for variant, x24 in variants.items():
            after = assign(x24, centers)
            matrix = np.zeros((k, k), dtype=int)
            np.add.at(matrix, (labels, after), 1)
            transitions[variant] = {'changed': int((after != labels).sum()), 'changed_share': float((after != labels).mean()),
                                    'ari_with_base': float(adjusted_rand_score(labels, after)),
                                    'sizes_2024': np.bincount(after, minlength=k).tolist(),
                                    'remote_2024': int((after == remote).sum()), 'transition_matrix': matrix.tolist()}
        out[name] = {'window': list(window), 'months': len(idx), 'inertia': inertia,
                     'ari_with_reference': float(adjusted_rand_score(reference, labels)),
                     'agreement_after_matching': float((labels == reference).mean()),
                     'contingency_reference_by_raw_label': table.tolist(), 'mapping_raw_to_reference': mapping.tolist(),
                     'sizes': np.bincount(labels, minlength=k).tolist(), 'remote_size': int((labels == remote).sum()),
                     'centers': centers.tolist(), 'feature_medians': group_medians(profile, labels, k),
                     'ratio_medians_pct': group_medians(base_ratio, labels, k),
                     'semantics': semantic_check(reference_centers, centers),
                     'monthly_remote_counts': [int((assign(m, centers) == remote).sum()) for m in monthly],
                     'transitions': transitions}
    return out, {'actual_shift': actual_shift.tolist(), 'h2_2023_trend_shift': trend_shift.tolist(),
                 'annual_actual_shift': np.median(actual_shift, axis=0).tolist(),
                 'annual_trend_shift': np.median(trend_shift, axis=0).tolist()}


def gap_statistic(x, ks, references, box, seed, n_init):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    mean = x.mean(axis=0)
    if box == 'pca':
        _, _, basis = np.linalg.svd(x - mean, full_matrices=False)
    elif box == 'uniform':
        basis = np.eye(x.shape[1])
    else:
        raise ValueError('gap box must be pca or uniform')
    z = (x - mean) @ basis.T
    low, high = z.min(axis=0), z.max(axis=0)

    def log_w(data, k):
        if k == 1:
            return float(np.log(((data - data.mean(axis=0)) ** 2).sum()))
        return float(np.log(kmeans(data, k, seed, n_init)[2]))

    observed = np.array([log_w(x, k) for k in ks])
    reference = np.array([[log_w(sample, k) for k in ks]
                          for sample in (rng.uniform(low, high, size=z.shape) @ basis + mean for _ in range(references))])
    gap = reference.mean(axis=0) - observed
    s = reference.std(axis=0) * np.sqrt(1 + 1 / references)
    return {'K': list(ks), 'log_W': observed.tolist(), 'gap': gap.tolist(), 's': s.tolist(),
            'one_se_K': one_se(ks, gap, s), 'max_gap_K': int(ks[int(np.argmax(gap))]), 'box': box, 'references': references}


def one_se(ks, gap, s):
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - s[i + 1]:
            return int(ks[i])
    return None


def interior_optima(values, direction):
    sign = 1 if direction == 'max' else -1
    v = [None if a is None else sign * a for a in values]
    return [i for i in range(1, len(v) - 1)
            if None not in v[i - 1:i + 2] and v[i] > v[i - 1] and v[i] > v[i + 1]]


def select_k(rows, rule):
    ks = [r['K'] for r in rows]
    votes = {k: [] for k in ks}
    for name, direction in rule['indices'].items():
        for i in interior_optima([r.get(name) for r in rows], direction):
            votes[ks[i]].append(name)
    admissible = [r['K'] for r in rows if r['bootstrap_ari_median'] >= rule['min_bootstrap_ari']
                  and r['seed_ari_median'] >= rule['min_seed_ari'] and r['min_share'] >= rule['min_share']]
    family = rule.get('same_index', {})
    vote_count = {k: len({family.get(name, name) for name in names}) for k, names in votes.items()}
    eligible = [k for k in admissible if vote_count[k] >= rule['min_votes']]
    if eligible:
        chosen, reason = max(eligible), 'largest admissible K that is an interior local optimum of at least min_votes distinct indices'
    elif admissible:
        chosen = max((r for r in rows if r['K'] in admissible), key=lambda r: r['SW'])['K']
        reason = 'no admissible K with enough votes; admissible K with maximal SW'
    else:
        chosen, reason = None, 'no admissible K'
    return {'chosen_K': chosen, 'reason': reason, 'admissible': admissible, 'eligible': eligible,
            'votes': {str(k): v for k, v in votes.items()}, 'vote_count': {str(k): v for k, v in vote_count.items()}}


def nesting(coarse, fine):
    coarse, fine = np.asarray(coarse, dtype=int), np.asarray(fine, dtype=int)
    table = np.zeros((fine.max() + 1, coarse.max() + 1), dtype=int)
    np.add.at(table, (fine, coarse), 1)
    return {'fine_by_coarse': table.tolist(), 'majority_share_per_fine': (table.max(axis=1) / table.sum(axis=1)).tolist(),
            'purity': float(table.max(axis=1).sum() / table.sum())}


def k_selection(x, reference, cfg, seed, n_init):
    graph, _ = knn_graph(x, cfg['graph_k'])
    rng = np.random.default_rng(seed)
    n = len(x)
    subsets = [np.sort(rng.choice(n, size=int(round(cfg['bootstrap_fraction'] * n)), replace=False))
               for _ in range(cfg['bootstrap_runs'])]
    rows, labels_by_k = [], {}
    for k in cfg['k_grid']:
        labels, _, inertia = kmeans(x, k, seed, n_init)
        labels_by_k[k] = labels
        m = all_metrics(x, labels, graph)
        boot = [adjusted_rand_score(labels[s], kmeans(x[s], k, seed + b + 1, n_init)[0]) for b, s in enumerate(subsets)]
        seeds = [adjusted_rand_score(labels, kmeans(x, k, s, n_init)[0]) for s in cfg['seeds'] if s != seed]
        sizes = np.bincount(labels, minlength=k)
        rows.append({'K': k, 'SW': m['SW'], 'CH': m['CH'], 'S_Dbw': m['S_Dbw'], 'S_Dbw_paper': m['S_Dbw_paper'],
                     'S_Dbw_status': m['S_Dbw_status'], 'MQ': m['MQ'], 'AVI': m['AVI'], 'AVU': m['AVU'], 'TurboMQ': m['TurboMQ'], 'inertia': inertia,
                     'sizes': sorted(sizes.tolist(), reverse=True), 'min_cluster_size': int(sizes.min()),
                     'min_share': float(sizes.min() / n),
                     'bootstrap_ari_median': float(np.median(boot)), 'bootstrap_ari_q10': float(np.quantile(boot, .1)),
                     'seed_ari_median': float(np.median(seeds)), 'seed_ari_min': float(np.min(seeds))})
    chain = {f'{a}->{b}': nesting(labels_by_k[a], labels_by_k[b])
             for a, b in zip(cfg['k_grid'][:-1], cfg['k_grid'][1:])}
    reference = np.asarray(reference)
    k2_k4 = nesting(labels_by_k[2], reference) if 2 in labels_by_k else None
    refit = float(adjusted_rand_score(reference, labels_by_k[4])) if 4 in labels_by_k else None
    gap = gap_statistic(x, [1] + list(cfg['k_grid']), cfg['gap_references'], cfg['gap_box'], seed, cfg['gap_n_init'])
    return {'rows': rows, 'gap': gap, 'selection': select_k(rows, cfg['rule']), 'nesting_chain': chain,
            'nesting_k2_reference_k4': k2_k4, 'reference_refit_ari_k4': refit}, labels_by_k


def ratio_tensor(panel, periods, ids):
    frame = panel.set_index(['period', 'entity_id'])
    out = []
    for p in periods:
        f = frame.loc[p].reindex(ids)
        out.append((f[CATEGORIES].to_numpy(dtype=float) / f[TOTAL].to_numpy(dtype=float)[:, None]) * 100)
    return np.stack(out)
