"""Frozen, retrospective peer validation. Future outcomes cannot select peers."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from .features import make_slices
from .graph import knn_graph
from .io import CATEGORIES, TOTAL, write_json, sha256
from .research import annual_profile, fit_candidate, local_scale_graph, score


def transform_frozen(frame, scaler):
    values = frame[CATEGORIES].to_numpy(float)
    total = frame[TOTAL].to_numpy(float)
    if not np.isfinite(values).all() or not np.isfinite(total).all() or (values <= 0).any() or (total <= 0).any():
        raise ValueError('Positive finite observations required')
    return ((np.log(values / total[:, None]) - np.asarray(scaler['ratio_center']))
            / np.asarray(scaler['ratio_iqr']) / np.sqrt(len(CATEGORIES)))


def nearest_indices(distance, k=15, allowed=None):
    distance = np.array(distance, dtype=float, copy=True)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1] or k < 1:
        raise ValueError('Square distance matrix required')
    if allowed is not None:
        allowed = np.asarray(allowed, bool)
        if allowed.shape != distance.shape:
            raise ValueError('Invalid candidate mask')
        distance[~allowed] = np.inf
    np.fill_diagonal(distance, np.inf)
    result = []
    for i, row in enumerate(distance):
        valid = np.flatnonzero(np.isfinite(row))
        if len(valid) == 0:
            raise ValueError(f'No peers for row {i}')
        result.append(valid[np.argsort(row[valid], kind='stable')[:k]])
    return result


def geographic_distances(lat, lon):
    lat = np.radians(np.asarray(lat, float)); lon = np.radians(np.asarray(lon, float))
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError('Coordinates must be finite')
    value = np.sin((lat[:, None] - lat[None, :]) / 2)**2
    value += np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def consensus_peers(monthly, k=15, candidate_k=60):
    """Rank a frozen annual candidate set by recurring directed monthly edges."""
    x = annual_profile(monthly)
    n = len(x)
    if not 1 <= k <= candidate_k < n:
        raise ValueError('Invalid consensus peer count')
    annual_distance = cdist(x, x)
    candidates = nearest_indices(annual_distance, candidate_k)
    votes = np.zeros((n, n), dtype=np.uint8)
    for month in monthly:
        month_neighbors = nearest_indices(cdist(month, month), k)
        for i, neighbors in enumerate(month_neighbors):
            votes[i, neighbors] += 1
    selected = []
    frequencies = []
    for i, neighbors in enumerate(candidates):
        order = np.lexsort((annual_distance[i, neighbors], -votes[i, neighbors].astype(int)))
        chosen = neighbors[order[:k]]
        selected.append(chosen)
        frequencies.append((votes[i, chosen] / len(monthly)).tolist())
    return selected, frequencies


def regional_bootstrap_difference(error, baseline, regions, seed=1729, repetitions=1000):
    """Resample entire regions; positive difference means smaller discrepancy."""
    error = np.asarray(error, float); baseline = np.asarray(baseline, float)
    regions = np.asarray(regions)
    if error.ndim != 1 or error.shape != baseline.shape or len(regions) != len(error):
        raise ValueError('Aligned errors and regions required')
    delta = baseline - error
    if not np.isfinite(delta).all():
        raise ValueError('Finite errors required')
    unique, encoded = np.unique(regions, return_inverse=True)
    sums = np.bincount(encoded, weights=delta)
    counts = np.bincount(encoded)
    if len(unique) < 2:
        raise ValueError('At least two regions required')
    rng = np.random.default_rng(seed)
    draw = rng.integers(len(unique), size=(repetitions, len(unique)))
    estimates = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {'mean_improvement': float(delta.mean()),
            'ci_95': np.quantile(estimates, [.025,.975]).tolist(),
            'n_territories': len(delta), 'n_regions': len(unique),
            'repetitions': repetitions, 'resampling_unit': 'region',
            'caveat': 'conditional descriptive bootstrap; dependence between regions not removed'}


def eta_squared(values, labels):
    values = np.asarray(values, float); labels = np.asarray(labels)
    total = ((values - values.mean())**2).sum()
    if total == 0:
        return 0.
    explained = sum(np.sum(labels == label) * (values[labels == label].mean() - values.mean())**2
                    for label in np.unique(labels))
    return float(explained / total)


def external_market_access(root, ids, labels, regions):
    from .canonical import read_parquet
    path = root / 'artifacts/sources/acquisition/hackathonlicence/market_access.parquet'
    frame = read_parquet(path)
    numeric_candidates = [c for c in frame.columns if c not in ('territory_id','year','date','__index_level_0__')]
    if len(numeric_candidates) != 1:
        return {'status': 'schema_review_required', 'columns': frame.columns.tolist()}
    column = numeric_candidates[0]
    if frame.territory_id.duplicated().any():
        raise ValueError('Market access has repeated territory IDs')
    mapping = dict(zip('tid_' + frame.territory_id.astype(str), frame[column]))
    values = np.array([mapping.get(entity, np.nan) for entity in ids], dtype=float)
    valid = np.isfinite(values)
    values, labels, regions = values[valid], labels[valid], np.asarray(regions)[valid]
    residual = values.copy()
    for region in np.unique(regions):
        mask = regions == region
        residual[mask] -= residual[mask].mean()
    observed = eta_squared(residual, labels)
    rng = np.random.default_rng(1729)
    null = []
    for _ in range(199):
        permuted = labels.copy()
        for region in np.unique(regions):
            indices = np.flatnonzero(regions == region)
            permuted[indices] = labels[rng.permutation(indices)]
        null.append(eta_squared(residual, permuted))
    return {'status': 'computed', 'source_sha256': sha256(path), 'column': column,
        'source_year': 2024, 'used_for_training_or_selection': False, 'n': int(valid.sum()),
        'eta_squared_raw': eta_squared(values, labels),
        'eta_squared_after_region_mean_removal': observed,
        'within_region_permutation_p': float((1 + np.count_nonzero(np.asarray(null) >= observed)) / 200),
        'permutations': 199,
        'cluster_summaries': [{'cluster': int(c), 'n': int((labels == c).sum()),
            'median': float(np.median(values[labels == c])),
            'q25': float(np.quantile(values[labels == c], .25)),
            'q75': float(np.quantile(values[labels == c], .75))} for c in np.unique(labels)],
        'interpretation': 'withheld attribute association, not causal or industrial-type validation'}


def validate(cfg, root, monthly, ids, scaler, out):
    settings = cfg['research']
    freeze_path = root / settings['freeze_file']
    if sha256(freeze_path) != settings['freeze_sha256']:
        raise ValueError('The frozen selection file changed')
    freeze = json.loads(freeze_path.read_text('utf-8'))
    spec = freeze['selected_candidate']
    if freeze['selection_data_end'] != '2023-12-01' or freeze['holdout_evaluated_before_freeze']:
        raise ValueError('Invalid development/confirmation separation')
    reference = annual_profile(monthly)
    common, _ = knn_graph(reference, 15)
    fitting_graph = local_scale_graph(reference,15,7)[0] if spec['method'].endswith('_local') else common
    labels, fit_info = fit_candidate(reference, fitting_graph, spec, cfg['seed'], cfg)
    centers = np.stack([reference[labels == c].mean(axis=0) for c in range(labels.max()+1)])
    nearest_reference = cdist(reference, centers).argmin(axis=1)
    write_json(out / 'frozen_prototypes.json', {'candidate': spec, 'centers': centers.tolist(),
        'ids': ids, 'reference_labels': labels.tolist(), 'scaler': scaler,
        'prototype_fidelity': float(np.mean(nearest_reference == labels)),
        'assignment_rule': 'nearest Euclidean mean prototype; not native spectral/Leiden inference',
        'freeze_sha256': sha256(freeze_path), 'fit_info': fit_info})

    panel = pd.read_csv(root / 'data/processed/panel.csv', dtype={'entity_id':str,'period':str})
    future_frames = []
    monthly_rows = []
    for period, frame in panel.groupby('period', sort=True):
        frame = frame.set_index('entity_id').loc[ids].reset_index()
        x = transform_frozen(frame, scaler)
        distance = cdist(x, centers)
        order = np.argsort(distance, axis=1)
        assigned = order[:,0]
        nearest, second = distance[np.arange(len(ids)),order[:,0]], distance[np.arange(len(ids)),order[:,1]]
        margin = (second - nearest) / np.maximum(second, np.finfo(float).eps)
        monthly_rows.extend({'entity_id':entity,'period':period,'cluster':int(c),'prototype_margin':float(m)}
                            for entity,c,m in zip(ids,assigned,margin))
        if period >= '2024-01-01':
            future_frames.append(x)
    if len(future_frames) != 12:
        raise ValueError('Expected all 12 confirmation months')
    future = annual_profile(np.stack(future_frames))
    future_labels = cdist(future, centers).argmin(axis=1)
    future_graph, _ = knn_graph(future,15)
    write_json(out / 'confirmation_metrics.json', {'candidate':spec,
        'frozen_prototype_scores_2024': score(future,future_labels,future_graph),
        'ARI_reference_to_2024': float(adjusted_rand_score(labels,future_labels)),
        'changed_label_fraction': float(np.mean(labels != future_labels)),
        'boundary_comparability': '2024 administrative transformations not independently cleared',
        'scope': 'fixed 2023 prototypes on annual 2024 profile; retrospective complete cohort'})
    pd.DataFrame(monthly_rows).to_csv(out/'monthly_assignments.csv',index=False)
    pd.DataFrame({'entity_id':ids,'cluster':labels,'cluster_2024':future_labels}).to_csv(out/'reference_assignments.csv',index=False)

    metadata = panel[panel.period=='2023-12-01'].set_index('entity_id').loc[ids]
    regions = metadata.region_code.to_numpy()
    level_2023 = panel[panel.period<'2024-01-01'].groupby('entity_id')[TOTAL].mean().loc[ids].to_numpy()
    level_2024 = panel[panel.period>='2024-01-01'].groupby('entity_id')[TOTAL].mean().loc[ids].to_numpy()
    growth = 100*(level_2024/level_2023-1)
    lat = metadata.municipal_district_center_lat.to_numpy(float)
    lon = metadata.municipal_district_center_lon.to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    # All peer methods use the same eligible set, including region and cluster leave-one-out.
    for _ in range(len(ids)):
        region_counts = Counter(regions[valid]); cluster_counts=Counter(labels[valid])
        next_valid = valid & np.array([region_counts[r]>=2 and cluster_counts[c]>=2 for r,c in zip(regions,labels)])
        if np.array_equal(valid,next_valid): break
        valid=next_valid
    selected_ids=np.array(ids)[valid]; x=reference[valid]; y=future[valid]
    target=growth[valid]; selected_regions=regions[valid]; selected_labels=labels[valid]
    distance=cdist(x,x)
    geo=geographic_distances(lat[valid],lon[valid])
    peers={
        'profile_15':nearest_indices(distance,15),
        'level_15':nearest_indices(cdist(np.log(level_2023[valid,None]),np.log(level_2023[valid,None])),15),
        'geographic_15':nearest_indices(geo,15),
        'cluster_profile_15':nearest_indices(distance,15,selected_labels[:,None]==selected_labels[None,:]),
        'region_all':nearest_indices(np.zeros_like(distance),len(x)-1,selected_regions[:,None]==selected_regions[None,:])}
    rng = np.random.default_rng(1729)
    peers['random_15'] = [rng.choice(np.delete(np.arange(len(x)), i), size=15, replace=False) for i in range(len(x))]
    peers['temporal_consensus_15'], frequencies=consensus_peers(monthly[:,valid,:],15,60)
    errors={}; rows=[]
    for name,neighbors in peers.items():
        growth_reference=np.array([np.median(target[n]) for n in neighbors])
        profile_reference=np.stack([np.median(y[n],axis=0) for n in neighbors])
        errors[name]={'growth':np.abs(target-growth_reference),
                      'profile':np.linalg.norm(y-profile_reference,axis=1)}
        rows.append({'method':name,'n':len(x),
            'growth_absolute_discrepancy_pp':float(errors[name]['growth'].mean()),
            'profile_euclidean_discrepancy':float(errors[name]['profile'].mean()),
            'min_peers':min(map(len,neighbors)),'max_peers':max(map(len,neighbors))})
    comparisons=[]
    for name in peers:
        for baseline in ('profile_15','region_all'):
            if name==baseline: continue
            for target_name in ('growth','profile'):
                comparisons.append({'method':name,'baseline':baseline,'target':target_name,
                    **regional_bootstrap_difference(errors[name][target_name],errors[baseline][target_name],selected_regions)})
    write_json(out/'peer_validation.json',{'n_common':len(x),'n_excluded':len(ids)-len(x),
        'results':rows,'paired_region_bootstrap':comparisons,
        'growth_definition':'percentage change of mean monthly total expenditure 2024 vs 2023',
        'profile_definition':'Euclidean discrepancy of median scaled log-relative-intensity profiles in 2024',
        'scope':'peer concordance using observed peer outcomes; NOT an as-of forecast',
        'selection':'all neighbor sets use only 2023 values and frozen rules',
        'cohort':'same retrospective complete cohort and available coordinates for all methods',
        'interval_note':'exploratory intervals across prespecified contrasts, no multiplicity correction'})
    neighbor_rows=[]
    for i,entity in enumerate(selected_ids):
        for rank,j in enumerate(peers['temporal_consensus_15'][i]):
            neighbor_rows.append({'entity_id':entity,'neighbor_id':selected_ids[j], 'rank':rank+1,
                'month_edge_frequency':frequencies[i][rank],
                'profile_distance_2023':float(distance[i,j]), 'distance_km':float(geo[i,j])})
    pd.DataFrame(neighbor_rows).to_csv(out/'consensus_neighbors.csv',index=False)
    write_json(out/'external_market_access.json',external_market_access(root,ids,labels,regions))
    write_json(out/'confounding.json',{'region_adjusted_mutual_information':float(adjusted_mutual_info_score(regions,labels)),
                                      'level_log_eta_squared':eta_squared(np.log(level_2023),labels)})
    profile_rows=[]
    for c in np.unique(labels):
        group=labels==c
        for index,category in enumerate(CATEGORIES):
            profile_rows.append({'cluster':int(c),'n':int(group.sum()),'category':category,
                'median_standardized_log_ratio':float(np.median(reference[group,index])),
                'q25':float(np.quantile(reference[group,index],.25)),
                'q75':float(np.quantile(reference[group,index],.75))})
    pd.DataFrame(profile_rows).to_csv(out/'cluster_profiles.csv',index=False)
