"""Finite train2023 temporal architecture experiment;2024 is already explored."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from threadpoolctl import threadpool_limits
from sbercluster.features import transform_frozen
from sbercluster.io import sha256, write_json
from sbercluster.research import load_development
from sbercluster.selection_frontier import predict_frontier, silhouette_from_distances
from sbercluster.temporal_profiles import (fit_temporal_model, predict_temporal_features,
    transform_temporal_rows, predict_temporal_panel, batch_relative_context)

PROTOCOL = {
    'scope': 'already_explored2024_historical_sensitivity_not_holdout',
    'cohort': '2016_retrospectively_complete2023_2024_entities;boundaries_unverified',
    'baseline_models': ['historical_kmeans4', 'sw_constrained_huber75', 'sw_constrained_ordinary'],
    'new_arms': [{'id': 'calendar_only', 'reliability': 0.},
                 {'id': 'calendar_reliability_half', 'reliability': .5},
                 {'id': 'calendar_reliability_full', 'reliability': 1.}],
    'fit': '2023_only_scaler_calendar_offsets_within_covariance_kmeans4_seed1729_ninit20',
    'covariance_shrinkage': .2,
    'no_label_balance_penalty': True,
    'criteria_before_effects': {
        'deployment': 'artifact_roundtrip_exact;future_fit_mutation_invariant;entity_batch_order_invariant',
        'joint_candidate_screen': '2023_common_SW>=baseline-.005;CH>=.90baseline;2024_common_SW>=baseline-.005;CH>=.90baseline;min_yearly_fraction>=.03;ARI>=baseline',
        'stress': 'frozen_fullrank_transform_retains_global_and_local_shift;relative_context_can_remove_shared_shift_but_must_report_it',
        'promotion': 'none_automatically;all_candidates_reported;no_new_2024_adaptive_grid',
    },
    'stop': 'exactly_three_new_fits;descriptive_context_not_fit;no_grid_after_outcomes',
}


def metrics(x, labels, distances=None):
    labels = np.asarray(labels)
    occupied = len(np.unique(labels))
    counts = np.bincount(labels, minlength=4)
    result = {'occupied_k': occupied, 'cluster_sizes': counts.tolist(), 'min_cluster_size': int(counts.min())}
    if occupied >= 2 and occupied < len(x):
        result['SW'] = silhouette_from_distances(cdist(x, x) if distances is None else distances, labels)[0]
        result['CH'] = float(calinski_harabasz_score(x, labels))
    else:
        result['SW'], result['CH'] = None, None
    return result


def decomposition(first, second, rule):
    delta = second - first
    mean = delta.mean(axis=0)
    median = np.median(delta, axis=0)
    residual = delta - mean
    total = float(np.sum(delta ** 2))
    common = float(len(delta) * np.dot(mean, mean))
    individual = float(np.sum(residual ** 2))
    z0 = predict_frontier(first, rule)
    za = predict_frontier(second, rule)
    zc = predict_frontier(first + mean, rule)
    zi = predict_frontier(first + residual, rule)
    actual, common_set, individual_set = za != z0, zc != z0, zi != z0
    return {'mean_common_displacement': mean.tolist(), 'median_paired_displacement': median.tolist(),
            'total_squared_displacement': total, 'common_squared_displacement': common,
            'individual_squared_displacement': individual,
            'identity_absolute_error': abs(total - common - individual),
            'common_fraction': common / total if total else 0.,
            'actual_label_changes': int(actual.sum()), 'common_only_counterfactual_changes': int(common_set.sum()),
            'individual_only_counterfactual_changes': int(individual_set.sum()),
            'actual_and_common_only': int((actual & common_set).sum()),
            'actual_and_individual_only': int((actual & individual_set).sum()),
            'scope': 'geometric_mean_displacement_decomposition;counterfactuals_not_causal_attribution'}


def run(output):
    if output.exists():
        raise FileExistsError('Use new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / 'protocol.json', PROTOCOL)
    write_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'threads': 1, 'priority': 'BelowNormal'})
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8'))
    development, slices, train_monthly, ids, scaler, manifest = load_development(ROOT, cfg)
    panel = pd.read_csv(ROOT / 'data/processed/panel.csv', dtype={'entity_id': str, 'period': str, 'territory_id': str})
    test_monthly = np.stack([transform_frozen(panel[panel.period == f'2024-{m:02d}-01'].sort_values('entity_id'), scaler) for m in range(1, 13)])
    tensors = {2023: train_monthly, 2024: test_monthly}
    raw_profiles = {year: np.median(x, axis=0) for year, x in tensors.items()}
    baselines = {name: json.loads((ROOT / 'reports/model-frontier-2026-10-03/models' / (name + '.json')).read_text('utf-8'))
                 for name in PROTOCOL['baseline_models']}
    candidates = {}
    for spec in PROTOCOL['new_arms']:
        model = fit_temporal_model(development, cfg, reliability=spec['reliability'])
        candidates[spec['id']] = model
        write_json(output / 'models' / (spec['id'] + '.json'), model)
    labels = {}
    annual, periods, own_geometry = [], [], []
    yearly_transformed = {}
    for year, monthly in tensors.items():
        dates = [f'{year}-{m:02d}-01' for m in range(1, 13)]
        for name, model in candidates.items():
            yearly_transformed[year, name] = np.stack([transform_temporal_rows(x, [p] * len(x), model['calibration']) for p, x in zip(dates, monthly)])
        profile = raw_profiles[year]
        distances = cdist(profile, profile)
        for name, artifact in baselines.items():
            labels[year, name] = predict_frontier(profile, artifact['rule'])
        for name, model in candidates.items():
            adapted = np.median(yearly_transformed[year, name], axis=0)
            labels[year, name] = predict_temporal_features(adapted, model)
            prediction = predict_temporal_panel(panel, model, year)
            if prediction.entity_id.tolist() != ids or not np.array_equal(prediction.cluster, labels[year, name]):
                raise AssertionError('Exported inference differs')
            prediction.to_csv(output / f'predictions_{year}_{name}.csv', index=False)
            own_geometry.append({'year': year, 'id': name, 'method_geometry_metrics': metrics(adapted, labels[year, name]),
                                 'distance_extrapolation_count': int(prediction.distance_extrapolation.sum()),
                                 'new_entity_count': int(prediction.new_entity.sum())})
        for name in list(baselines) + list(candidates):
            annual.append({'year': year, 'id': name, 'common_metrics': metrics(profile, labels[year, name], distances)})
        for span, grouping in [('month', [[i] for i in range(12)]), ('quarter', [list(range(i, i + 3)) for i in range(0, 12, 3)])]:
            for indices in grouping:
                x = np.median(monthly[indices], axis=0)
                distances = cdist(x, x)
                for name, artifact in baselines.items():
                    z = predict_frontier(x, artifact['rule'])
                    periods.append({'year': year, 'span': span, 'months': [dates[i] for i in indices], 'id': name, 'common_metrics': metrics(x, z, distances)})
                for name, model in candidates.items():
                    adapted = np.median(yearly_transformed[year, name][indices], axis=0)
                    z = predict_temporal_features(adapted, model)
                    periods.append({'year': year, 'span': span, 'months': [dates[i] for i in indices], 'id': name, 'common_metrics': metrics(x, z, distances)})
    frozen = json.loads((ROOT / 'reports/model-frontier-2026-10-03/annual-2024-application.json').read_text('utf-8'))
    exact_baselines = {}
    for row in frozen['models']:
        exact_baselines[row['id']] = bool(np.array_equal(row['labels_2024'], labels[2024, row['id']]))
    if not all(exact_baselines.values()):
        raise AssertionError('Immutable2024 baselines differ')
    comparisons = []
    for name in list(baselines) + list(candidates):
        pair = [next(r for r in annual if r['year'] == y and r['id'] == name) for y in (2023, 2024)]
        ref = [next(r for r in annual if r['year'] == y and r['id'] == 'historical_kmeans4') for y in (2023, 2024)]
        gains = [a['common_metrics']['SW'] - b['common_metrics']['SW'] for a, b in zip(pair, ref)]
        ratios = [a['common_metrics']['CH'] / b['common_metrics']['CH'] for a, b in zip(pair, ref)]
        ari = float(adjusted_rand_score(labels[2023, name], labels[2024, name]))
        min_fraction = min(a['common_metrics']['min_cluster_size'] / len(ids) for a in pair)
        screen = min(gains) >= -.005 and min(ratios) >= .9 and min_fraction >= .03 and ari >= .5596999549170326
        comparisons.append({'id': name, 'yearly_SW_gains2023_2024': gains, 'yearly_CH_ratios2023_2024': ratios,
                            'ARI2023_to2024': ari, 'changed_saved_label_indices': int((labels[2023, name] != labels[2024, name]).sum()),
                            'min_yearly_fraction': min_fraction, 'prespecified_joint_screen_passed': screen,
                            'default_promotion': False})
    baseline_rule = baselines['historical_kmeans4']['rule']
    drift = {'annual': decomposition(raw_profiles[2023], raw_profiles[2024], baseline_rule),
             'same_calendar_month': [dict(month=i + 1, **decomposition(train_monthly[i], test_monthly[i], baseline_rule)) for i in range(12)]}
    contexts, shifts = [], []
    for i in range(12):
        context = batch_relative_context(test_monthly[i], test_monthly[i], np.median(train_monthly[i], axis=0))
        contexts.append(context['corrected_features'])
        shifts.append(context['common_location_drift'].tolist())
    corrected = np.median(np.stack(contexts), axis=0)
    corrected_labels = predict_frontier(corrected, baseline_rule)
    relative = {'scope': 'transductive_same_calendar_month_fixed2016_reference;diagnostic_not_inductive',
                'shared_drift_by_month': shifts, 'absolute_ARI': float(adjusted_rand_score(labels[2023, 'historical_kmeans4'], labels[2024, 'historical_kmeans4'])),
                'relative_ARI': float(adjusted_rand_score(labels[2023, 'historical_kmeans4'], corrected_labels)),
                'relative_changes': int((corrected_labels != labels[2023, 'historical_kmeans4']).sum()),
                'common_absolute_geometry_metrics': metrics(raw_profiles[2024], corrected_labels),
                'own_relative_geometry_metrics': metrics(corrected, corrected_labels)}
    shared = np.array([.3, -.15, .05, .2, -.1])
    stress = []
    for name, model in candidates.items():
        original = yearly_transformed[2023, name]
        transformed_shift = shared @ np.asarray(model['calibration']['metric'])
        synthetic = np.stack([transform_temporal_rows(x + shared, [f'2023-{i+1:02d}-01'] * len(x), model['calibration']) for i, x in enumerate(train_monthly)])
        expected = original + transformed_shift
        stress.append({'id': name, 'shared_shift_retained_norm': float(np.linalg.norm(transformed_shift)),
                       'exact_translation_error': float(np.max(np.abs(synthetic - expected))),
                       'shared_shift_label_changes': int((predict_temporal_features(np.median(synthetic, axis=0), model) != labels[2023, name]).sum()),
                       'metric_condition_number': float(np.linalg.cond(np.asarray(model['calibration']['metric'])))})
    shared_context = batch_relative_context(train_monthly[0] + shared, train_monthly[0] + shared, np.median(train_monthly[0], axis=0))
    stress.append({'id': 'relative_context', 'shared_shift_removed_error': float(np.max(np.abs(shared_context['corrected_features'] - train_monthly[0]))),
                   'shared_shift_reported_error': float(np.max(np.abs(shared_context['common_location_drift'] - shared)))})
    for filename, obj in [('annual.json', annual), ('periods.json', periods), ('own_geometry.json', own_geometry),
                          ('comparisons.json', comparisons), ('drift_decomposition.json', drift),
                          ('relative_context.json', relative), ('stress.json', stress)]:
        write_json(output / filename, obj)
    np.savez_compressed(output / 'labels.npz', **{f'{year}__{name}': z for (year, name), z in labels.items()}, relative2024=corrected_labels)
    sources = ['sbercluster/temporal_profiles.py', 'scripts/benchmark_temporal_profiles.py', 'sbercluster/features.py', 'sbercluster/selection_frontier.py']
    for relative_path in sources:
        destination = output / 'source' / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative_path).read_bytes())
    write_json(output / 'provenance.json', {'panel_sha256': manifest['panel_sha256'], 'protocol_sha256': sha256(output / 'protocol.json'),
        'baseline2024_labels_exact': exact_baselines, 'source_sha256': {p: sha256(ROOT / p) for p in sources},
        'python': platform.python_version(), 'numpy': np.__version__, 'cohort_size': len(ids), 'no2024fit': True})
    write_json(output / 'status.json', {'status': 'completed', 'pid': os.getpid(), 'elapsed_seconds': time.perf_counter() - started})
    write_json(output / 'manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output': str(output), 'comparisons': comparisons, 'drift_common_fraction': drift['annual']['common_fraction'],
                      'relative_ARI': relative['relative_ARI'], 'elapsed_seconds': time.perf_counter() - started}), flush=True)


def main():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot set BelowNormal priority')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            run(args.output.resolve())
    except BaseException as error:
        if args.output.exists():
            write_json(args.output / 'status.json', {'status': 'failed', 'error': str(error), 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
