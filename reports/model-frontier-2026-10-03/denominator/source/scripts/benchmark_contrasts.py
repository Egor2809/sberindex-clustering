"""Denominator sensitivity and exactly invariant four-contrast alternatives.

The geometry changes deliberately; common original-coordinate and intrinsic
contrast scores are both reported. There is no metric-only promotion rule.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster.graph import knn_graph
from sbercluster.io import CATEGORIES, TOTAL, sha256, write_json
from sbercluster.metrics import all_metrics
from sbercluster.published_inputs import read_archive
from sbercluster.research import annual_profile, block_month_indices, load_development
from sbercluster.selection_frontier import fit_frontier, predict_frontier
from sbercluster.selection_contrasts import denominator_direction, contrast_basis_transform, contrast_annual, axis_decomposition
from sbercluster.selection_artifacts import make_artifact, predict_annual_panel

SOURCES = ['scripts/benchmark_contrasts.py', 'sbercluster/selection_contrasts.py',
           'sbercluster/selection_frontier.py', 'sbercluster/selection_artifacts.py']


def run(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / 'status.json', {'status': 'running'})
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8'))
    development, _, monthly, ids, scaler, panel_manifest = load_development(ROOT, cfg)
    original, contrasted = annual_profile(monthly), contrast_annual(monthly, scaler)
    direction, basis = denominator_direction(scaler), contrast_basis_transform(scaler)
    common_graph, common_graph_info = knn_graph(original, 15, 'union')
    contrast_graph, contrast_graph_info = knn_graph(contrasted, 15, 'union')
    screen = ROOT / '.local/frontier-20261003/benchmark-constrained'
    prior = json.loads((screen / 'labels.json').read_text('utf-8'))
    prior_models = json.loads((screen / 'models.json').read_text('utf-8'))
    if prior['ids'] != ids:
        raise ValueError('Previous shortlist IDs differ')
    baseline2 = read_archive(ROOT, 'review-20260924-small-k', 'labels.json.gz')
    if baseline2['ids'] != ids:
        raise ValueError('Published K2 IDs differ')
    labels = {name: np.asarray(prior['labels'][name], dtype=np.int32) for name in
              ('historical_kmeans4', 'sw_constrained_huber75', 'sw_constrained_ordinary')}
    labels['historical_kmeans2'] = np.asarray(baseline2['labels']['kmeans_k2'], dtype=np.int32)
    direction_unit = direction / np.linalg.norm(direction)
    common_axis = original @ direction_unit
    annual_total = development.assign(log_total=np.log(development[TOTAL])).groupby('entity_id').log_total.median().loc[ids].to_numpy()
    raw_ratios = np.log(development[CATEGORIES].to_numpy() / development[TOTAL].to_numpy()[:, None])
    raw_mean_ratio = development.assign(mean_ratio=raw_ratios.mean(axis=1)).groupby('entity_id').mean_ratio.median().loc[ids].to_numpy()
    for name, variable in [('denominator_axis_kmeans4', common_axis), ('total_level_kmeans4', annual_total)]:
        z, _, _ = fit_frontier(variable[:, None], {'k': 4, 'min_fraction': .03})
        labels[name] = z
    models, rows = {}, []
    provenance = {'started_at': datetime.now(timezone.utc).isoformat(), 'panel_sha256': panel_manifest['panel_sha256'],
                  'ids': ids, 'scaler': scaler, 'direction': direction.tolist(), 'monthly_basis_transform': basis.tolist(),
                  'pipeline': 'frozen monthly log-ratio/IQR -> canonical four-contrast Householder basis -> componentwise 12-month median -> frozen prototype rule',
                  'basis_orientation': 'Householder reflector mapping positive normalized denominator direction to fifth coordinate; first four coordinates retained, fifth zero',
                  'common_graph': common_graph_info, 'contrast_graph': contrast_graph_info,
                  'source_sha256': {name: sha256(ROOT / name) for name in SOURCES},
                  'limits': ['Geometry differs: original-space SW is not a sufficient criterion to accept or reject denominator invariance.',
                             'Fixed scaler invariance holds; refitting IQR after denominator perturbation is a different pipeline.',
                             'All comparisons use inspected retrospective 2023 and full-period complete cohort.',
                             'Denominator stress is a mathematical counterfactual, not an observed measurement error distribution.',
                             'Contrast basis orientation is fixed and declared; robust componentwise medians depend on it.']}
    write_json(output / 'provenance.json', provenance)
    np.save(output / 'original_features.npy', original, allow_pickle=False)
    np.save(output / 'contrast_features.npy', contrasted, allow_pickle=False)
    for k in (2, 4):
        name = f'contrast_kmeans{k}'
        z, model, info = fit_frontier(contrasted, {'k': k, 'min_fraction': .03})
        labels[name], models[name] = z, model
        artifact = make_artifact(model, scaler, pre_aggregation_transform=basis,
                                 provenance={'candidate_id': name, 'training_year': 2023, 'pipeline': provenance['pipeline'],
                                             'scope': 'exploratory denominator-invariant consumer contrasts'})
        write_json(output / (name + '.json'), artifact)
        panel = pd.read_csv(ROOT / 'data/processed/panel.csv', dtype={'entity_id': str, 'period': str})
        replay = predict_annual_panel(panel, artifact, 2023)
        if replay.entity_id.tolist() != ids or not np.array_equal(replay.cluster, z):
            raise ValueError('Contrast raw-panel replay differs')
        print(json.dumps({'candidate': name, 'sizes': info['cluster_sizes'], 'raw_replay_exact': True}), flush=True)
    for name, z in labels.items():
        def eta(variable):
            total = float(np.square(variable - variable.mean()).sum())
            between = sum(float(np.count_nonzero(z == c) * (variable[z == c].mean() - variable.mean()) ** 2)
                          for c in np.unique(z))
            return between / total if total else None
        rows.append({'id': name, 'common_original_geometry': all_metrics(original, z, common_graph),
                     'intrinsic_contrast_geometry': all_metrics(contrasted, z, contrast_graph),
                     'original_axis': axis_decomposition(original, z, scaler),
                     'raw_common_log_ratio_eta_squared': eta(raw_mean_ratio),
                     'raw_log_total_eta_squared': eta(annual_total),
                     'ARI_to_denominator_axis_k4': float(adjusted_rand_score(labels['denominator_axis_kmeans4'], z)),
                     'ARI_to_total_level_k4': float(adjusted_rand_score(labels['total_level_kmeans4'], z)),
                     'ARI_to_original_k4': float(adjusted_rand_score(labels['historical_kmeans4'], z))})
    write_json(output / 'comparisons.json', rows)
    write_json(output / 'labels.json', {'ids': ids, 'labels': {name: z.tolist() for name, z in labels.items()}})
    stress = []
    for mode in ('constant_per_entity', 'independent_per_entity_month'):
        for scale in (.1, .25, .5):
            changes = np.random.default_rng(17003).normal(size=(1 if mode == 'constant_per_entity' else 12, len(ids))) * scale
            perturbed = monthly - changes[:, :, None] * direction
            stressed_original = annual_profile(perturbed)
            stressed_contrasts = contrast_annual(perturbed, scaler)
            maximum_difference = float(np.max(np.abs(stressed_contrasts - contrasted)))
            for name in ('historical_kmeans4', 'sw_constrained_huber75', 'sw_constrained_ordinary'):
                z = predict_frontier(stressed_original, prior_models[name])
                stress.append({'id': name, 'mode': mode, 'log_multiplier_sd': scale,
                               'ARI': float(adjusted_rand_score(labels[name], z)),
                               'changed': int(np.count_nonzero(labels[name] != z)),
                               'contrast_feature_max_abs_difference': maximum_difference})
            for name, model in models.items():
                z = predict_frontier(stressed_contrasts, model)
                if not np.array_equal(z, labels[name]):
                    raise ArithmeticError('Denominator-invariant contrast labels changed under stress')
                stress.append({'id': name, 'mode': mode, 'log_multiplier_sd': scale, 'ARI': 1., 'changed': 0,
                               'contrast_feature_max_abs_difference': maximum_difference})
    write_json(output / 'stress.json', stress)
    stability, transfer = {name: [] for name in models}, {name: [] for name in models}
    for fold in range(4):
        test = np.arange(3 * fold, 3 * fold + 3)
        train = np.setdiff1d(np.arange(12), test)
        train_x, test_x = contrast_annual(monthly[train], scaler), contrast_annual(monthly[test], scaler)
        test_original = annual_profile(monthly[test])
        original_graph, _ = knn_graph(test_original, 15, 'union')
        test_graph, _ = knn_graph(test_x, 15, 'union')
        for name, model in models.items():
            _, fitted, _ = fit_frontier(train_x, model['spec'])
            z = predict_frontier(test_x, fitted)
            transfer[name].append({'fold': fold, 'common_SW': all_metrics(test_original, z, original_graph)['SW'],
                                   'contrast_SW': all_metrics(test_x, z, test_graph)['SW'],
                                   'sizes': np.bincount(z).tolist()})
    for draw in range(args.bootstrap):
        indices = block_month_indices(2200 + draw)
        xb = contrast_annual(monthly[indices], scaler)
        for name, model in models.items():
            z, _, _ = fit_frontier(xb, model['spec'])
            stability[name].append({'draw': draw, 'seed': 2200 + draw,
                                    'ARI': float(adjusted_rand_score(labels[name], z))})
    write_json(output / 'transfer.json', transfer)
    write_json(output / 'stability.json', stability)
    for name in SOURCES:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    write_json(output / 'status.json', {'status': 'completed', 'elapsed_seconds': time.perf_counter() - started})
    write_json(output / 'manifest.json', {'files_sha256': {p.relative_to(output).as_posix(): sha256(p)
                                                         for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'status': 'completed', 'output': str(output), 'seconds': time.perf_counter() - started}), flush=True)


def main():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError('Cannot set BelowNormal worker priority')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=30)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('Positive bootstrap count required')
    try:
        with threadpool_limits(limits=1):
            run(args)
    except BaseException as exc:
        path = Path(args.output) / 'status.json'
        if path.exists():
            write_json(path, {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
