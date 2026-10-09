"""Independent saved-fold/OOF arithmetic audit; no clustering/probe refit."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sbercluster.io import CATEGORIES, TOTAL
RUN = ROOT / '.local/core-round3-20261003/economic/strict-wage'
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def nearest(x, model):
    y = x @ np.asarray(model['transform'])
    return (((y[:, None] - np.asarray(model['centers'])[None]) ** 2).sum(axis=2) + np.asarray(model['biases'])).argmin(axis=1)


def sse(x, z):
    return sum(float(((x[z == c] - x[z == c].mean(axis=0)) ** 2).sum()) for c in np.unique(z))


def main():
    protocol = json.loads((RUN / 'protocol.json').read_text('utf-8'))
    status = json.loads((RUN / 'status.json').read_text('utf-8'))
    results = json.loads((RUN / 'results.json').read_text('utf-8'))
    assert stable(protocol) == status['protocol_sha256'] == results['protocol_sha256']
    assert status['results_sha256'] == digest(RUN / 'results.json')
    assert status['predictions_sha256'] == digest(RUN / 'all-oof-predictions.csv')
    for name, sha in protocol['input_sha256'].items():
        assert digest(ROOT / name) == sha
    for name, sha in protocol['source_sha256'].items():
        assert digest(RUN / 'source' / name) == sha
    labels_file = ROOT / 'reports/external-national-2026-10-03/frozen-labels.json'
    ids = np.array(json.loads(labels_file.read_text('utf-8'))['ids'])
    cohorts = {year: pd.read_csv(ROOT / f'reports/external-national-2026-10-03/cohort-{year}.csv').set_index('entity_id').reindex(ids) for year in [2023, 2024]}
    regions = cohorts[2023].region_code.to_numpy(int)
    panel = pd.read_csv(ROOT / 'data/processed/panel.csv')
    logs, totals = [], []
    for month in range(1, 13):
        block = panel[panel.period == f'2023-{month:02d}-01'].set_index('entity_id').reindex(ids)
        values, total = block[CATEGORIES].to_numpy(float), block[TOTAL].to_numpy(float)
        logs.append(np.log(values / total[:, None])); totals.append(total)
        np.testing.assert_array_equal(block.region_code.to_numpy(int), regions)
    logs = np.asarray(logs)
    numeric = np.column_stack([np.log(cohorts[2023].population_total.where(cohorts[2023].population_total > 0)), np.log(np.median(totals, axis=0))])
    types = cohorts[2023].municipal_district_type.fillna('__missing__').to_numpy(str)
    good = np.isfinite(numeric).all(axis=1) & (types != '__missing__')
    ys = {year: cohorts[year][protocol['primary_outcome']].to_numpy(float) for year in [2023, 2024]}
    expected_regions = np.unique(regions[good & (np.isfinite(ys[2023]) | np.isfinite(ys[2024]))])
    frames, ratios, minimums = [], [], []
    max_profile_error = 0.
    reference_fold = None
    for held_out in expected_regions:
        directory = RUN / 'folds' / str(held_out)
        art = json.loads((directory / 'representation.json').read_text('utf-8'))
        probes = json.loads((directory / 'probes.json').read_text('utf-8'))
        original = {k: v for k, v in art.items() if k not in ['artifact_sha256', 'protocol_sha256', 'elapsed_seconds', 'cache_sha256']}
        assert stable(original) == art['artifact_sha256'] == probes['representation_sha256']
        assert art['protocol_sha256'] == probes['protocol_sha256'] == stable(protocol)
        assert digest(directory / 'profiles-and-labels.npz') == art['cache_sha256']
        assert digest(directory / 'predictions.csv') == probes['predictions_sha256']
        train = regions != held_out
        assert art['training_entity_ids'] == ids[train].tolist()
        assert held_out not in art['training_regions']
        assert art['training_entities'] == train.sum() and art['calibration_observations'] == 12 * train.sum()
        reference = logs[:, train].reshape(-1, 5)
        center, iqr = np.median(reference, axis=0), np.percentile(reference, 75, axis=0) - np.percentile(reference, 25, axis=0)
        np.testing.assert_array_equal(center, art['scaler']['ratio_center'])
        np.testing.assert_array_equal(iqr, art['scaler']['ratio_iqr'])
        profiles = np.median((logs - center) / iqr / np.sqrt(5), axis=0)
        with np.load(directory / 'profiles-and-labels.npz', allow_pickle=False) as cache:
            error = float(np.max(np.abs(cache['profiles'] - profiles)))
            max_profile_error = max(error, max_profile_error)
            assert error <= 1e-14
            all_labels = {name: cache[name].copy() for name in ['kmeans4', 'huber75']}
        for name, z in all_labels.items():
            model, info = art['models'][name], art['model_info'][name]
            np.testing.assert_array_equal(z, nearest(profiles, model))
            counts = np.bincount(z[train], minlength=4)
            assert counts.tolist() == info['cluster_sizes']
            assert info['min_required_size'] == max(2, int(np.ceil(.03 * train.sum())))
            assert bool(counts.min() >= info['min_required_size']) == info['eligible_occupancy']
        minimums.append(int(np.bincount(all_labels['huber75'][train], minlength=4).min()))
        tss = float(((profiles[train] - profiles[train].mean(axis=0)) ** 2).sum())
        sk, sh = [sse(profiles[train], all_labels[name][train]) for name in ['kmeans4', 'huber75']]
        ratio = (tss / sh - 1) / (tss / sk - 1)
        ratios.append(float(ratio))
        assert abs(art['model_info']['huber75']['maximum_common_space_sse'] - tss / (1 + .95 * (tss / sk - 1))) < 1e-9
        assert ratio >= .95 - 1e-11
        assert art['models']['huber75']['spec'] == protocol['clustering']['huber75']
        assert len(art['model_info']['huber75']['trace']) <= 9
        observed_train = train & good & np.isfinite(ys[2023])
        assert probes['training_outcome_entities'] == ids[observed_train].tolist()
        expected_vocab = sorted(set(types[observed_train].tolist()))
        assert probes['type_vocabulary'] == expected_vocab and probes['training_outcome_year'] == 2023
        test = (regions == held_out) & good
        frame = pd.read_csv(directory / 'predictions.csv')
        assert frame.entity_id.tolist() == ids[test].tolist()
        np.testing.assert_allclose(frame.observed2023, ys[2023][test], rtol=0, atol=2e-14)
        np.testing.assert_allclose(frame.observed2024, ys[2024][test], rtol=0, atol=2e-14)
        assert len(probes['probes']) == 12
        if reference_fold is None:
            reference_fold = (frame, profiles, all_labels, observed_train, test, expected_vocab, probes)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    oof = pd.read_csv(RUN / 'all-oof-predictions.csv')
    assert not oof.entity_id.duplicated().any()
    assert oof.entity_id.tolist() == combined.entity_id.tolist()
    predictions = [c for c in oof if c not in ['entity_id', 'held_out_region', 'observed2023', 'observed2024']]
    np.testing.assert_allclose(oof[predictions], combined[predictions], rtol=0, atol=2e-14)
    # Independent SVD numerical reference for first actual fold's linear models.
    frame, x, z, observed_train, test, vocab, probes = reference_fold
    reference_errors = {}
    dummy_train = np.column_stack([types[observed_train] == t for t in vocab]).astype(float)
    dummy_test = np.column_stack([types[test] == t for t in vocab]).astype(float)
    controls_a, controls_b = np.c_[dummy_train, numeric[observed_train]], np.c_[dummy_test, numeric[test]]
    for name in ['controls', 'continuous5', 'kmeans4', 'huber75']:
        a, b = controls_a, controls_b
        if name == 'continuous5':
            a, b = np.c_[a, x[observed_train]], np.c_[b, x[test]]
        elif name in z:
            a, b = np.c_[a, np.eye(4)[z[name][observed_train]]], np.c_[b, np.eye(4)[z[name][test]]]
        mean, scale = a.mean(axis=0), a.std(axis=0)
        active = scale > 1e-12
        design, target = np.c_[np.ones(len(a)), (a[:, active] - mean[active]) / scale[active]], np.c_[np.ones(len(b)), (b[:, active] - mean[active]) / scale[active]]
        u, s, vt = np.linalg.svd(design, full_matrices=False)
        keep = s > s.max() * 1e-11
        beta = (vt[keep].T / s[keep]) @ (u[:, keep].T @ ys[2023][observed_train])
        reference_errors[name] = float(np.max(np.abs(target @ beta - frame['linear_' + name])))
        assert reference_errors[name] < 1e-10
    max_score_error, year_summary = 0., []
    for year in [2023, 2024]:
        part = oof[np.isfinite(oof[f'observed{year}'])]
        y, rr = part[f'observed{year}'].to_numpy(), part.held_out_region.to_numpy(int)
        groups = [np.flatnonzero(rr == r) for r in np.unique(rr)]
        rng = np.random.default_rng(20261003)
        draws = rng.integers(len(groups), size=(1999, len(groups)))
        pair_groups = [g for g in groups if len(g) >= 2]
        pair_draws = rng.integers(len(pair_groups), size=(1999, len(pair_groups)))
        loss, centered, signed = {}, {}, {}
        for name in predictions:
            error = y - part[name].to_numpy()
            loss[name] = np.array([np.dot(error[g], error[g]) / len(g) for g in groups])
            centered[name] = np.array([np.dot(error[g] - error[g].mean(), error[g] - error[g].mean()) / (len(g) - 1) for g in pair_groups])
            signed[name] = error
        for stored in [r for r in results['scores'] if r['outcome_year'] == year]:
            name, error = stored['predictor'], signed[stored['predictor']]
            expect = {'municipality_rmse': float(np.sqrt(np.dot(error, error) / len(error))), 'municipality_mae': float(np.mean(np.abs(error))),
                      'equal_region_mse': float(loss[name].mean()), 'equal_region_centered_error_sample_variance': float(centered[name].mean()),
                      'equal_region_mean_error': float(np.mean([error[g].mean() for g in groups]))}
            assert stored['n'] == len(y) and stored['regions'] == len(groups) and stored['regions_with_pairs'] == len(pair_groups)
            for key, value in expect.items():
                gap = abs(value - stored[key]); max_score_error = max(gap, max_score_error); assert gap < 2e-12, (year, name, key, gap)
            for other in predictions:
                comparison = stored['comparisons'][other]
                gain, centered_gain = loss[other] - loss[name], centered[other] - centered[name]
                expected = {'equal_region_mse_improvement': gain.mean(), 'conditional_region_bootstrap_95': np.quantile(gain[draws].mean(axis=1), [.025, .975]),
                            'equal_region_centered_error_variance_improvement': centered_gain.mean(), 'conditional_centered_region_bootstrap_95': np.quantile(centered_gain[pair_draws].mean(axis=1), [.025, .975]),
                            'region_fraction_improved_mse': np.mean(gain > 0)}
                for key, value in expected.items():
                    gap = float(np.max(np.abs(np.asarray(value) - np.asarray(comparison[key])))); max_score_error = max(gap, max_score_error); assert gap < 2e-12, (year, name, other, key, gap)
        rows = {r['predictor']: r for r in results['scores'] if r['outcome_year'] == year}
        year_summary.append({'year': year, 'n': len(y), 'regions': len(groups), 'RMSE': {name: rows[name]['municipality_rmse'] for name in ['histogram_controls', 'histogram_continuous5', 'histogram_kmeans4', 'histogram_huber75']},
            'h75_vs_k4_by_family': {family: rows[family + '_huber75']['comparisons'][family + '_kmeans4'] for family in ['linear', 'quadratic', 'histogram']},
            'histogram_h75_vs_continuous5': rows['histogram_huber75']['comparisons']['histogram_continuous5']})
    # Priority-only CLI portability changes may alter file SHA, never the
    # numerical functions. Verify AST equality for every function except main.
    numerical_equal = {}
    for relative in protocol['source_sha256']:
        saved_tree, current_tree = [ast.parse(p.read_text('utf-8')) for p in [RUN / 'source' / relative, ROOT / relative]]
        saved = {n.name: ast.dump(n, include_attributes=False) for n in saved_tree.body if isinstance(n, ast.FunctionDef) and n.name != 'main'}
        current = {n.name: ast.dump(n, include_attributes=False) for n in current_tree.body if isinstance(n, ast.FunctionDef) and n.name != 'main'}
        numerical_equal[relative] = all(value == current.get(name) for name, value in saved.items())
        assert numerical_equal[relative], relative
        added = set(current) - set(saved)
        assert not added or (relative == 'scripts/validate_economic_generalization.py' and added == {'set_runtime_priority'}), (relative, added)
    report = {'status': 'PASS', 'scope': 'Independent fixed-prediction arithmetic and saved fold contract audit, no new fit', 'folds': len(expected_regions),
        'representation_rules_verified': 144, 'eligible_training_H75_CH_ratio_range': [min(ratios), max(ratios)], 'minimum_H75_training_cluster_size': min(minimums),
        'largest_profile_error': max_profile_error, 'first_fold_independent_SVD_prediction_errors': reference_errors,
        'score_rows': len(results['scores']), 'all_pair_bootstrap_comparisons': len(results['scores']) * len(predictions), 'bootstrap_draws': 1999,
        'largest_score_or_interval_error': max_score_error, 'year_summary': year_summary, 'numeric_functions_unchanged_from_run': numerical_equal,
        '2024_same2023_probes_and_inputs': True, 'architecture_selection_still_historical': True, 'strict_run_protocol_sha256': stable(protocol),
        'results_sha256': digest(RUN / 'results.json'), 'predictions_sha256': digest(RUN / 'all-oof-predictions.csv'), 'oracle_sha256': digest(Path(__file__))}
    (OUT / 'economic-evidence-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'folds': report['folds'], 'scores': report['score_rows'], 'comparisons': report['all_pair_bootstrap_comparisons'],
        'max_score_error': max_score_error, 'H75_CH_ratio_min': min(ratios), 'output': str(OUT / 'economic-evidence-review.json')}))


if __name__ == '__main__':
    main()
