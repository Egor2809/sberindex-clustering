"""Independent small adversarial controls for the new train-only contracts.

Run with the pinned Python, one BLAS/OpenMP thread, and --mode all. These tests
use synthetic source-compatible rows, never the national fitting workload.
They test counterfactual isolation and independently computed mathematical
quantities rather than replaying the accepted 245-test suite.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sbercluster.io import CATEGORIES, TOTAL


def fixture():
    rng = np.random.default_rng(401703)
    n = 48
    regions = np.repeat(np.arange(10, 16), 8)
    entity_effect = rng.normal(size=(n, 5)) * .34
    entity_effect += np.eye(5)[np.arange(n) % 4] * .6
    season = rng.normal(size=(12, 5)) * .12
    logs = -2.4 + entity_effect[None] + season[:, None] + rng.normal(size=(12, n, 5)) * .07
    ids = np.array([f'tid_{i + 1}' for i in range(n)])
    frames = []
    for year in [2023, 2024]:
        for month in range(12):
            total = np.exp(9.5 + np.arange(n) * .003)
            values = np.exp(logs[month] + (year - 2023) * .025) * total[:, None]
            frame = pd.DataFrame(values, columns=CATEGORIES)
            frame[TOTAL] = total
            frame['entity_id'] = ids
            frame['territory_id'] = np.arange(1, n + 1)
            frame['period'] = f'{year}-{month + 1:02d}-01'
            frames.append(frame)
    return logs, regions, ids, pd.concat(frames, ignore_index=True)


def assert_state_equal(a, b):
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert canonical(a) == canonical(b), 'Excluded observations changed fitted state'


def assert_context_invariance(predict, x):
    original = predict(x)
    extended = np.vstack([x, np.repeat([[90.] * x.shape[1]], 31, axis=0)])
    np.testing.assert_allclose(predict(extended)[:len(x)], original, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.concatenate([predict(x[i:i + 1]) for i in range(len(x))]),
                               original, rtol=0, atol=1e-12)


def expect_detection(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError('The deliberately invalid implementation evaded the oracle')


def selftests():
    x = np.array([[0., 2.], [1., 5.], [4., 1.]])
    expect_detection(lambda: assert_context_invariance(lambda a: a - np.median(a, axis=0), x))
    a, b = np.array([1., 3., 8.]), np.array([1., 3., 800.])
    expect_detection(lambda: assert_state_equal({'center': float(a.mean())}, {'center': float(b.mean())}))
    # Median and a nondiagonal full-rank linear map do not generally commute.
    rows = np.array([[0., 10.], [2., 2.], [10., 0.]])
    matrix = np.array([[1., .6], [.6, 1.]])
    correct = np.median(rows @ matrix, axis=0)
    wrong = np.median(rows, axis=0) @ matrix
    assert np.max(np.abs(correct - wrong)) > 2., 'Preaggregation counterexample has no power'
    return {'detected_bad_evaluation_median': True, 'detected_bad_excluded_fit': True,
            'median_linear_map_noncommutativity_gap': float(np.max(np.abs(correct - wrong)))}


def economic_tests():
    from sbercluster.economic_generalization import (fit_region_representation,
        fit_predict_probe, type_design, score_predictions)
    logs, regions, ids, _ = fixture()
    holdout = 10
    base = fit_region_representation(logs, regions, holdout, ids)
    mutated = logs.copy()
    mutated[:, regions == holdout] += np.array([8., -4., 6., 1., -3.])
    other = fit_region_representation(mutated, regions, holdout, ids)
    assert_state_equal(base['artifact'], other['artifact'])
    train = np.flatnonzero(regions != holdout)
    np.testing.assert_array_equal(base['train_indices'], train)
    np.testing.assert_array_equal(base['labels']['kmeans4'][train], other['labels']['kmeans4'][train])
    np.testing.assert_array_equal(base['profiles'][train], other['profiles'][train])
    ref = logs[:, train].reshape(-1, 5)
    np.testing.assert_allclose(base['artifact']['scaler']['ratio_center'], np.median(ref, axis=0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(base['artifact']['scaler']['ratio_iqr'],
                               np.percentile(ref, 75, axis=0) - np.percentile(ref, 25, axis=0), rtol=0, atol=1e-15)
    # Outcome missingness cannot select the expense fitting cohort: no outcome
    # argument exists, and every eligible other-region entity appears in state.
    assert base['artifact']['training_entity_ids'] == ids[train].tolist()
    # Extra heldout entities can change predictions, never training state.
    extended = fit_region_representation(np.concatenate([logs, mutated[:, :2]], axis=1),
        np.r_[regions, [holdout, holdout]], holdout, np.r_[ids, ['new_a', 'new_b']])
    assert_state_equal(base['artifact'], extended['artifact'])
    x = base['profiles'][train]
    y = 4. + .3 * x[:, 0] - .5 * x[:, 1] + .2 * x[:, 2] ** 2
    test = base['profiles'][regions == holdout][:3]
    probe_results = {}
    for kind in ['linear', 'quadratic', 'histogram']:
        args = {'numeric_columns': list(range(5))} if kind == 'quadratic' else {}
        predictions, fitted = fit_predict_probe(x, y, test, kind, **args)
        prediction_more, fitted_more = fit_predict_probe(x, y, np.vstack([test, [[25.] * 5]]), kind, **args)
        assert_state_equal(fitted, fitted_more)
        np.testing.assert_allclose(prediction_more[:len(test)], predictions, rtol=0, atol=1e-11)
        assert_context_invariance(lambda a: fit_predict_probe(x, y, a, kind, **args)[0], test)
        probe_results[kind] = 'PASS'
    ta, tb, vocab = type_design(np.array(['city', 'district', 'city']), np.array(['new_type', 'city']))
    assert vocab == ['city', 'district'] and not tb[0].any()
    np.testing.assert_array_equal(tb[1], ta[0])
    # An outcome offset can disappear under centered scoring. This diagnostic
    # therefore cannot establish calibrated absolute-level forecast accuracy.
    r = np.repeat([1, 2, 3], 3)
    yy = np.arange(9.) + np.repeat([10., 100., 1000.], 3)
    pp = yy + np.repeat([5., 20., 80.], 3)
    scored = score_predictions(yy, {'offset_bias': pp}, r, bootstrap=99)[0]
    assert scored['equal_region_centered_error_sample_variance'] == 0.
    assert scored['equal_region_mse'] > 2000.
    return {'excluded_expense_fit_state': 'PASS', 'independent_scaler_oracle': 'PASS',
            'all_eligible_training_entities': len(train), 'extra_heldout_entities': 'PASS',
            'probe_context': probe_results, 'unknown_type_encoding': 'PASS',
            'centered_zero_with_absolute_mse': scored['equal_region_mse']}


def temporal_tests():
    from sbercluster.temporal_profiles import (fit_temporal_calibration, transform_temporal_rows,
        fit_temporal_model, predict_temporal_panel)
    from sbercluster.features import transform_frozen
    logs, _, _, panel = fixture()
    periods = [f'2023-{m:02d}-01' for m in range(1, 13)]
    calibration = fit_temporal_calibration(logs, periods, 1.)
    offsets = np.asarray(calibration['calendar_offsets'])
    metric = np.asarray(calibration['metric'])
    expected_offsets = np.median(logs, axis=1) - np.median(logs, axis=1).mean(axis=0)
    np.testing.assert_allclose(offsets, expected_offsets, rtol=0, atol=1e-15)
    query = logs[0, :3]
    assert_context_invariance(lambda a: transform_temporal_rows(a, ['2024-01-01'] * len(a), calibration), query)
    np.testing.assert_allclose(transform_temporal_rows(query, ['2024-01-01'] * len(query), calibration),
                               (query - expected_offsets[0]) @ metric, rtol=0, atol=1e-15)
    cfg = json.loads((ROOT / 'configs/research_validation.json').read_text('utf-8'))
    artifact = fit_temporal_model(panel, cfg, reliability=1.)
    changed = panel.copy()
    changed.loc[changed.period.str.startswith('2024'), CATEGORIES] *= np.exp(4.)
    artifact_other = fit_temporal_model(changed, cfg, reliability=1.)
    assert_state_equal(artifact, artifact_other)
    predictions = predict_temporal_panel(panel, artifact, 2024).set_index('entity_id')
    shuffled = predict_temporal_panel(panel.sample(frac=1., random_state=44), artifact, 2024).set_index('entity_id')
    pd.testing.assert_frame_equal(predictions, shuffled)
    one = panel[panel.entity_id == 'tid_1']
    pd.testing.assert_frame_equal(predictions.loc[['tid_1']], predict_temporal_panel(one, artifact, 2024).set_index('entity_id'))
    new = one.copy()
    new['entity_id'], new['territory_id'] = 'tid_99999', 99999
    new[CATEGORIES] *= np.exp(5.)
    more = predict_temporal_panel(pd.concat([panel, new]), artifact, 2024).set_index('entity_id')
    pd.testing.assert_frame_equal(predictions, more.loc[predictions.index])
    assert bool(more.loc['tid_99999', 'new_entity'])
    # Independent nearest-centre oracle from original rows: transform month
    # first, median second, and explicitly compare complete assignment rule.
    for entity in ['tid_1', 'tid_17', 'tid_38']:
        rows = panel[(panel.entity_id == entity) & panel.period.str.startswith('2024')].sort_values('period')
        x = transform_frozen(rows, artifact['feature_scaler'])
        profile = np.median((x - np.asarray(artifact['calibration']['calendar_offsets'])) @
                            np.asarray(artifact['calibration']['metric']), axis=0)
        square = ((np.asarray(artifact['centers']) - profile) ** 2).sum(axis=1)
        assert predictions.loc[entity, 'cluster'] == int(square.argmin())
    expect_detection(lambda: fit_temporal_calibration(logs[:6], periods[:6]))
    bad_cfg = deepcopy(cfg)
    bad_cfg['features']['calibration_end'] = '2023-06-01'
    expect_detection(lambda: fit_temporal_model(panel, bad_cfg))
    expect_detection(lambda: predict_temporal_panel(one.iloc[1:], artifact, 2023))
    expect_detection(lambda: transform_temporal_rows(query, ['2024-13-01'] * len(query), calibration))
    return {'future_fit_isolation': 'PASS', 'independent_offset_formula': 'PASS',
            'row_chunk_batch_order': 'PASS', 'unknown_entity_metadata': 'PASS',
            'independent_preaggregation_labels': 'PASS', 'unsupported_calibration_calendar': 'PASS',
            'missing_or_invalid_month_rejected': 'PASS'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['selftest', 'economic', 'temporal', 'all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    results = {}
    for name, fn in [('selftest', selftests), ('economic', economic_tests), ('temporal', temporal_tests)]:
        if args.mode not in ['all', name]:
            continue
        try:
            results[name] = {'status': 'PASS', 'checks': fn()}
        except Exception as exc:
            results[name] = {'status': 'FAIL', 'error': str(exc), 'traceback': traceback.format_exc()}
    source_names = ['sbercluster/temporal_profiles.py', 'sbercluster/economic_generalization.py',
                    'sbercluster/features.py', 'sbercluster/selection_frontier.py']
    output = {'status': 'PASS' if all(r['status'] == 'PASS' for r in results.values()) else 'FAIL',
              'scope': 'Independent synthetic mechanism controls; no national fit or untouched future evidence',
              'elapsed_seconds': time.perf_counter() - started, 'results': results,
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
              'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'cases': {name: r['status'] for name, r in results.items()},
                      'output': str(args.output), 'seconds': output['elapsed_seconds']}))
    if output['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
