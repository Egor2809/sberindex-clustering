import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import unittest

spec = importlib.util.spec_from_file_location('external_validation_v4', Path(__file__).resolve().parents[1] / 'scripts/external_validation_v4.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def test_registry_half_open_validity_and_ambiguity():
    labels = pd.DataFrame({'entity_id': ['tid_1'], 'cluster': [0]})
    registry = pd.DataFrame({'territory_id': [1, 1], 'year_from': [2020, 2023], 'year_to': [2023, 9999], 'name': ['old', 'new']})
    assert v.join_registry(labels, registry).iloc[0]['name'] == 'new'
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Duplicate'):
        v.join_registry(labels, pd.concat([registry, registry.iloc[[1]]]))
    with unittest.TestCase().assertRaisesRegex(ValueError, 'no registry'):
        v.join_registry(labels, registry, year=2019)


def test_partial_r2_matches_full_regression_and_removes_region_confound():
    rng = np.random.default_rng(52)
    n = 160
    region = np.repeat(range(8), 20)
    cluster = rng.integers(0, 4, n)
    y = 3 * region + .7 * cluster + rng.normal(size=n)
    f = pd.DataFrame({'region_code': region, 'cluster': cluster, 'municipal_district_type': np.where(np.arange(n) % 2, 'rural', 'urban'), 'y': y, 'log_expense_2023': rng.normal(size=n)})
    m, _ = v.residual_matrix(f, 'y', 'region_type_level')
    actual = v.partial_r2(m.T @ m, True)
    controls = pd.get_dummies(f.region_code.astype(str) + '|' + f.municipal_district_type, dtype=float).to_numpy()
    controls = np.column_stack([controls, f.log_expense_2023])
    labels = pd.get_dummies(f.cluster, dtype=float).to_numpy()
    rss0 = np.sum((y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]) ** 2)
    full = np.column_stack([controls, labels])
    rss1 = np.sum((y - full @ np.linalg.lstsq(full, y, rcond=None)[0]) ** 2)
    assert abs(actual - (rss0 - rss1) / rss0) < 1e-12
    f['y'] = region.astype(float)
    m, _ = v.residual_matrix(f, 'y', 'region')
    assert v.partial_r2(m.T @ m) is None


def test_bootstrap_is_reproducible_and_does_not_mutate_labels():
    f = pd.DataFrame({'region_code': np.repeat(range(4), 12), 'cluster': np.tile([0, 1, 2, 3], 12), 'municipal_district_type': ['district'] * 48, 'y': np.arange(48) % 7})
    before = f.copy(deep=True)
    a = v.audit_effect(f, 'y', 'region_type', bootstrap=20)
    assert a == v.audit_effect(f, 'y', 'region_type', bootstrap=20)
    pd.testing.assert_frame_equal(f, before)
    assert 0 <= a['partial_r_squared'] <= 1
    assert a['n_in_mixed_cluster_strata'] == 48


def test_partial_r2_is_invariant_to_outcome_and_control_units():
    rng = np.random.default_rng(551)
    matrix = rng.normal(size=(60, 6))
    matrix[:, 0] += .8 * matrix[:, 1] + .3 * matrix[:, -1]
    cross = matrix.T @ matrix
    expected = v.partial_r2(cross, True)
    for outcome_scale, level_scale in [(1e-9, 1.), (1e9, 1e-9), (1e-9, 1e9)]:
        scales = np.array([outcome_scale, 1, 1, 1, 1, level_scale])
        actual = v.partial_r2(cross * scales[:, None] * scales[None, :], True)
        assert actual is not None
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_missing_control_and_unregistered_label_cannot_change_the_cohort_silently():
    frame = pd.DataFrame({'region_code': [1, 1, 2, 2], 'cluster': [0, 1, 2, 3],
                          'municipal_district_type': ['district'] * 4, 'y': [1., 2., 3., 4.]})
    for column, value in [('region_code', None), ('municipal_district_type', None), ('cluster', 4)]:
        bad = frame.copy()
        bad.loc[0, column] = value
        with unittest.TestCase().assertRaises(ValueError):
            v.residual_matrix(bad, 'y', 'region_type')


def test_fractional_constant_and_fully_controlled_outcomes_are_undefined():
    rng = np.random.default_rng(222)
    frame = pd.DataFrame({'region_code': np.repeat([1, 2], 12), 'cluster': np.tile([0, 1, 2, 3], 6),
                          'municipal_district_type': ['district'] * 24,
                          'log_expense_2023': rng.normal(size=24)})
    for values, control in [(np.full(24, .1), 'region'),
                            (np.repeat([.1, .7], 12), 'region'),
                            (.1 + 2.3 * frame.log_expense_2023, 'region_type_level')]:
        frame['y'] = values
        matrix, _ = v.residual_matrix(frame, 'y', control)
        assert v.partial_r2(matrix.T @ matrix, control == 'region_type_level') is None


def test_small_genuine_residual_is_not_discarded():
    rng = np.random.default_rng(223)
    frame = pd.DataFrame({'region_code': np.repeat([1, 2], 60),
                          'cluster': rng.integers(0, 4, size=120),
                          'municipal_district_type': ['district'] * 120,
                          'log_expense_2023': rng.normal(size=120)})
    frame['y'] = 2 * frame.log_expense_2023 + 1e-4 * (frame.cluster + rng.normal(size=120))
    matrix, _ = v.residual_matrix(frame, 'y', 'region_type_level')
    actual = v.partial_r2(matrix.T @ matrix, True)
    controls = np.column_stack([pd.get_dummies(frame.region_code, dtype=float), frame.log_expense_2023])
    full = np.column_stack([controls, pd.get_dummies(frame.cluster, dtype=float)])
    y = frame.y.to_numpy()
    rss0 = np.sum((y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]) ** 2)
    rss1 = np.sum((y - full @ np.linalg.lstsq(full, y, rcond=None)[0]) ** 2)
    assert actual is not None
    # A double-precision Gram matrix loses precision when nearly all outcome
    # variance is removed. Bound that cancellation relative to residual variance.
    gram_roundoff = 8 * np.finfo(float).eps * np.sum(matrix[:, 0] ** 2) / rss0
    np.testing.assert_allclose(actual, (rss0 - rss1) / rss0, atol=gram_roundoff, rtol=0)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function))
    return suite

if __name__ == "__main__":
    unittest.main()
