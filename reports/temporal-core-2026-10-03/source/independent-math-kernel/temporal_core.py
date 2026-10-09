"""Small independently authored mathematical kernel; not a production pipeline.
Inputs are transformed arrays, never silently aligned by position in production.
Usage: python temporal_core.py --self-test
"""
from __future__ import annotations
import argparse
import itertools
import json
import platform
import numpy as np


def frozen_features(values: np.ndarray, total: np.ndarray, center: np.ndarray,
                    iqr: np.ndarray) -> np.ndarray:
    v, t, a, s = [np.asarray(z, dtype=np.float64) for z in (values, total, center, iqr)]
    if v.ndim != 2 or v.shape[1] != 5 or t.shape != (len(v),) or a.shape != (5,) or s.shape != (5,):
        raise ValueError('Expected N x 5 positive categories, N totals, and five scaler coordinates')
    if not all(np.isfinite(z).all() for z in (v, t, a, s)) or (v <= 0).any() or (t <= 0).any() or (s <= 0).any():
        raise ValueError('Nonfinite or nonpositive inputs')
    with np.errstate(over='raise', divide='raise', invalid='raise', under='raise'):
        x = (np.log(v / t[:, None]) - a) / s / np.sqrt(5.0)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed output')
    return x


def median_bounds(observed_deltas: np.ndarray, reference_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sharp marginal bounds for full-reference median; missing deltas are unrestricted."""
    d = np.asarray(observed_deltas, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] < 1 or not np.isfinite(d).all() or reference_n < 1 or len(d) > reference_n:
        raise ValueError('Invalid reference or observed deltas')
    q = reference_n - len(d)
    lo = np.concatenate((np.full((q, d.shape[1]), -np.inf), d), axis=0)
    hi = np.concatenate((d, np.full((q, d.shape[1]), np.inf)), axis=0)
    return np.median(lo, axis=0), np.median(hi, axis=0)


def annual_bounds(monthly_lo: np.ndarray, monthly_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(monthly_lo, float), np.asarray(monthly_hi, float)
    if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[0] != 12 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo > hi).any():
        raise ValueError('Exactly twelve finite monthly intervals required')
    return np.median(lo, axis=0), np.median(hi, axis=0)


def classify_box(lo: np.ndarray, hi: np.ndarray, centers: np.ndarray,
                 radius: float = np.inf, point: np.ndarray | None = None) -> dict:
    """Stable nearest-prototype certificate, conditional on a supplied box.
    Infinite radius disables support screening ONLY for mathematical tests.
    Production must supply a frozen radius and separate semantic/coverage gates.
    Pass the reported profile as point, especially for annual medians: it need
    not be the midpoint of its enclosing annual interval.
    """
    lo, hi, c = [np.asarray(z, dtype=np.float64) for z in (lo, hi, centers)]
    if lo.ndim != 1 or lo.size == 0 or hi.shape != lo.shape or c.ndim != 2 or c.shape[1] != len(lo) or len(c) < 2 or (lo > hi).any() or not all(np.isfinite(z).all() for z in (lo, hi, c)):
        raise ValueError('Invalid finite classification box')
    if np.isnan(radius) or radius < 0:
        raise ValueError('Invalid support radius')
    mid = lo / 2 + hi / 2
    representative = mid if point is None else np.asarray(point, dtype=np.float64)
    if representative.shape != lo.shape or not np.isfinite(representative).all() or (representative < lo).any() or (representative > hi).any():
        raise ValueError('Representative must lie in the supplied box')
    d2 = ((c - representative) ** 2).sum(axis=1)
    d2_mid = ((c - mid) ** 2).sum(axis=1)
    if not np.isfinite(d2).all() or not np.isfinite(d2_mid).all():
        raise ValueError('Nonfinite distance arithmetic')
    k = int(np.argmin(d2))  # Exact point tie goes to smallest stored index.
    margins = []
    for ell in range(len(c)):
        if ell == k:
            continue
        v = 2 * (c[k] - c[ell])
        bound = np.where(v >= 0, lo, hi)
        margins.append(float(v @ bound + c[ell] @ c[ell] - c[k] @ c[k]))
    max_distance = float(np.sqrt(np.maximum((lo - c[k]) ** 2, (hi - c[k]) ** 2).sum()))
    if not np.isfinite(margins).all() or not np.isfinite(max_distance):
        raise ValueError('Nonfinite certificate arithmetic')
    tau = 1e-10 * (1 + float(np.max(d2_mid)))
    stable = min(margins) > tau
    support = max_distance <= radius
    return dict(nearest_label=k, certified_label=k if stable and support else None,
                min_squared_margin=min(margins), tolerance=tau,
                worst_distance=max_distance, support=support,
                status='certified_geometry' if stable and support else ('out_of_support' if not support else 'boundary_ambiguous'))


def mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Invalid MMD input')
    def kernel(a, b):
        return np.exp(-((a[:, None] - b[None, :]) ** 2).sum(axis=2) / (2 * sigma ** 2))
    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())


def adjusted_rand(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError('Two equal label vectors required')
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    tab = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(tab, (ia, ib), 1)
    pair = lambda z: float(np.sum(z * (z - 1) / 2))
    cells, rows, cols = pair(tab), pair(tab.sum(1)), pair(tab.sum(0))
    expected = rows * cols / (len(a) * (len(a) - 1) / 2)
    denom = (rows + cols) / 2 - expected
    return 1.0 if denom == 0 else (cells - expected) / denom


def self_test() -> dict:
    rng = np.random.default_rng(1729)
    checks = {}
    # Exhaustive discrete completions: not an empirical municipality test.
    completions = 0
    for n in range(1, 8):
        for q in range(n + 1):
            observed = np.arange(n - q, dtype=float)[:, None]
            lo, hi = median_bounds(observed, n)
            for fill in itertools.product((-10., 0., 10.), repeat=q):
                full = np.concatenate((observed[:, 0], np.asarray(fill)))
                truth = np.median(full)
                assert lo[0] <= truth <= hi[0]
                completions += 1
    checks['missing_median_completions'] = completions
    # Compare affine lower bounds with all 2**5 vertices of 250 boxes.
    corner_checks = 0
    for _ in range(250):
        c = rng.normal(size=(4, 5)); lo = rng.normal(size=5); hi = lo + rng.uniform(0, .8, size=5)
        r = classify_box(lo, hi, c)
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        k = r['nearest_label']
        d2 = ((corners[:, None] - c[None, :]) ** 2).sum(2)
        true_min = min(np.min(d2[:, j] - d2[:, k]) for j in range(4) if j != k)
        assert np.isclose(r['min_squared_margin'], true_min, atol=1e-12)
        assert np.isclose(r['worst_distance'], np.sqrt(d2[:, k].max()), atol=1e-12)
        if r['certified_label'] is not None:
            assert np.all(np.argmin(d2, axis=1) == k)
        corner_checks += len(corners)
    checks['boxes_and_vertices'] = [250, corner_checks]
    # Translation invariance at the monthly stage and a sparse genuine change.
    baseline = rng.normal(size=(12, 101, 5)); shifts = rng.normal(size=(12, 1, 5))
    current = baseline + shifts
    anchor = np.median(current - baseline, axis=1)
    relative = current - anchor[:, None]
    error = float(np.max(np.abs(relative - baseline)))
    assert error < 2e-14
    sparse = current.copy(); sparse[:, :20, 0] += 2
    sparse_anchor = np.median(sparse - baseline, axis=1)
    assert np.allclose(sparse_anchor, shifts[:, 0], atol=1e-14)
    assert np.allclose((sparse - sparse_anchor[:, None] - baseline)[:, :20, 0], 2)
    checks['translation_max_abs_error'] = error
    checks['sparse_shift_20_of_101_preserved'] = True
    # Annual order-statistic monotonicity for 100 random completions.
    for _ in range(100):
        lo = rng.normal(size=(12, 5)); hi = lo + rng.uniform(size=(12, 5))
        a, b = annual_bounds(lo, hi)
        y = lo + rng.uniform(size=(12, 5)) * (hi - lo)
        assert np.all(a <= np.median(y, axis=0)) and np.all(np.median(y, axis=0) <= b)
    checks['annual_box_completions'] = 100
    # The exported annual representative need not be the annual box midpoint.
    report_point = np.array([.1])
    nonmid = classify_box(np.array([0.]), np.array([3.]), np.array([[0.], [2.]]), point=report_point)
    assert nonmid['nearest_label'] == 0 and nonmid['certified_label'] is None
    checks['nonmidpoint_representative_not_replaced'] = True
    x = np.repeat([0., 0., 2.], 4)
    shift = np.repeat([0., 2., 2.], 4)
    correct, wrong = float(np.median(x - shift)), float(np.median(x) - np.median(shift))
    assert correct == 0 and wrong == -2
    checks['annual_noncommutation'] = dict(correct=correct, wrong=wrong)
    # Harmful ARI increase in a fully specified synthetic absolute-target world.
    centers = np.array([-3., -1., 1., 3.])[:, None]
    old = np.array([-3.4, -2.6, -1.4, -.6, .6, 1.4, 2.6, 3.4])[:, None]
    lab = lambda p: ((p[:, None] - centers[None, :]) ** 2).sum(2).argmin(1)
    before, absolute, corrected = lab(old), lab(old + 1), lab(old + 1 - 1)
    assert np.count_nonzero(before != absolute) == 3
    checks['synthetic_ari_counterexample'] = dict(before=before.tolist(), absolute=absolute.tolist(),
        corrected=corrected.tolist(), absolute_ARI=adjusted_rand(before, absolute), relative_ARI=adjusted_rand(before, corrected))
    # Median-of-pairs differs from population median difference.
    a, b = np.array([0., 10., 11.]), np.array([9., 10., 20.])
    assert np.median(b-a) == 9 and np.median(b)-np.median(a) == 0
    checks['paired_vs_cross_section_medians'] = [9., 0.]
    # Distribution-preserving ID permutation still moves entities.
    permuted = old[::-1]
    assert abs(mmd2(old, permuted, 1.)) < 1e-14
    assert np.max(np.abs(old - permuted)) > 1
    checks['permutation_marginal_mmd_zero_paired_motion_nonzero'] = True
    symmetric = np.array([-2., -1., 0., 1., 2.])[:, None]
    assert np.median(1.5*symmetric - symmetric) == 0
    shape_stat = mmd2(symmetric, 1.5*symmetric, 1.)
    assert shape_stat > 0
    checks['synthetic_shape_only_mmd2'] = shape_stat
    joint_a = np.array([[-1., -1.], [1., 1.]])
    joint_b = np.array([[-1., 1.], [1., -1.]])
    assert np.all(np.median(joint_b-joint_a, axis=0) == 0)
    assert mmd2(joint_a, joint_b, 1.) > 1e-10
    checks['dependence_only_shift_with_zero_paired_median'] = True
    # No occupancy quotas and abstention at an exact decision boundary.
    assert np.all(lab(np.full((100, 1), -3.)) == 0)
    assert classify_box(np.array([-2.]), np.array([-2.]), centers)['certified_label'] is None
    checks['no_occupancy_quota_and_boundary_abstention'] = True
    # Common nominal rescaling is invisible with level weight zero.
    v = np.exp(rng.normal(size=(20, 5))); t = np.exp(rng.normal(size=20))
    f1 = frozen_features(v, t, np.zeros(5), np.ones(5))
    f2 = frozen_features(10*v, 10*t, np.zeros(5), np.ones(5))
    assert np.allclose(f1, f2, atol=1e-14)
    checks['common_nominal_rescaling_invariant'] = True
    return dict(status='passed', scope='synthetic_kernel_checks_only_not_panel_validation',
                python=platform.python_version(), numpy=np.__version__, seed=1729, checks=checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('This mathematical kernel only exposes --self-test as a CLI')
    print(json.dumps(self_test(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
