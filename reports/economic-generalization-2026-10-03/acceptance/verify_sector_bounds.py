"""Independent sharp-bound/denominator controls against the actual sector API."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sbercluster.economic_sector_bounds import squared_loss_difference_bounds as bounds
OUT = Path(__file__).resolve().parent


def score(y, a, b, regions):
    # Direct squared errors, deliberately no algebraic endpoint formula.
    d = (y - b) ** 2 - (y - a) ** 2
    return float(np.mean([d[regions == r].mean() for r in np.unique(regions)])), float(d.mean())


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid observed data or missing prediction accepted')


def main():
    # Entire second region missing: fixed equal-region lower=0 even though
    # observed-only advantage is1; municipality weighting lower=-.5 differs.
    regions = np.r_[np.repeat(1, 3), np.repeat(2, 9)]
    y = np.r_[np.zeros(3), np.repeat(np.nan, 9)]
    a, b = np.r_[np.zeros(3), np.ones(9)], np.r_[np.ones(3), np.zeros(9)]
    r = bounds(y, a, b, regions)
    assert [r['equal_region_lower'], r['equal_region_upper']] == [0., 1.]
    assert [r['municipality_lower'], r['municipality_upper']] == [-.5, 1.]
    assert r['eligible_n'] == 12 and r['eligible_regions'] == 2 and r['observed_regions'] == 1
    assert r['regions_with_no_observed_target'] == [2]
    assert r['conclusion'] == 'not_identified_under_arbitrary_missingness'
    assert r['per_region'][0]['observed_mean_loss_difference'] == 1.
    rng = np.random.default_rng(132871)
    checked = 0
    for size in [5, 7]:
        region = np.array([1, 1, 2, 3, 3, 3, 3][:size])
        yy = rng.uniform(size=size); missing = np.arange(size) % 2 == 0; yy[missing] = np.nan
        aa, bb = rng.uniform(size=size), rng.uniform(size=size)
        result = bounds(yy, aa, bb, region)
        actual = []
        for fill in itertools.product([0., 1.], repeat=int(missing.sum())):
            complete = yy.copy(); complete[missing] = fill
            actual.append(score(complete, aa, bb, region)); checked += 1
        values = np.asarray(actual)
        np.testing.assert_allclose([result['equal_region_lower'], result['equal_region_upper']], [values[:, 0].min(), values[:, 0].max()], rtol=0, atol=1e-15)
        np.testing.assert_allclose([result['municipality_lower'], result['municipality_upper']], [values[:, 1].min(), values[:, 1].max()], rtol=0, atol=1e-15)
        reverse = bounds(yy, bb, aa, region)
        np.testing.assert_allclose([reverse['equal_region_lower'], reverse['equal_region_upper']], [-result['equal_region_upper'], -result['equal_region_lower']], rtol=0, atol=1e-15)
        order = rng.permutation(size)
        shuffled = bounds(yy[order], aa[order], bb[order], region[order])
        np.testing.assert_allclose([shuffled['equal_region_lower'], shuffled['equal_region_upper']], [result['equal_region_lower'], result['equal_region_upper']], rtol=0, atol=1e-15)
    known = np.array([.1, .2, .8, .3]); aa = np.array([.2, .3, .9, .4]); bb = np.array([.3, .4, .7, .5]); rr = np.array([1, 2, 2, 2])
    exact = bounds(known, aa, bb, rr)
    truth, municipality = score(known, aa, bb, rr)
    np.testing.assert_allclose([exact['equal_region_lower'], exact['equal_region_upper']], [truth, truth], rtol=0, atol=1e-15)
    rejected(lambda: bounds(np.array([1.01]), np.array([.5]), np.array([.5]), np.array([1])))
    rejected(lambda: bounds(np.array([np.nan]), np.array([np.nan]), np.array([.5]), np.array([1])))
    rejected(lambda: bounds(np.array([np.inf]), np.array([.5]), np.array([.5]), np.array([1])))
    report = {'status': 'PASS', 'endpoint_fills_direct_squared_loss': checked, 'allmissing_region_fixed_denominator': 'PASS',
              'weighting_distinction': {'equal_region_bounds': [0., 1.], 'municipality_bounds': [-.5, 1.], 'observed_only_gain': 1.},
              'allobserved_collapse_reversal_permutation': 'PASS', 'illegal_outcome_or_prediction_rejected': 'PASS',
              'scope': 'Sharp finite eligible-cohort identification bounds, not confidence intervals or future validity',
              'source_sha256': hashlib.sha256((ROOT / 'sbercluster/economic_sector_bounds.py').read_bytes()).hexdigest(),
              'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'sector-bounds-controls.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
