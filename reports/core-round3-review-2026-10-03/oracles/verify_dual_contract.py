"""Independent end-to-end paired-reference/partial-coverage controls, no fit."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sbercluster.io import CATEGORIES, TOTAL
from sbercluster.selection_artifacts import make_artifact
from sbercluster.dual_profiles import (build_reference_lock, build_reference_packet,
    predict_dual_panel, median_missing_bounds, classify_profile_box, _seal)
OUT = Path(__file__).resolve().parent


def panel(year, first_coordinates, territories=(1, 2, 3)):
    rows = []
    for month in range(1, 13):
        for territory, coordinate in zip(territories, first_coordinates):
            x = np.array([coordinate, -2., -2., -2., -2.])
            row = dict(zip(CATEGORIES, 10. * np.exp(x * np.sqrt(5))))
            row.update({TOTAL: 10., 'entity_id': f'tid_{territory}', 'territory_id': territory,
                        'period': f'{year}-{month:02d}-01', 'region_code': territory + 10})
            rows.append(row)
    return pd.DataFrame(rows)


def order_statistic_bounds(observed, n):
    # Independent rank formula, without inserting infinity rows or calling
    # median: missing values can occupy the extreme ranks in either direction.
    x = np.sort(np.asarray(observed), axis=0); missing = n - len(x)
    ranks = [(n + 1) // 2] if n % 2 else [n // 2, n // 2 + 1]
    lower, upper = [], []
    for rank in ranks:
        lower.append(x[rank - missing - 1] if rank > missing else np.full(x.shape[1], -np.inf))
        upper.append(x[rank - 1] if rank <= len(x) else np.full(x.shape[1], np.inf))
    return np.mean(lower, axis=0), np.mean(upper, axis=0)


def main():
    scaler = {'mode': 'log_ratios_to_total', 'level_weight': 0., 'ratio_center': [0.] * 5,
              'ratio_iqr': [1.] * 5, 'calibration_end': '2023-12-01'}
    centers = np.zeros((4, 5)); centers[:, 0] = [0., 2., 4., 6.]
    artifact = make_artifact({'revision': 'prototype_frontier_v1', 'transform': np.eye(5).tolist(),
        'centers': centers.tolist(), 'biases': [0.] * 4}, scaler, provenance={'scope': 'independent_synthetic_control'})
    base, current = panel(2023, [0., 10., 10.1]), panel(2024, [.1, .2, 10.2])
    lock = build_reference_lock(base, artifact)
    # Future valid observations cannot affect train-only reference content.
    future_changed = current.copy(); future_changed[CATEGORIES] *= np.exp(2.)
    assert lock == build_reference_lock(pd.concat([base, future_changed]), artifact)
    dates = [f'2024-{m:02d}-01' for m in range(1, 13)]
    packets = {date: build_reference_packet(current, lock, date) for date in dates}
    p = packets[dates[0]]
    assert abs(p['point_paired_median'][0] - .1) < 1e-12
    assert abs(p['population_median_difference'][0] + 9.8) < 1e-12
    assert abs(p['point_paired_median'][0] - p['population_median_difference'][0]) > 9.
    assert packets[dates[0]] == build_reference_packet(current.sample(frac=1, random_state=83), lock, dates[0])
    result = predict_dual_panel(current, lock, 2024, packets).set_index('entity_id')
    assert result.absolute_label.tolist() == [0, 0, 3]
    assert result.relative_label.tolist() == [0, 0, 3]
    assert all(result.transition_status == 'unresolved') and all(result.causal_interpretation == 'unknown')
    one = current[current.entity_id == 'tid_1']
    chunk = predict_dual_panel(one, lock, 2024, packets).set_index('entity_id')
    assert chunk.loc['tid_1', 'relative_profile'] == result.loc['tid_1', 'relative_profile']
    extra = panel(2024, [7.], territories=(999,))
    larger = predict_dual_panel(pd.concat([current, extra]), lock, 2024, packets).set_index('entity_id')
    assert result.relative_profile.tolist() == larger.loc[result.index, 'relative_profile'].tolist()
    assert bool(larger.loc['tid_999', 'new_entity'])
    # 2/3 coverage can be bounded despite lying below Pro's optional80% policy.
    partial = current[current.entity_id != 'tid_1']
    partial_packets = {date: build_reference_packet(partial, lock, date) for date in dates}
    assert partial_packets[dates[0]]['reference_n'] == 3
    assert partial_packets[dates[0]]['point_paired_median'] is None
    assert partial_packets[dates[0]]['status'] == 'partial_reference_bounded'
    partial_result = predict_dual_panel(one, lock, 2024, partial_packets).iloc[0]
    assert partial_result.status == 'descriptive_dual_profile'
    assert partial_result.relative_label == 2 and partial_result.relative_geometric_label is None
    assert partial_result.absolute_label == 0
    sparse = current[current.entity_id == 'tid_3']
    sparse_packets = {date: build_reference_packet(sparse, lock, date) for date in dates}
    refused = predict_dual_panel(one, lock, 2024, sparse_packets).iloc[0]
    assert refused.status == 'relative_unavailable' and refused.absolute_label == 0 and refused.relative_label is None
    incomplete = predict_dual_panel(one.iloc[1:], lock, 2024, packets).iloc[0]
    assert incomplete.status == 'incomplete_entity_year' and incomplete.absolute_label is None
    assert sum(incomplete.month_mask) == 11
    # A corrupted sealed context must never suppress otherwise valid A.
    malformed = deepcopy(packets)
    malformed[dates[0]]['representative_correction'] = [0., 0.]
    malformed[dates[0]].pop('content_sha256')
    malformed[dates[0]] = _seal(malformed[dates[0]])
    refused_bad = predict_dual_panel(one, lock, 2024, malformed).iloc[0]
    assert refused_bad.status == 'relative_unavailable' and refused_bad.absolute_label == 0
    median_cases = 0
    for n, observed in [(3, [[-9.8], [.1]]), (4, [[2.], [4.], [8.]]), (5, [[1.], [2.]]), (6, [[-1.], [3.], [4.], [7.]])]:
        a, b = median_missing_bounds(observed, n)
        aa, bb = order_statistic_bounds(observed, n)
        np.testing.assert_array_equal(a, aa); np.testing.assert_array_equal(b, bb)
        median_cases += 1
    rng = np.random.default_rng(193700)
    checked_vertices, stable_boxes = 0, 0
    for _ in range(12):
        c = rng.normal(size=(4, 5))
        point = c[rng.integers(4)] + rng.normal(size=5) * .1
        width = rng.uniform(.001, .15, 5)
        lower, upper = point - width, point + width
        cert = classify_profile_box(point, lower, upper, c)
        vertices = np.array(list(itertools.product(*zip(lower, upper))))
        labels = ((vertices[:, None] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        if cert['geometric_label'] is not None:
            assert (labels == cert['geometric_label']).all()
            stable_boxes += 1
        checked_vertices += len(vertices)
    assert stable_boxes >= 5
    report = {'status': 'PASS', 'scope': 'Independent end-to-end adversarial A/R semantics; no data fit or causal/future guarantee',
        'paired_median': p['point_paired_median'][0], 'population_median_difference': p['population_median_difference'][0],
        'row_pairing_permutation_fixed_packet_query_batch': 'PASS', 'partial_denominator_and_below80percent': 'PASS',
        'ambiguous_box_keeps_representative_label_but_no_certificate': 'PASS', 'unbounded_missing_malformed_R_preserves_A': 'PASS',
        'incomplete_target_year': 'PASS', 'independent_order_statistic_cases': median_cases,
        'independent_certificate_vertices': checked_vertices, 'stable_boxes': stable_boxes,
        'source_sha256': hashlib.sha256((ROOT / 'sbercluster/dual_profiles.py').read_bytes()).hexdigest(),
        'oracle_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'dual-contract-review.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
