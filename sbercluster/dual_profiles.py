"""Frozen absolute profiles plus a sealed, explicitly transductive paired view.

The reference median is a descriptive normalization, not a causal common shock.
No balancing, support probability, measurement-error or territorial-transition
guarantee is inferred. Exact missing-reference bounds concern arbitrary missing
values; geometric certificates are conditional on the declared feature box.
"""
from copy import deepcopy
import hashlib
import json
from numbers import Integral
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .features import transform_frozen
from .io import CATEGORIES, TOTAL
from .selection_artifacts import validate_artifact
from .selection_frontier import predict_frontier
from .temporal_profiles import _periods, _validate_source_identity

LOCK_REVISION = 'paired_reference_lock_v1'
PACKET_REVISION = 'paired_reference_packet_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                   allow_nan=False).encode('utf-8')).hexdigest()


def _seal(value):
    result = deepcopy(value)
    result['content_sha256'] = _digest(result)
    return result


def _verify(value, revision):
    if not isinstance(value, dict) or value.get('revision') != revision:
        raise ValueError('Unsupported sealed reference revision')
    contents = {k: v for k, v in value.items() if k != 'content_sha256'}
    if value.get('content_sha256') != _digest(contents):
        raise ValueError('Reference content SHA256 mismatch')


def _panel_keys(panel):
    if panel.empty or 'entity_id' not in panel or 'period' not in panel:
        raise ValueError('Nonempty entity/month panel required')
    if (panel.entity_id.isna().any() or not panel.entity_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or panel.duplicated(['entity_id', 'period']).any()):
        raise ValueError('Invalid or conflicting duplicate entity/month keys')
    _periods(panel.period)
    _validate_source_identity(panel, 'source_territory_id')


def _historical_rule(artifact):
    validate_artifact(artifact)
    rule = artifact['rule']
    if (artifact['feature_scaler']['mode'] != 'log_ratios_to_total'
            or artifact['feature_scaler'].get('level_weight', 0.) != 0.
            or np.asarray(rule['centers']).shape != (4, 5)
            or not np.array_equal(np.asarray(rule['transform']), np.eye(5))
            or not np.array_equal(np.asarray(artifact['pre_aggregation_transform']), np.eye(5))
            or not np.array_equal(np.asarray(rule['biases']), np.zeros(4))):
        raise ValueError('Dual contract currently requires frozen zero-bias identity-space K4')
    return rule


def build_reference_lock(base_panel, frozen_artifact, source_provenance=None):
    """Lock complete2023 paired reference; ignore future rows before validation.

    Reference selection remains retrospectively complete if that was its origin.
    Baseline transformed observations are needed for pairing and belong in the
    private deployment artifact, not a public raw-data/evidence archive.
    """
    rule = _historical_rule(frozen_artifact)
    cutoff = frozen_artifact['feature_scaler']['calibration_end']
    _, parsed = _periods([cutoff]); year = int(parsed.year[0])
    if parsed.month[0] != 12:
        raise ValueError('Complete calendar baseline required')
    base = base_panel[base_panel.period.str.startswith(str(year) + '-') & (base_panel.period <= cutoff)].copy()
    _panel_keys(base)
    ids = sorted(base.entity_id.unique().tolist())
    expected = [f'{year}-{m:02d}-01' for m in range(1, 13)]
    features = []
    for period in expected:
        rows = base[base.period == period].sort_values('entity_id')
        if rows.entity_id.tolist() != ids:
            raise ValueError('Every reference entity needs all12 baseline months')
        features.append(transform_frozen(rows, frozen_artifact['feature_scaler']).tolist())
    regions = []
    for entity in ids:
        rows = base[base.entity_id == entity]
        values = rows.region_code.dropna().astype(str).unique() if 'region_code' in rows else []
        if len(values) > 1:
            raise ValueError('Baseline region changes require explicit reconciliation')
        regions.append(str(values[0]) if len(values) else 'unknown')
    return _seal({'revision': LOCK_REVISION, 'baseline_year': year, 'feature_order': list(CATEGORIES),
        'entity_ids': ids, 'entity_ids_sha256': _digest(ids), 'regions': regions,
        'baseline_features': features, 'frozen_artifact': deepcopy(frozen_artifact),
        'frozen_artifact_content_sha256': _digest(frozen_artifact),
        'source_provenance': deepcopy(source_provenance or {}),
        'semantics': {'scope': 'descriptive_paired_ID_records', 'causal_interpretation': 'unknown',
                      'territory_boundary_continuity_verified': False,
                      'confirmed_transitions_supported': False}})


def validate_reference_lock(lock):
    _verify(lock, LOCK_REVISION)
    _historical_rule(lock['frozen_artifact'])
    ids = lock['entity_ids']; regions = lock['regions']; x = np.asarray(lock['baseline_features'], float)
    year = lock['baseline_year']
    if (not ids or ids != sorted(set(ids)) or any(not isinstance(v, str) or not v.strip() for v in ids)
            or len(regions) != len(ids) or x.shape != (12, len(ids), 5) or not np.isfinite(x).all()
            or not isinstance(year, Integral) or isinstance(year, bool)
            or lock['frozen_artifact']['feature_scaler']['calibration_end'] != f'{year}-12-01'
            or lock['entity_ids_sha256'] != _digest(ids)
            or lock['frozen_artifact_content_sha256'] != _digest(lock['frozen_artifact'])
            or lock.get('feature_order') != list(CATEGORIES)
            or lock.get('semantics', {}).get('confirmed_transitions_supported') is not False):
        raise ValueError('Invalid paired reference lock contract')
    return lock


def median_missing_bounds(observed, reference_n):
    """Sharp coordinatewise bounds for full median, missing values unrestricted."""
    x = np.asarray(observed, float)
    if (x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all()
            or not isinstance(reference_n, Integral) or isinstance(reference_n, bool)
            or reference_n < 1 or len(x) > reference_n):
        raise ValueError('Invalid observed/reference cardinality')
    missing = int(reference_n) - len(x)
    lower = np.median(np.concatenate([np.full((missing, x.shape[1]), -np.inf), x]), axis=0)
    upper = np.median(np.concatenate([x, np.full((missing, x.shape[1]), np.inf)]), axis=0)
    return lower, upper


def _json_bounds(array):
    return [float(v) if np.isfinite(v) else None for v in array]


def build_reference_packet(current_panel, lock, period, source_provenance=None):
    """New content-addressed packet; no mutation/revision of an existing packet.

    Missing/invalid reference records remain in the fixed denominator. Duplicate
    keys refuse the packet. Valid pairs are aligned by ID, never row position.
    Complete paired median and difference of population medians remain distinct.
    """
    validate_reference_lock(lock)
    values, parsed = _periods([period]); month = int(parsed.month[0])
    _panel_keys(current_panel)
    selected = current_panel[current_panel.period == period].copy()
    ids = lock['entity_ids']; id_to_index = {v: i for i, v in enumerate(ids)}
    selected = selected[selected.entity_id.isin(id_to_index)].sort_values('entity_id')
    current_reference_rows_sha256 = _digest(selected.astype(str).to_dict('records'))
    numeric = selected[CATEGORIES + [TOTAL]].to_numpy(float)
    valid = np.isfinite(numeric).all(axis=1) & (numeric > 0).all(axis=1)
    invalid_ids = selected.entity_id[~valid].tolist()
    selected = selected[valid]
    observed_ids = selected.entity_id.tolist()
    baseline = np.asarray(lock['baseline_features'])[month - 1]
    if len(selected):
        current = transform_frozen(selected, lock['frozen_artifact']['feature_scaler'])
        delta = current - baseline[[id_to_index[v] for v in observed_ids]]
    else:
        current, delta = np.empty((0, 5)), np.empty((0, 5))
    lower, upper = median_missing_bounds(delta, len(ids))
    complete = len(observed_ids) == len(ids)
    finite = bool(np.isfinite(lower).all() and np.isfinite(upper).all())
    point = np.median(delta, axis=0) if complete else None
    representative = point if complete else (lower / 2. + upper / 2. if finite else None)
    regional = {}
    observed_set = set(observed_ids)
    for entity, region in zip(ids, lock['regions']):
        item = regional.setdefault(region, {'reference_n': 0, 'observed_valid_n': 0})
        item['reference_n'] += 1; item['observed_valid_n'] += int(entity in observed_set)
    for item in regional.values(): item['coverage'] = item['observed_valid_n'] / item['reference_n']
    region_mismatch = 0
    if 'region_code' in selected:
        for entity, region in zip(selected.entity_id, selected.region_code):
            region_mismatch += int(str(region) != lock['regions'][id_to_index[entity]])
    audit = {}
    for name in CATEGORIES + [TOTAL]:
        logs = np.log(selected[name].to_numpy(float))
        audit[name] = np.quantile(logs, [.1, .5, .9]).tolist() if len(logs) else None
    return _seal({'revision': PACKET_REVISION, 'reference_lock_sha256': lock['content_sha256'],
        'period': period, 'scope': 'transductive_fixed_ID_paired_median_change',
        'current_reference_rows_sha256': current_reference_rows_sha256,
        'source_provenance': deepcopy(source_provenance or {}),
        'reference_n': len(ids), 'observed_valid_n': len(observed_ids), 'coverage': len(observed_ids) / len(ids),
        'observed_ids_sha256': _digest(observed_ids), 'missing_ids': sorted(set(ids) - observed_set),
        'invalid_ids': invalid_ids, 'regional_coverage': regional, 'region_code_mismatch_n': region_mismatch,
        'point_paired_median': point.tolist() if complete else None,
        'representative_correction': representative.tolist() if representative is not None else None,
        'correction_lower': _json_bounds(lower), 'correction_upper': _json_bounds(upper),
        'lower_unbounded': (~np.isfinite(lower)).tolist(), 'upper_unbounded': (~np.isfinite(upper)).tolist(),
        'available_case_paired_median': np.median(delta, axis=0).tolist() if len(delta) else None,
        'population_median_difference': (np.median(current, axis=0) - np.median(baseline, axis=0)).tolist() if complete else None,
        'log_input_quantiles_valid_reference': audit,
        'status': 'complete_reference' if complete else ('partial_reference_bounded' if finite else 'reference_unbounded'),
        'measurement_epsilon': 0., 'measurement_error_known': False,
        'causal_interpretation': 'unknown', 'territory_transition_status': 'unresolved'})


def validate_reference_packet(packet, lock, period=None):
    _verify(packet, PACKET_REVISION)
    if (packet.get('reference_lock_sha256') != lock['content_sha256']
            or (period is not None and packet.get('period') != period)
            or packet.get('reference_n') != len(lock['entity_ids'])):
        raise ValueError('Packet reference/period mismatch')
    _periods([packet.get('period')])
    try:
        n = packet['observed_valid_n']; total = len(lock['entity_ids'])
        missing = packet['missing_ids']; invalid = packet['invalid_ids']
        if (not isinstance(n, Integral) or isinstance(n, bool) or not 0 <= n <= total
                or packet['coverage'] != n / total or missing != sorted(set(missing))
                or len(missing) != total - n or not set(missing) <= set(lock['entity_ids'])
                or len(invalid) != len(set(invalid)) or not set(invalid) <= set(missing)
                or packet['observed_ids_sha256'] != _digest(sorted(set(lock['entity_ids']) - set(missing)))
                or packet['measurement_epsilon'] != 0. or packet['measurement_error_known'] is not False
                or packet['territory_transition_status'] != 'unresolved'):
            raise ValueError('Invalid reference coverage/status')
        lower_raw, upper_raw = packet['correction_lower'], packet['correction_upper']
        low_unbounded, up_unbounded = packet['lower_unbounded'], packet['upper_unbounded']
        if any(len(v) != 5 for v in (lower_raw, upper_raw, low_unbounded, up_unbounded)):
            raise ValueError('Correction vectors must have five coordinates')
        if (any(type(v) is not bool for v in low_unbounded + up_unbounded)
                or any((v is None) != flag for v, flag in zip(lower_raw, low_unbounded))
                or any((v is None) != flag for v, flag in zip(upper_raw, up_unbounded))):
            raise ValueError('Unbounded correction flags disagree')
        lower = np.asarray([-np.inf if v is None else v for v in lower_raw], float)
        upper = np.asarray([np.inf if v is None else v for v in upper_raw], float)
        if np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any():
            raise ValueError('Invalid correction interval')
        finite = np.isfinite(lower).all() and np.isfinite(upper).all()
        representative = packet['representative_correction']; point = packet['point_paired_median']
        expected_status = 'complete_reference' if n == total else ('partial_reference_bounded' if finite else 'reference_unbounded')
        if packet['status'] != expected_status or (representative is None) == bool(finite):
            raise ValueError('Correction status/representative mismatch')
        if representative is not None:
            representative = np.asarray(representative, float)
            if (representative.shape != (5,) or not np.isfinite(representative).all()
                    or (representative < lower).any() or (representative > upper).any()):
                raise ValueError('Invalid correction representative')
        if n == total:
            point = np.asarray(point, float)
            if point.shape != (5,) or not np.isfinite(point).all() or not np.array_equal(point, lower) or not np.array_equal(point, upper) or not np.array_equal(point, representative):
                raise ValueError('Complete paired correction is not a point')
        elif point is not None:
            raise ValueError('Missing reference cannot identify full paired median')
        expected_regions = {}
        missing_set = set(missing)
        for entity, region in zip(lock['entity_ids'], lock['regions']):
            item = expected_regions.setdefault(region, {'reference_n': 0, 'observed_valid_n': 0})
            item['reference_n'] += 1; item['observed_valid_n'] += int(entity not in missing_set)
        for item in expected_regions.values(): item['coverage'] = item['observed_valid_n'] / item['reference_n']
        if packet['regional_coverage'] != expected_regions:
            raise ValueError('Invalid regional reference coverage')
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Malformed reference packet schema') from error
    return packet


def classify_profile_box(point, lower, upper, centers):
    """Conditional nearest-center certificate over a finite coordinate box.

    No empirical radius/coverage policy is imposed. A numerical geometric label
    never certifies measurement error, economic identity or geography continuity.
    """
    point, lower, upper, centers = [np.asarray(x, float) for x in (point, lower, upper, centers)]
    if (point.ndim != 1 or lower.shape != point.shape or upper.shape != point.shape
            or centers.ndim != 2 or centers.shape[1] != len(point) or len(centers) < 2
            or not all(np.isfinite(x).all() for x in (point, lower, upper, centers))
            or (lower > upper).any() or (point < lower).any() or (point > upper).any()):
        raise ValueError('Invalid finite profile box')
    squared = np.sum((centers - point) ** 2, axis=1)
    chosen = int(squared.argmin()); margins = []
    for other in range(len(centers)):
        if other == chosen: continue
        slope = 2 * (centers[chosen] - centers[other])
        endpoint = np.where(slope >= 0, lower, upper)
        margins.append(float(slope @ endpoint + centers[other] @ centers[other] - centers[chosen] @ centers[chosen]))
    midpoint = lower / 2 + upper / 2
    tolerance = 1.e-10 * (1. + np.max(np.sum((centers - midpoint) ** 2, axis=1)))
    worst = np.sqrt(np.sum(np.maximum((lower - centers[chosen]) ** 2, (upper - centers[chosen]) ** 2)))
    if not np.isfinite(squared).all() or not np.isfinite(margins).all() or not np.isfinite(worst):
        raise ValueError('Certificate arithmetic overflow')
    stable = min(margins) > tolerance
    return {'nearest_label': chosen, 'geometric_label': chosen if stable else None,
            'min_squared_margin': min(margins), 'tolerance': float(tolerance), 'worst_distance': float(worst),
            'status': 'conditional_geometry_stable' if stable else 'boundary_ambiguous'}


def gaussian_mmd2(first, second, bandwidth):
    """Biased Gaussian-kernel V-statistic, descriptive with no p-value promise."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]
            or not len(first) or not len(second) or not np.isfinite(first).all() or not np.isfinite(second).all()
            or not np.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('Finite aligned samples and positive bandwidth required')
    def average_kernel(a, b):
        return float(np.exp(-cdist(a, b, metric='sqeuclidean') / (2 * bandwidth ** 2)).mean())
    return average_kernel(first, first) + average_kernel(second, second) - 2 * average_kernel(first, second)


def predict_dual_panel(panel, lock, year, packets=None):
    """Annual descriptive A/R; sealed context makes scoring batch-independent.

    Missing or wrong-month packets refuse R while valid A remains available.
    Incomplete entity years are returned explicitly with no annual label.
    Geometry certificates refer to exact missing-reference boxes at epsilon0.
    They do not establish substantive or future validity.
    """
    validate_reference_lock(lock); _panel_keys(panel)
    if isinstance(year, bool) or not isinstance(year, Integral) or not 1900 <= year <= 9998:
        raise ValueError('Supported integer year required')
    selected = panel[panel.period.str.startswith(f'{year}-')].sort_values(['entity_id', 'period'])
    if selected.empty: raise ValueError('Requested year absent')
    dates = [f'{year}-{m:02d}-01' for m in range(1, 13)]
    artifact = lock['frozen_artifact']; centers = np.asarray(artifact['rule']['centers'])
    packets = packets or {}; packet_by_month = {}; failures = {}
    for period in dates:
        packet = packets.get(period)
        if packet is None: failures[period] = 'packet_absent'; continue
        try: validate_reference_packet(packet, lock, period)
        except ValueError as error: failures[period] = str(error); continue
        if packet['representative_correction'] is None: failures[period] = 'reference_unbounded'; continue
        packet_by_month[period] = packet
    rows = []
    for entity, records in selected.groupby('entity_id', sort=True):
        result = {'entity_id': entity, 'year': int(year), 'reference_lock_sha256': lock['content_sha256'],
                  'month_mask': [p in set(records.period) for p in dates],
                  'new_entity': entity not in set(lock['entity_ids']),
                  'transition_status': 'unresolved', 'causal_interpretation': 'unknown',
                  'absolute_label': None, 'relative_label': None, 'absolute_geometric_label': None,
                  'relative_geometric_label': None, 'absolute_profile': None, 'relative_profile': None}
        if records.period.tolist() != dates:
            result['status'] = 'incomplete_entity_year'; rows.append(result); continue
        x = transform_frozen(records, artifact['feature_scaler'])
        absolute = np.median(x, axis=0)
        a_certificate = classify_profile_box(absolute, absolute, absolute, centers)
        result.update({'absolute_label': int(predict_frontier(absolute[None], artifact['rule'])[0]),
                       'absolute_profile': absolute.tolist(), 'absolute_geometric_label': a_certificate['geometric_label'],
                       'absolute_certificate': a_certificate,
                       'log_input_annual_medians': {name: float(np.median(np.log(records[name].to_numpy(float)))) for name in CATEGORIES + [TOTAL]}})
        if failures:
            result.update({'status': 'relative_unavailable', 'relative_reasons': dict(failures)}); rows.append(result); continue
        correction = np.asarray([packet_by_month[p]['representative_correction'] for p in dates])
        correction_lower = np.asarray([packet_by_month[p]['correction_lower'] for p in dates], float)
        correction_upper = np.asarray([packet_by_month[p]['correction_upper'] for p in dates], float)
        relative = np.median(x - correction, axis=0)
        lower = np.median(x - correction_upper, axis=0); upper = np.median(x - correction_lower, axis=0)
        certificate = classify_profile_box(relative, lower, upper, centers)
        result.update({'status': 'descriptive_dual_profile', 'relative_label': certificate['nearest_label'],
                       'relative_geometric_label': certificate['geometric_label'], 'relative_certificate': certificate,
                       'relative_profile': relative.tolist(), 'relative_lower': lower.tolist(), 'relative_upper': upper.tolist(),
                       'effective_removed': (absolute - relative).tolist(),
                       'packet_sha256': [packet_by_month[p]['content_sha256'] for p in dates],
                       'reference_complete_all_months': all(packet_by_month[p]['status'] == 'complete_reference' for p in dates)})
        rows.append(result)
    return pd.DataFrame(rows)
