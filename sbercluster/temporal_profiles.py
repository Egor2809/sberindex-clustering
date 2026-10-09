"""Frozen seasonal/reliability profiles and separate transductive drift context.

These are exploratory descriptive consumer-profile models. A current-period
population correction is deliberately not part of their inductive inference.
Calendar effects and the full-rank reliability metric are fitted only on the
declared training year, before per-coordinate annual median aggregation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .features import make_slices, transform_frozen
from .input_contracts import _canonical_territory
from .models import require_execution

REVISION = 'frozen_temporal_profile_v1'
CALIBRATION_REVISION = 'frozen_calendar_reliability_v1'


def _matrix(features):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite feature matrix')
    return x


def _periods(periods):
    values = list(periods)
    if not values or any(not isinstance(p, str) or not re.fullmatch(r'\d{4}-\d{2}-01', p) for p in values):
        raise ValueError('Expected canonical month-start periods')
    parsed = pd.to_datetime(values, format='%Y-%m-%d', errors='raise')
    return values, parsed


def _validate_source_identity(panel, identity_mode):
    if identity_mode == 'source_territory_id':
        if 'territory_id' not in panel:
            raise ValueError('Source territory identity requires territory_id')
        if not panel.entity_id.eq('tid_' + panel.territory_id.map(_canonical_territory)).all():
            raise ValueError('Entity IDs differ from source territory IDs')


def validate_temporal_calibration(calibration, dimension=None):
    if not isinstance(calibration, dict) or calibration.get('revision') != CALIBRATION_REVISION:
        raise ValueError('Unsupported temporal calibration revision')
    periods, parsed = _periods(calibration.get('training_periods', []))
    year = calibration.get('training_year')
    if (not isinstance(year, int) or isinstance(year, bool) or periods != [f'{year}-{m:02d}-01' for m in range(1, 13)]
            or calibration.get('calibration_end') != periods[-1]):
        raise ValueError('Calibration requires one complete declared calendar year')
    offset = np.asarray(calibration.get('calendar_offsets'), dtype=float)
    metric = np.asarray(calibration.get('metric'), dtype=float)
    center = np.asarray(calibration.get('reference_location'), dtype=float)
    alpha = calibration.get('reliability')
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha not in (0., .5, 1.)
            or offset.ndim != 2 or offset.shape[0] != 12 or not offset.shape[1]
            or metric.shape != (offset.shape[1], offset.shape[1]) or center.shape != (offset.shape[1],)
            or not all(np.isfinite(a).all() for a in (offset, metric, center))
            or (dimension is not None and offset.shape[1] != dimension)
            or not np.allclose(metric, metric.T, rtol=1e-12, atol=1e-12)
            or np.linalg.eigvalsh(metric).min() <= 0
            or calibration.get('covariance_shrinkage') != .2
            or calibration.get('current_period_recalibration') is not False):
        raise ValueError('Invalid full-rank frozen calendar/reliability transform')
    return calibration


def fit_temporal_calibration(monthly_features, periods, reliability=0.):
    """Fit fixed calendar offsets and regularized within-entity noise precision.

    Reliability alpha=0 is the calendar-only ablation; .5/1 use precision to
    the alpha/2 power. A fixed 20% spherical shrinkage prevents unstable axes.
    The metric Frobenius norm is normalized to sqrt(d); no population balancing
    or evaluation-period quantities enter the fitted calibration.
    """
    x = np.asarray(monthly_features, dtype=float)
    values, parsed = _periods(periods)
    if (x.ndim != 3 or x.shape[0] != len(values) or x.shape[1] < 2 or not x.shape[2]
            or not np.isfinite(x).all() or len(set(parsed.year)) != 1
            or values != [f'{parsed.year[0]}-{m:02d}-01' for m in range(1, 13)]
            or isinstance(reliability, bool) or reliability not in (0., .5, 1.)):
        raise ValueError('Expected complete training year and prespecified reliability 0/.5/1')
    locations = np.median(x, axis=1)
    location = locations.mean(axis=0)
    offsets = locations - location
    corrected = x - offsets[:, None, :]
    residuals = corrected - corrected.mean(axis=0, keepdims=True)
    residuals = residuals.reshape(-1, x.shape[2])
    covariance = residuals.T @ residuals / len(residuals)
    variance = float(np.trace(covariance) / x.shape[2])
    if variance <= 0:
        metric = np.eye(x.shape[2])
    else:
        regularized = .8 * covariance + .2 * variance * np.eye(x.shape[2])
        eig, basis = np.linalg.eigh(regularized / variance)
        metric = (basis * eig ** (-float(reliability) / 2.)) @ basis.T
        metric *= np.sqrt(x.shape[2] / np.sum(metric ** 2))
    result = {'revision': CALIBRATION_REVISION, 'training_year': int(parsed.year[0]),
              'training_periods': values, 'calibration_end': values[-1],
              'calendar_offsets': offsets.tolist(), 'reference_location': location.tolist(),
              'metric': metric.tolist(), 'reliability': float(reliability),
              'covariance_shrinkage': .2, 'within_covariance': covariance.tolist(),
              'current_period_recalibration': False}
    return validate_temporal_calibration(result, x.shape[2])


def transform_temporal_rows(features, periods, calibration):
    """Apply saved quantities independently to every entity/month row."""
    x = _matrix(features)
    values, parsed = _periods(periods)
    validate_temporal_calibration(calibration, x.shape[1])
    if len(values) != len(x):
        raise ValueError('One canonical period per feature row is required')
    offsets = np.asarray(calibration['calendar_offsets'])
    return (x - offsets[parsed.month.to_numpy() - 1]) @ np.asarray(calibration['metric'])


def fit_temporal_model(train_panel, config, reliability=0., seed=1729):
    """Fit on declared complete calibration year; ignore future rows exactly.

    Saving training entity IDs enables honest new-entity flags, not rejection.
    Identical IDs do not establish boundary continuity.
    """
    require_execution(config, True)
    cfg = deepcopy(config)
    cutoff = cfg['features']['calibration_end']
    values, parsed = _periods([cutoff])
    year = int(parsed.year[0])
    if parsed.month[0] != 12:
        raise ValueError('Temporal model needs all12 training calendar months')
    training = train_panel[(train_panel.period <= cutoff) & train_panel.period.str.startswith(str(year) + '-')].copy()
    _validate_source_identity(training, cfg['data_contract']['identity_mode'])
    slices, scaler = make_slices(training, cfg)
    if not slices or any(ids != slices[0][1] for _, ids, _ in slices):
        raise ValueError('Training calendar requires the same entities every month')
    monthly = np.stack([features for _, _, features in slices])
    calibration = fit_temporal_calibration(monthly, [p for p, _, _ in slices], reliability)
    transformed = np.stack([transform_temporal_rows(x, [p] * len(x), calibration) for p, _, x in slices])
    profiles = np.median(transformed, axis=0)
    km = KMeans(n_clusters=4, n_init=20, random_state=seed).fit(profiles)
    distances = cdist(profiles, km.cluster_centers_)
    artifact = {'revision': REVISION, 'feature_scaler': scaler, 'calibration': calibration,
                'centers': km.cluster_centers_.tolist(), 'k': 4, 'seed': int(seed), 'n_init': 20,
                'aggregation': 'transform_each_calendar_month_then_coordinate_median_of12',
                'identity_mode': cfg['data_contract']['identity_mode'],
                'training_entity_ids': slices[0][1],
                'training_entity_ids_sha256': hashlib.sha256('\n'.join(slices[0][1]).encode()).hexdigest(),
                'distance_q99': float(np.quantile(distances.min(axis=1), .99)),
                'prediction_scope': 'inductive_descriptive_consumer_profiles',
                'territory_boundary_continuity_verified': False,
                'default_promotion': False}
    validate_temporal_model(artifact)
    return artifact


def validate_temporal_model(artifact):
    if not isinstance(artifact, dict) or artifact.get('revision') != REVISION:
        raise ValueError('Unsupported temporal model revision')
    centers = _matrix(artifact.get('centers'))
    validate_temporal_calibration(artifact.get('calibration'), centers.shape[1])
    if (artifact.get('k') != 4 or len(centers) != 4
            or artifact.get('aggregation') != 'transform_each_calendar_month_then_coordinate_median_of12'
            or artifact.get('feature_scaler', {}).get('calibration_end') != artifact['calibration']['calibration_end']
            or not isinstance(artifact.get('distance_q99'), (int, float))
            or not np.isfinite(artifact['distance_q99']) or artifact['distance_q99'] < 0
            or not isinstance(artifact.get('training_entity_ids'), list)
            or len(artifact['training_entity_ids']) < 4
            or any(not isinstance(v, str) or not v.strip() for v in artifact['training_entity_ids'])
            or len(set(artifact['training_entity_ids'])) != len(artifact['training_entity_ids'])
            or artifact.get('training_entity_ids_sha256') != hashlib.sha256('\n'.join(artifact['training_entity_ids']).encode()).hexdigest()
            or artifact.get('identity_mode') not in ('source_territory_id', 'verified_registry_id')
            or artifact.get('prediction_scope') != 'inductive_descriptive_consumer_profiles'):
        raise ValueError('Invalid temporal model contract')
    return artifact


def predict_temporal_features(profiles, artifact):
    validate_temporal_model(artifact)
    x = _matrix(profiles)
    centers = np.asarray(artifact['centers'])
    if x.shape[1] != centers.shape[1]:
        raise ValueError('Temporal feature dimension differs')
    distances = cdist(x, centers)
    if not np.isfinite(distances).all():
        raise ValueError('Distance overflow')
    return distances.argmin(axis=1).astype(np.int32)


def predict_temporal_panel(panel, artifact, year):
    """Predict complete years, independent of unrelated entities or row order.

    distance_extrapolation marks a descriptive training-distance flag, without a
    calibrated probability, abstention guarantee or claim of prospective validity.
    """
    validate_temporal_model(artifact)
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)) or not 1900 <= year <= 9998:
        raise ValueError('Expected supported integer calendar year')
    if panel.empty or 'entity_id' not in panel or 'period' not in panel:
        raise ValueError('Expected nonempty entity-month panel')
    if (panel.entity_id.isna().any() or not panel.entity_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
            or panel.duplicated(['entity_id', 'period']).any()):
        raise ValueError('Expected unique nonempty string entity-month keys')
    _validate_source_identity(panel, artifact['identity_mode'])
    _, parsed = _periods(panel.period)
    selected = panel[parsed.year == year].sort_values(['entity_id', 'period'])
    if selected.empty:
        raise ValueError('Requested year is absent')
    expected = [f'{year}-{m:02d}-01' for m in range(1, 13)]
    ids, profiles = [], []
    for entity, rows in selected.groupby('entity_id', sort=True):
        if rows.period.tolist() != expected:
            raise ValueError('Each entity requires all12 calendar months: ' + entity)
        x = transform_frozen(rows, artifact['feature_scaler'])
        transformed = transform_temporal_rows(x, rows.period, artifact['calibration'])
        ids.append(entity)
        profiles.append(np.median(transformed, axis=0))
    distances = cdist(np.asarray(profiles), np.asarray(artifact['centers']))
    ordered = np.sort(distances, axis=1)
    margins = np.divide(ordered[:, 1] - ordered[:, 0], ordered[:, 1], out=np.zeros(len(ids)), where=ordered[:, 1] > 0)
    training_ids = set(artifact['training_entity_ids'])
    return pd.DataFrame({'entity_id': ids, 'year': int(year), 'cluster': distances.argmin(axis=1),
                         'nearest_distance': ordered[:, 0], 'relative_distance_margin': margins,
                         'distance_extrapolation': ordered[:, 0] > artifact['distance_q99'],
                         'new_entity': [entity not in training_ids for entity in ids]})


def batch_relative_context(features, reference_features, reference_location):
    """Explicitly transductive location context; preserves the measured drift.

    Context may change with the reference cohort. It is not row-local inference,
    train-only correction, a causal effect or proof that common change is noise.
    """
    x, reference = _matrix(features), _matrix(reference_features)
    location = np.asarray(reference_location, dtype=float)
    if x.shape[1] != reference.shape[1] or location.shape != (x.shape[1],) or not np.isfinite(location).all():
        raise ValueError('Aligned context/reference dimensions required')
    drift = np.median(reference, axis=0) - location
    return {'scope': 'transductive_fixed_reference_cohort_relative_location',
            'corrected_features': x - drift, 'common_location_drift': drift,
            'reference_count': len(reference)}
