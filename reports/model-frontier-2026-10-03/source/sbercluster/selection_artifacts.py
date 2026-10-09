"""Fingerprint-checked annual prediction for complete prototype frontier rules."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .features import transform_frozen
from .io import sha256
from .selection_frontier import predict_frontier

MODEL_KIND = 'prototype_frontier'
FORMAT_VERSION = 1


def validate_artifact(artifact):
    if (not isinstance(artifact, dict) or artifact.get('model_kind') != MODEL_KIND
            or type(artifact.get('format_version')) is not int or artifact.get('format_version') != FORMAT_VERSION):
        raise ValueError('Expected prototype_frontier artifact format 1')
    scaler = artifact['feature_scaler']
    if scaler.get('mode') != 'log_ratios_to_total' or scaler.get('level_weight', 0.) != 0.:
        raise ValueError('Frontier annual artifact requires five frozen log-ratio features')
    centers = np.asarray(artifact['rule']['centers'], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 5 or len(centers) < 2:
        raise ValueError('Expected at least two five-dimensional prototype centres')
    predict_frontier(np.zeros((1, 5)), artifact['rule'])
    projection = np.asarray(artifact['pre_aggregation_transform'], dtype=float)
    if projection.shape != (5, 5) or not np.isfinite(projection).all():
        raise ValueError('Invalid pre-aggregation feature transform')
    mapping = artifact['cluster_mapping']
    if not isinstance(mapping, list) or any(type(value) is not int for value in mapping) or mapping != list(range(len(centers))):
        raise ValueError('Cluster mapping must preserve the saved rule indices')
    if artifact.get('aggregation') != 'calendar_year_median_of_12_months':
        raise ValueError('Unsupported aggregation rule')
    if artifact.get('prediction_scope') != 'descriptive_consumer_profiles':
        raise ValueError('Unsupported prediction scope')
    return artifact


def load_artifact(path, expected_sha256):
    path = Path(path)
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected_sha256):
        raise ValueError('An explicit 64-digit expected SHA256 is required')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise ValueError('Frontier artifact fingerprint differs')
    return validate_artifact(json.loads(raw))


def make_artifact(rule, scaler, *, provenance, pre_aggregation_transform=None):
    """Build a discriminator-separated export; the old frozen CLI rejects it."""
    matrix = np.eye(5) if pre_aggregation_transform is None else pre_aggregation_transform
    artifact = {'model_kind': MODEL_KIND, 'format_version': FORMAT_VERSION,
                'feature_scaler': scaler, 'pre_aggregation_transform': np.asarray(matrix).tolist(),
                'aggregation': 'calendar_year_median_of_12_months', 'rule': rule,
                'cluster_mapping': list(range(len(rule['centers']))),
                'prediction_scope': 'descriptive_consumer_profiles', 'provenance': provenance}
    return validate_artifact(artifact)


def predict_annual_panel(panel, artifact, year):
    """Frozen monthly transform, declared transform, annual median, full rule.

    Every entity in the selected calendar year must contain all twelve months.
    Territory ID persistence alone does not certify unchanged boundaries.
    """
    validate_artifact(artifact)
    if isinstance(year, bool) or not isinstance(year, (int, np.integer)) or not 1900 <= year <= 9998:
        raise ValueError('Expected a supported integer calendar year')
    if 'entity_id' not in panel or 'period' not in panel or panel.empty:
        raise ValueError('Expected nonempty entity_id/period panel')
    if (panel.entity_id.isna().any() or not panel.entity_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or panel.period.isna().any() or not panel.period.map(lambda v: isinstance(v, str)).all()
            or not panel.period.str.fullmatch(r'\d{4}-\d{2}-01').all()):
        raise ValueError('Expected complete string IDs and canonical month-start periods')
    parsed = pd.to_datetime(panel.period, format='%Y-%m-%d', errors='raise')
    if panel.duplicated(['entity_id', 'period']).any():
        raise ValueError('Duplicate entity-month')
    selected = panel[parsed.dt.year == year].copy().sort_values(['entity_id', 'period'])
    if selected.empty:
        raise ValueError('Selected calendar year is absent')
    expected = pd.date_range(f'{year}-01-01', f'{year}-12-01', freq='MS').strftime('%Y-%m-%d').tolist()
    identifiers, features = [], []
    transform = np.asarray(artifact['pre_aggregation_transform'], dtype=float)
    for entity_id, rows in selected.groupby('entity_id', sort=True):
        if rows.period.tolist() != expected:
            raise ValueError('Each selected entity requires all twelve calendar months: ' + entity_id)
        monthly = transform_frozen(rows, artifact['feature_scaler']) @ transform
        identifiers.append(entity_id)
        features.append(np.median(monthly, axis=0))
    labels = predict_frontier(np.asarray(features), artifact['rule'])
    return pd.DataFrame({'entity_id': identifiers, 'year': year, 'cluster': labels})
