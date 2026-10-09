"""Pinned continuous consumer profiles, preserving both saved label rules.

Weights describe convex coordinates, not probabilities or spending shares.
Only complete expense years 2023/2024 are supported; no fitting is performed.
"""
from __future__ import annotations

import csv
import hashlib
from io import BytesIO, StringIO
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .continuous_geometry import boundary_radius, convex_encode
from .features import transform_frozen
from .io import CATEGORIES, TOTAL
from .selection_frontier import predict_frontier

REVISION = 'sberai_continuous_atlas_release_v1'
CODECS = ('legacy_centers', 'observed_full', 'observed_core95')
LEGACY_RULES = ('historical_kmeans4', 'sw_constrained_huber75')


def pinned_read(path, expected_sha256):
    """Read once and require a caller-supplied hash of these exact bytes."""
    if (not isinstance(expected_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256)):
        raise ValueError('Lowercase 64-character SHA256 required')
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Pinned bytes differ: ' + str(path))
    return content


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON model field: ' + key)
        result[key] = value
    return result


def _finite_values(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_values(child)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite model value')


def _array(value, shape, name):
    # Reject booleans and numeric strings rather than silently coercing JSON.
    object_array = np.asarray(value, dtype=object)
    if (object_array.shape != shape
            or any(isinstance(v, bool) or not isinstance(v, Real)
                   for v in object_array.flat)):
        raise ValueError('Invalid ' + name)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid ' + name) from exc
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite ' + name)
    return array


def validate_model(model):
    """Validate the supported release, including unused explanation codecs."""
    if not isinstance(model, dict) or model.get('revision') != REVISION:
        raise ValueError('Unsupported atlas revision')
    _finite_values(model)
    if type(model.get('training_year')) is not int or model['training_year'] != 2023:
        raise ValueError('Unsupported training year')
    if model.get('category_order') != CATEGORIES:
        raise ValueError('Unsupported category order')
    scaler = model.get('feature_scaler')
    if (not isinstance(scaler, dict) or scaler.get('mode') != 'log_ratios_to_total'
            or isinstance(scaler.get('level_weight'), bool)
            or scaler.get('level_weight') != 0
            or scaler.get('calibration_end') != '2023-12-01'):
        raise ValueError('Unsupported feature geometry')
    _array(scaler.get('ratio_center'), (5,), 'scaler center')
    if (_array(scaler.get('ratio_iqr'), (5,), 'scaler IQR') <= 0).any():
        raise ValueError('Positive scaler IQR required')
    ids = model.get('training_entity_ids')
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(i, str) or not re.fullmatch(r'tid_(0|[1-9][0-9]*)', i)
                   for i in ids) or len(set(ids)) != len(ids)):
        raise ValueError('Invalid training membership')
    rules = model.get('legacy_rules')
    if not isinstance(rules, dict) or set(rules) != set(LEGACY_RULES):
        raise ValueError('Both legacy rules must be preserved')
    for rule in rules.values():
        if not isinstance(rule, dict) or rule.get('revision') != 'prototype_frontier_v1':
            raise ValueError('Unsupported legacy rule')
        if not np.array_equal(_array(rule.get('transform'), (5, 5), 'transform'), np.eye(5)):
            raise ValueError('Unsupported legacy distance transform')
        _array(rule.get('centers'), (4, 5), 'legacy centers')
        _array(rule.get('biases'), (4,), 'legacy biases')
    codecs = model.get('codecs')
    if not isinstance(codecs, dict) or set(codecs) != set(CODECS):
        raise ValueError('Unexpected codec vocabulary')
    for name, codec in codecs.items():
        if not isinstance(codec, dict):
            raise ValueError('Invalid explanation codec: ' + name)
        _array(codec.get('centers'), (4, 5), 'explanation centers: ' + name)
    return model


def load_model(path, expected_sha256):
    """Load a byte-pinned JSON model, rejecting duplicate and nonfinite fields."""
    def reject_constant(value):
        raise ValueError('Nonfinite JSON model value: ' + value)
    try:
        model = json.loads(pinned_read(path, expected_sha256).decode('utf-8'),
                           object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed model JSON') from exc
    return validate_model(model)


def _validate_csv_quotes(content):
    # csv.reader(strict=True) accepts quotes inside unquoted fields. Reject
    # these as well, while preserving escaped quotes and quoted line breaks.
    state = 'start'
    for char in content:
        if state == 'quoted':
            if char == '"':
                state = 'closed'
        elif state == 'closed':
            if char == '"':
                state = 'quoted'
            elif char == ',':
                state = 'start'
            elif char in '\r\n':
                state = 'start'
            else:
                raise ValueError('Broken CSV quoting')
        elif char == '"':
            if state != 'start':
                raise ValueError('Broken CSV quoting')
            state = 'quoted'
        elif char == ',' or char in '\r\n':
            state = 'start'
        else:
            state = 'unquoted'
    if state == 'quoted':
        raise ValueError('Broken CSV quoting')


def _validate_panel(panel):
    required = ['entity_id', 'territory_id', 'period'] + CATEGORIES + [TOTAL]
    if (not isinstance(panel, pd.DataFrame) or panel.empty
            or panel.columns.duplicated().any() or any(c not in panel for c in required)):
        raise ValueError('Nonempty panel with unique required columns expected')
    if panel[required].isna().any().any():
        raise ValueError('Missing identity, calendar or expense cells')
    if not panel.period.map(lambda p: isinstance(p, str) and bool(
            re.fullmatch(r'202[34]-(0[1-9]|1[012])-01', p))).all():
        raise ValueError('Only canonical 2023/2024 month-start rows allowed')
    tids = panel.territory_id.map(str)
    if (not tids.str.fullmatch(r'0|[1-9][0-9]*').all()
            or not panel.entity_id.eq('tid_' + tids).all()):
        raise ValueError('Source territory/entity ID mismatch')
    if panel.duplicated(['entity_id', 'period']).any():
        raise ValueError('Duplicate entity-month')
    values = panel[CATEGORIES + [TOTAL]]
    if any(isinstance(value, (bool, np.bool_)) for value in values.to_numpy(dtype=object).flat):
        raise ValueError('Positive finite expense estimates required')
    try:
        numeric = values.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Positive finite expense estimates required') from exc
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError('Positive finite expense estimates required')


def read_panel_csv(path, expected_sha256):
    """Read pinned CSV with exact row widths, IDs and positive finite expenses."""
    raw = pinned_read(path, expected_sha256)
    try:
        content = raw.decode('utf-8-sig')
        _validate_csv_quotes(content)
        reader = csv.reader(StringIO(content, newline=''), strict=True)
        header = next(reader, [])
        if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
            raise ValueError('Missing or duplicate input column names')
        for row in reader:
            if len(row) != len(header):
                raise ValueError('CSV row width differs from header')
        panel = pd.read_csv(BytesIO(raw), dtype={'entity_id': str, 'territory_id': str, 'period': str})
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Malformed panel CSV') from exc
    _validate_panel(panel)
    return panel


def encode_panel(panel, model, year, codec):
    """Return the original five coordinates, convex explanation and saved labels."""
    validate_model(model)
    if type(year) is not int or year not in (2023, 2024):
        raise ValueError('Only expense years 2023/2024 supported; no future-outcome operation')
    if not isinstance(codec, str) or codec not in CODECS:
        raise ValueError('Unknown explicitly selected codec')
    _validate_panel(panel)
    annual = panel.loc[panel.period.str.startswith(str(year) + '-')].copy()
    if annual.empty:
        raise ValueError('No rows in selected year')
    ids = sorted(annual.entity_id.unique())
    expected = [f'{year}-{month:02d}-01' for month in range(1, 13)]
    if not annual.groupby('entity_id').period.agg(lambda values: sorted(values) == expected).all():
        raise ValueError('Incomplete calendar year for an entity')
    rows, totals = [], []
    for period in expected:
        block = annual.loc[annual.period.eq(period)].set_index('entity_id').loc[ids]
        rows.append(transform_frozen(block, model['feature_scaler']))
        totals.append(block[TOTAL].to_numpy(dtype=float))
    x = np.median(rows, axis=0)
    centers = np.asarray(model['codecs'][codec]['centers'], dtype=float)
    enc = convex_encode(x, centers)
    result = pd.DataFrame({'entity_id': ids, 'year': year, 'codec': codec,
        'new_entity': ~np.isin(ids, model['training_entity_ids']), 'month_coverage': 12,
        'log_median_total': np.log(np.median(totals, axis=0)),
        'territory_transition_status': 'unresolved',
        'weight_semantics': 'convex_coordinates_not_probabilities'})
    for j in range(5):
        result[f'x{j}'] = x[:, j]
        result[f'reconstructed{j}'] = enc['reconstructed'][:, j]
        result[f'residual{j}'] = enc['residual'][:, j]
    for j in range(4):
        result[f'weight{j}'] = enc['weights'][:, j]
    result['residual_norm'] = np.linalg.norm(enc['residual'], axis=1)
    if 'boundary_review_flag' in annual:
        flags = annual.groupby('entity_id').boundary_review_flag.agg(
            lambda values: ';'.join(sorted(set(values.dropna().astype(str)))))
        result['source_boundary_review_flags'] = flags.reindex(ids).to_numpy()
    for name, rule in model['legacy_rules'].items():
        result[name] = predict_frontier(x, rule)
        result[name + '_boundary_radius'] = boundary_radius(x, rule['centers'], rule['biases'])
    return result
