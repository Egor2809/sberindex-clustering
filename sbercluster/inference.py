"""Strict application of frozen consumer-profile prototypes; never fits a model."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .features import transform_frozen
from .input_contracts import validate_prepared_panel
from .io import CATEGORIES, TOTAL


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON model field: ' + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class FrozenProfileModel:
    """Validated fixed coordinate system and centers, with source fingerprint."""
    _centers: np.ndarray
    _scaler: dict
    _reference_ids: frozenset
    model_sha256: str

    @classmethod
    def load(cls, path, expected_sha256=None):
        raw = Path(path).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError('Frozen model fingerprint differs')
        def reject_constant(value):
            raise ValueError('Nonfinite JSON model value: ' + value)
        model = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                           parse_constant=reject_constant)
        if not isinstance(model, dict):
            raise ValueError('Frozen model must be a JSON object')
        # Historical archives have no discriminator. Explicit newer formats
        # must never be interpreted by silently ignoring their decision rule.
        if 'model_kind' in model and model['model_kind'] != 'frozen_profile':
            raise ValueError('Unsupported frozen model_kind: ' + str(model['model_kind']))
        if ('format_version' in model
                and (type(model['format_version']) is not int or model['format_version'] != 1)):
            raise ValueError('Unsupported frozen format_version: ' + str(model['format_version']))
        try:
            centers = np.asarray(model['centers'], dtype=float)
            scaler = deepcopy(model['scaler'])
            ids = model['ids']
            labels = np.asarray(model['reference_labels'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Incomplete or malformed frozen model') from exc
        if not isinstance(scaler, dict):
            raise ValueError('Frozen scaler must be an object')
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Frozen reference IDs must be unique nonempty strings')
        if centers.ndim != 2 or not 2 <= len(centers) < len(ids) or not np.isfinite(centers).all():
            raise ValueError('Finite nondegenerate frozen centers required')
        if (labels.shape != (len(ids),) or labels.dtype.kind not in 'iu'
                or not np.array_equal(np.unique(labels), np.arange(len(centers)))):
            raise ValueError('Frozen labels must match every ID and every center')
        cutoff = scaler.get('calibration_end')
        if not isinstance(cutoff, str) or not pd.Series([cutoff]).str.fullmatch(r'[0-9]{4}-[0-9]{2}-01').all():
            raise ValueError('Frozen calibration cutoff must be an ISO month-start date')
        pd.to_datetime(cutoff, format='%Y-%m-%d', errors='raise')
        # Validate even scalers for modes not used by the current reference model.
        probe = pd.DataFrame([{**{c: 1. for c in CATEGORIES}, TOTAL: 10.}])
        try:
            dimensions = transform_frozen(probe, scaler).shape[1]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid frozen feature space') from exc
        if centers.shape[1] != dimensions:
            raise ValueError('Frozen centers and feature space dimensions differ')
        centers.setflags(write=False)
        return cls(centers, scaler, frozenset(ids), actual)

    @property
    def scaler(self):
        return deepcopy(self._scaler)

    def predict(self, frame, aggregation='monthly'):
        """Return descriptive nearest-prototype assignments and distance margins.

        Annual profiles require all 12 distinct calendar months for every
        entity-year. New IDs are allowed and explicitly marked. The input's
        source semantics and boundary comparability remain the caller's remit.
        """
        if aggregation not in ('monthly', 'annual'):
            raise ValueError('aggregation must be monthly or annual')
        required = {'entity_id', 'territory_id', 'period', TOTAL, *CATEGORIES}
        if not required <= set(frame.columns):
            raise ValueError('Inference table is missing required columns')
        if frame.columns.duplicated().any():
            raise ValueError('Duplicate input column names')
        if not frame.period.map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError('Inference periods must be nonempty strings')
        envelope = {'identity_mode': 'source_territory_id',
                    'n_entities': int(frame.entity_id.nunique()),
                    'months': sorted(frame.period.dropna().unique().tolist())}
        validate_prepared_panel(frame, envelope, {'data_contract': {'identity_mode': 'source_territory_id'}})
        ordered = frame.sort_values(['entity_id', 'period']).reset_index(drop=True)
        transformed = transform_frozen(ordered, self._scaler)
        if aggregation == 'monthly':
            vectors = transformed
            result = ordered[['entity_id', 'period']].rename(columns={'period': 'period_start'})
            result['period_end'] = result.period_start
            result['observations'] = 1
        else:
            keys = ordered[['entity_id', 'period']].copy()
            keys['year'] = keys.period.str[:4]
            metadata, vectors = [], []
            for (entity, year), indices in keys.groupby(['entity_id', 'year'], sort=True).groups.items():
                periods = keys.loc[indices, 'period'].tolist()
                expected = [f'{year}-{month:02d}-01' for month in range(1, 13)]
                if periods != expected:
                    raise ValueError(f'Complete calendar year required for {entity}, {year}')
                vectors.append(np.median(transformed[np.asarray(indices)], axis=0))
                metadata.append({'entity_id': entity, 'period_start': expected[0],
                                 'period_end': expected[-1], 'observations': 12})
            result = pd.DataFrame(metadata)
            vectors = np.stack(vectors)
        distances = cdist(vectors, self._centers)
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite distance to frozen prototypes')
        ranks = np.argsort(distances, axis=1, kind='stable')
        best = ranks[:, 0]
        first = distances[np.arange(len(vectors)), best]
        second = distances[np.arange(len(vectors)), ranks[:, 1]]
        result['cluster'] = best
        result['distance_to_centroid'] = first
        result['relative_distance_margin'] = np.divide(second - first, second,
            out=np.zeros_like(first), where=second > 0)
        result['in_reference_cohort'] = result.entity_id.isin(self._reference_ids)
        result['aggregation'] = aggregation
        return result.sort_values(['period_start', 'entity_id']).reset_index(drop=True)
