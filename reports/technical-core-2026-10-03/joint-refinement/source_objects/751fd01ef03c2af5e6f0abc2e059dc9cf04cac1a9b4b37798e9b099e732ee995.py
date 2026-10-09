"""Validate prepared panel identity independently of feature transformation."""
from numbers import Integral
import re

import pandas as pd


def _canonical_territory(value):
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return str(int(value))
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        return str(int(value))
    raise ValueError('Source territory IDs must be exact nonnegative integers')


def validate_prepared_panel(panel, manifest, cfg):
    """Reject contract drift before calibration or clustering can consume it."""
    identity = cfg.get('data_contract', {}).get('identity_mode')
    if not identity or manifest.get('identity_mode') != identity:
        raise ValueError('Prepared panel identity mode differs from configured data contract')
    if not {'entity_id', 'period'} <= set(panel.columns) or panel.empty:
        raise ValueError('Prepared panel requires entity identifiers and months')
    for column in ('entity_id', 'period'):
        if not panel[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError('Prepared panel keys must be nonempty strings')
    if panel.duplicated(['entity_id', 'period']).any():
        raise ValueError('Prepared panel has duplicate entity-month keys')
    if not panel.period.str.fullmatch(r'[0-9]{4}-[0-9]{2}-01').all():
        raise ValueError('Prepared panel periods must be ISO month-start dates')
    pd.to_datetime(panel.period.unique(), format='%Y-%m-%d', errors='raise')
    count = manifest.get('n_entities')
    if type(count) is not int or count != panel.entity_id.nunique():
        raise ValueError('Prepared panel entity count differs from manifest')
    if manifest.get('months') != sorted(panel.period.unique().tolist()):
        raise ValueError('Prepared panel months differ from manifest')
    if identity == 'source_territory_id':
        if 'territory_id' not in panel:
            raise ValueError('Source territory identity requires territory_id')
        canonical = panel.territory_id.map(_canonical_territory)
        if not panel.entity_id.eq('tid_' + canonical).all():
            raise ValueError('Prepared entity IDs differ from source territory IDs')
        pairs = pd.DataFrame({'entity_id': panel.entity_id, 'territory_id': canonical}).drop_duplicates()
        if pairs.entity_id.duplicated().any() or pairs.territory_id.duplicated().any():
            raise ValueError('Source territories and entity IDs must have a one-to-one mapping')
