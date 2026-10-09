"""Exact denominator-axis decomposition in the existing frozen ratio space."""
from __future__ import annotations
import numpy as np


def denominator_direction(scaler):
    if scaler.get('mode') != 'log_ratios_to_total' or scaler.get('level_weight', 0.) != 0.:
        raise ValueError('Expected the frozen five-dimensional log-ratio space')
    scale = np.asarray(scaler['ratio_iqr'], dtype=float)
    if scale.shape != (5,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Expected five finite positive frozen IQRs')
    # Multiplying the denominator by q shifts x by -log(q)*direction.
    return 1 / scale / np.sqrt(5)


def contrast_projection(scaler):
    direction = denominator_direction(scaler)
    unit = direction / np.linalg.norm(direction)
    return np.eye(5) - np.outer(unit, unit)


def contrast_basis_transform(scaler):
    """Canonical Householder contrast basis, four coordinates plus a zero.

    The reflector maps the standardized denominator direction to axis five.
    Its first four columns are an orthonormal basis of the orthogonal space.
    A fixed basis matters because component-wise monthly medians depend on it.
    """
    unit = denominator_direction(scaler)
    unit /= np.linalg.norm(unit)
    target = np.array([0., 0., 0., 0., 1.])
    difference = unit - target
    norm_squared = float(difference @ difference)
    reflector = np.eye(5) if norm_squared < 1e-30 else np.eye(5) - 2 * np.outer(difference, difference) / norm_squared
    reflector[:, -1] = 0.
    return reflector


def contrast_annual(monthly, scaler):
    """Transform each month to four contrasts first, then take their medians."""
    monthly = np.asarray(monthly, dtype=float)
    if monthly.ndim != 3 or monthly.shape[2] != 5 or not len(monthly) or not np.isfinite(monthly).all():
        raise ValueError('Expected finite month-by-entity-by-five-feature tensor')
    return np.median(monthly @ contrast_basis_transform(scaler), axis=0)


def axis_decomposition(x, labels, scaler):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if x.ndim != 2 or x.shape[1] != 5 or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Expected aligned finite five-dimensional profiles and labels')
    unit = denominator_direction(scaler)
    unit /= np.linalg.norm(unit)
    axis = x @ unit
    total = float(np.square(x - x.mean(axis=0)).sum())
    axis_total = float(np.square(axis - axis.mean()).sum())
    between = axis_between = 0.
    for c in np.unique(labels):
        mask = labels == c
        between += float(mask.sum() * np.square(x[mask].mean(axis=0) - x.mean(axis=0)).sum())
        axis_between += float(mask.sum() * (axis[mask].mean() - axis.mean()) ** 2)
    return {'total_ss': total, 'axis_total_ss': axis_total,
            'axis_total_fraction': axis_total / total if total else None,
            'between_ss': between, 'axis_between_ss': axis_between,
            'axis_between_fraction': axis_between / between if between else None,
            'axis_eta_squared': axis_between / axis_total if axis_total else None}
