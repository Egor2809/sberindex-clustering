"""Continuous-first consumer atlas components; no economic outcomes are used.

The convex weights are geometric coordinates, NOT probabilities or estimated
expenditure shares. A legacy hard label is a separate output and is never
silently replaced by argmax(weights).
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.spatial.distance import cdist

REVISION = "continuous_atlas_v1"


def finite_matrix(value, name="matrix"):
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name}: nonempty finite 2D array required")
    return x


def convex_encode(x, centers, tolerance=1e-10):
    """Exact small-K convex-hull projection by exhaustive independent faces.

    Includes every vertex, hence reconstruction SSE cannot exceed nearest-center
    SSE except floating-point tolerance. For K=4 only 15 faces are considered.
    Independent faces suffice even with duplicate/affinely dependent centers.
    The projection is unique; weights need not be. Enumeration resolves ties.
    """
    x, centers = finite_matrix(x), finite_matrix(centers, "centers")
    if x.shape[1] != centers.shape[1] or not 1 <= len(centers) <= 8:
        raise ValueError("Aligned dimensions and 1..8 prototypes required")
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError("tolerance must be in (0,1e-6]")
    d2 = cdist(x, centers, "sqeuclidean")
    if not np.isfinite(d2).all():
        raise ValueError("Squared distance overflow; rescale inputs")
    nearest = d2.argmin(axis=1)
    weights = np.eye(len(centers))[nearest]
    best = d2[np.arange(len(x)), nearest].copy()
    for size in range(2, min(len(centers), x.shape[1] + 1) + 1):
        for face in combinations(range(len(centers)), size):
            vertices = centers[list(face)]
            basis = vertices[:-1] - vertices[-1]
            if np.linalg.matrix_rank(basis) != size - 1:
                continue
            coefficients = (x - vertices[-1]) @ np.linalg.pinv(basis)
            local = np.column_stack([coefficients, 1 - coefficients.sum(axis=1)])
            valid = (local >= -tolerance).all(axis=1)
            local = np.maximum(local, 0.)
            local /= local.sum(axis=1, keepdims=True)
            projected = local @ vertices
            error = np.square(x - projected).sum(axis=1)
            improve = valid & (error < best - tolerance * np.maximum(1., best))
            rows = np.flatnonzero(improve)
            if len(rows):
                weights[rows] = 0.
                weights[np.ix_(rows, list(face))] = local[rows]
                best[rows] = error[rows]
    reconstructed = weights @ centers
    residual = x - reconstructed
    return {"weights": weights, "reconstructed": reconstructed,
            "residual": residual, "squared_residual": np.square(residual).sum(axis=1),
            "nearest_center": nearest}


def legacy_predict(x, centers, biases=None):
    """Squared Euclidean distance plus fitted bias; stable first-index ties."""
    x, centers = finite_matrix(x), finite_matrix(centers, "centers")
    if x.shape[1] != centers.shape[1]:
        raise ValueError("Feature dimensions differ")
    biases = np.zeros(len(centers)) if biases is None else np.asarray(biases, float)
    if biases.shape != (len(centers),) or not np.isfinite(biases).all():
        raise ValueError("Finite aligned prototype biases required")
    distances = cdist(x, centers, "sqeuclidean") + biases
    if not np.isfinite(distances).all():
        raise ValueError("Distance rule overflow")
    return distances.argmin(axis=1)


def boundary_radius(x, centers, biases=None):
    """Distance to nearest decision boundary for the saved identity-metric rule.

    Radius in standardized feature units, NOT a calibrated error probability.
    At a tie it is zero. Identical centers with identical biases also yield zero.
    Strict perturbations with L2 norm < radius preserve the legacy label.
    """
    x, centers = finite_matrix(x), finite_matrix(centers, "centers")
    labels = legacy_predict(x, centers, biases)
    b = np.zeros(len(centers)) if biases is None else np.asarray(biases, float)
    scores = cdist(x, centers, "sqeuclidean") + b
    gap = scores - scores[np.arange(len(x)), labels, None]
    normals = 2 * np.linalg.norm(centers[labels, None, :] - centers[None, :, :], axis=2)
    radius = np.full(gap.shape, np.inf)
    np.divide(gap, normals, out=radius, where=normals > 0)
    radius[(normals == 0) & (gap == 0)] = 0.
    radius[np.arange(len(x)), labels] = np.inf
    return np.maximum(0., radius.min(axis=1))
