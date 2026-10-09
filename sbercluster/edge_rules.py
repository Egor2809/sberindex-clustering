"""Alternative edge rules on one cohort: profile distance, cosine, co-movement, lead-lag, DTW."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components


def residual_series(values):
    """Log monthly spending minus the national monthly median, demeaned and scaled per territory and series."""
    x = np.log(np.asarray(values, dtype=float))
    if x.ndim != 3 or not np.isfinite(x).all() or len(x) < 4:
        raise ValueError('Expected positive time x territory x series array')
    r = x - np.median(x, axis=1, keepdims=True)
    r -= r.mean(axis=0, keepdims=True)
    scale = r.std(axis=0, keepdims=True)
    return np.divide(r, scale, out=np.zeros_like(r), where=scale > 1e-12)


def similarity_knn(similarity, k=15):
    s = np.array(similarity, dtype=float, copy=True)
    n = len(s)
    if s.shape != (n, n) or not 0 < k < n or not np.isfinite(s).all():
        raise ValueError('Finite square similarity and 0 < k < n required')
    np.fill_diagonal(s, -np.inf)
    order = np.argsort(-s, axis=1, kind='stable')[:, :k]
    selected = np.take_along_axis(s, order, axis=1)
    keep = selected > 0
    rows = np.broadcast_to(np.arange(n)[:, None], order.shape)[keep]
    directed = csr_matrix((selected[keep], (rows, order[keep])), shape=(n, n))
    a = directed.maximum(directed.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def distance_knn(distance, k=15):
    d = np.array(distance, dtype=float, copy=True)
    n = len(d)
    if d.shape != (n, n) or not 0 < k < n or np.isnan(d).any() or (d < 0).any():
        raise ValueError('Nonnegative square distance and 0 < k < n required')
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1, kind='stable')[:, :k]
    selected = np.take_along_axis(d, order, axis=1)
    finite = np.isfinite(selected)
    positive = selected[finite & (selected > 0)]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    rows = np.broadcast_to(np.arange(n)[:, None], order.shape)[finite]
    weights = np.exp(-0.5 * (selected[finite] / sigma) ** 2)
    directed = csr_matrix((weights, (rows, order[finite])), shape=(n, n))
    a = directed.maximum(directed.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a, sigma


def cosine_similarity(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1)
    if (norms <= 0).any():
        raise ValueError('Zero vector has no direction')
    u = x / norms[:, None]
    return u @ u.T


def lagged_correlation(series, max_lag=3):
    """Mean over series of Pearson r(i at t, j at t+lag); returns best |lag|<=max_lag, positive lag means i leads j."""
    s = np.asarray(series, dtype=float)
    t, n, c = s.shape
    if not 0 <= max_lag < t - 2:
        raise ValueError('Lag too long for the window')
    best = np.full((n, n), -np.inf)
    lag = np.zeros((n, n), dtype=int)
    for shift in range(max_lag + 1):
        total = np.zeros((n, n))
        for m in range(c):
            a, b = s[:t - shift, :, m], s[shift:, :, m]
            a = (a - a.mean(axis=0)) / np.where(a.std(axis=0) > 1e-12, a.std(axis=0), np.inf)
            b = (b - b.mean(axis=0)) / np.where(b.std(axis=0) > 1e-12, b.std(axis=0), np.inf)
            total += a.T @ b / (t - shift)
        total /= c
        for value, sign in ((total, 1), (total.T, -1)):
            better = value > best
            best[better] = value[better]
            lag[better] = sign * shift
    return best, lag


def dtw_distance_matrix(series, window=2):
    """Multivariate DTW with a Sakoe-Chiba band; local cost is squared Euclidean distance."""
    s = np.asarray(series, dtype=float)
    t, n, _ = s.shape
    if window < 0:
        raise ValueError('Window must be nonnegative')
    out = np.zeros((n, n))
    for i in range(n - 1):
        others = s[:, i + 1:, :]
        cost = np.full((t + 1, t + 1, others.shape[1]), np.inf)
        cost[0, 0] = 0
        for p in range(1, t + 1):
            for q in range(max(1, p - window), min(t, p + window) + 1):
                local = np.square(others[q - 1] - s[p - 1, i]).sum(axis=1)
                cost[p, q] = local + np.minimum(np.minimum(cost[p - 1, q], cost[p, q - 1]), cost[p - 1, q - 1])
        out[i, i + 1:] = np.sqrt(cost[t, t])
    return out + out.T


def structure(a, groups):
    """Edge count, components, transitivity and the share of edges joining members of each grouping."""
    a = csr_matrix(a)
    b = a.copy()
    b.data = np.ones_like(b.data)
    degree = np.asarray(b.sum(axis=1)).ravel()
    triangles = float((b @ b).multiply(b).sum()) / 6
    triples = float((degree * (degree - 1)).sum()) / 2
    upper = triu(b, k=1)
    rows, cols = upper.nonzero()
    result = {'edges': int(upper.nnz), 'components': int(connected_components(b, directed=False)[0]),
              'isolates': int((degree == 0).sum()), 'mean_degree': float(degree.mean()),
              'transitivity': 3 * triangles / triples if triples else None}
    for name, labels in groups.items():
        labels = np.asarray(labels)
        result['within_' + name] = float(np.mean(labels[rows] == labels[cols])) if len(rows) else None
    return result


def edge_jaccard(a, b):
    x, y = triu(csr_matrix(a), k=1), triu(csr_matrix(b), k=1)
    x.data, y.data = np.ones_like(x.data), np.ones_like(y.data)
    both = x.multiply(y).nnz
    union = x.nnz + y.nnz - both
    return float(both / union) if union else None
