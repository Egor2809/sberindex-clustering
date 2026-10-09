from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def knn_graph(x, k=15, symmetry="union"):
    """Exact deterministic kNN. No estimator fit. One monthly n*n matrix at a time."""
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all()
            or isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k < len(x)):
        raise ValueError("Finite 2D matrix and 1 <= k < n required")
    if symmetry not in ("union", "mutual"):
        raise ValueError("symmetry must be union or mutual")
    distances = cdist(x, x)
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise distances overflowed; rescale the finite features")
    np.fill_diagonal(distances, np.inf)
    # Stable sorting breaks exact ties by row index, which is sorted entity_id upstream.
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(distances, neighbors, axis=1)
    positive = selected[selected > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(-np.square(selected / sigma) / 2)
    rows = np.repeat(np.arange(len(x)), k)
    directed = csr_matrix((weights.ravel(), (rows, neighbors.ravel())), shape=(len(x), len(x)))
    a = directed.maximum(directed.T) if symmetry == "union" else directed.minimum(directed.T)
    a.setdiag(0)
    a.eliminate_zeros()
    components, _ = connected_components(a, directed=False)
    return a, {"k": k, "symmetry": symmetry, "sigma": sigma, "edges": a.nnz // 2,
               "components": int(components), "isolates": int((a.getnnz(axis=1) == 0).sum()),
               "median_strength": float(np.median(np.asarray(a.sum(axis=1)).ravel()))}
