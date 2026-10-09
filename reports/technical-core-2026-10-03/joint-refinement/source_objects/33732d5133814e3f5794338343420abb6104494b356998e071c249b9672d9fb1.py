"""Transparent joint baseline: attribute distortion plus a road graph cut.

This is a Potts-regularized KMeans baseline, not a reproduction of KEFRiN,
WSNMF or DMoN. For nonempty clusters its objective is
    J(z,C) = sum_i ||x_i-C[z_i]||^2 / TSS
             + alpha * sum_{i<j} a_ij [z_i != z_j] / sum_{i<j} a_ij.
TSS = sum_i ||x_i-mean(x)||^2. Both terms are invariant to uniform rescaling
of their respective inputs. A graph term is not independent validation.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, triu

ALGORITHM_REVISION = 'exact_centroid_moves_v2'


def _inputs(x, a, labels, alpha):
    x = np.asarray(x, dtype=float)
    a = csr_matrix(a, dtype=float, copy=True)
    a.sum_duplicates()
    z = np.asarray(labels)
    if x.ndim != 2 or len(x) < 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Expected a finite nonempty attribute matrix')
    if (a.shape != (len(x), len(x)) or (a != a.T).nnz or np.any(a.diagonal())
            or not np.isfinite(a.data).all() or np.any(a.data < 0)):
        raise ValueError('Expected a symmetric nonnegative loop-free aligned graph')
    if z.shape != (len(x),) or not np.issubdtype(z.dtype, np.integer):
        raise ValueError('Expected one integer label per vertex')
    unique, z = np.unique(z, return_inverse=True)
    if not 2 <= len(unique) < len(x):
        raise ValueError('Expected between 2 and n-1 nonempty clusters')
    with np.errstate(over='ignore', invalid='ignore'):
        tss = float(np.square(x - x.mean(axis=0)).sum())
        mass = float(a.sum()) / 2
    if (not np.isfinite(alpha) or alpha < 0 or not np.isfinite(tss)
            or not np.isfinite(mass) or tss <= 0 or mass <= 0):
        raise ValueError('Require nonnegative finite alpha, positive finite TSS and graph mass')
    a.eliminate_zeros()
    return x, a, z, tss, mass


def _components(x, a, z, tss, mass, alpha):
    centers = np.array([x[z == k].mean(axis=0) for k in range(z.max()+1)])
    sse = float(np.square(x-centers[z]).sum())
    edges = triu(a, k=1).tocoo()
    cut = float(edges.data[z[edges.row] != z[edges.col]].sum())
    return {'objective': sse/tss + alpha*cut/mass,
            'attribute_SSE_over_TSS': sse/tss, 'graph_cut_fraction': cut/mass,
            'SSE': sse, 'TSS': tss, 'undirected_edge_mass': mass}


def joint_objective(x, a, labels, alpha):
    """Evaluate the specified objective at optimal centroids for these labels."""
    x, a, z, tss, mass = _inputs(x, a, labels, alpha)
    return _components(x, a, z, tss, mass, alpha)


def fit_graph_regularized_kmeans(x, a, initial_labels, alpha=.25, seed=1729,
                                 max_sweeps=50):
    """Monotone single-vertex minimization from a supplied attribute baseline.

    Each move accounts exactly for the centroids changing in both clusters.
    Never empty a cluster; accept strict objective decreases only. Seed controls
    vertex order, not initialization. The alpha=0 ablation returns the supplied
    baseline unchanged without claiming convergence, so archived fits can be reused.
    Positive alpha converges to a coordinate fixed point or a recorded cap;
    global optimality and balanced cluster sizes are not guaranteed.
    """
    x, a, z, tss, mass = _inputs(x, a, initial_labels, alpha)
    if (isinstance(max_sweeps, (bool, np.bool_))
            or not isinstance(max_sweeps, (int, np.integer)) or max_sweeps < 1):
        raise ValueError('max_sweeps must be a positive integer')
    k = int(z.max()+1)
    rng = np.random.default_rng(seed)
    trace = [_components(x, a, z, tss, mass, alpha)]
    moves = []
    converged = False
    for _ in range(0 if alpha == 0 else max_sweeps):
        centers = np.array([x[z == c].mean(axis=0) for c in range(k)])
        counts = np.bincount(z, minlength=k)
        changed = 0
        for i in rng.permutation(len(x)):
            old = z[i]
            if counts[old] == 1:
                continue
            start, end = a.indptr[i:i+2]
            neighbors, weights = a.indices[start:end], a.data[start:end]
            same = np.bincount(z[neighbors], weights=weights, minlength=k)
            squared_distances = np.square(x[i] - centers).sum(axis=1)
            # Removing a point reduces optimal SSE by n/(n-1) times its old
            # squared distance; adding it increases SSE by n/(n+1) times the
            # distance to the recipient's current centroid (Hartigan move).
            removal = counts[old] / (counts[old] - 1) * squared_distances[old]
            costs = (counts / (counts + 1) * squared_distances - removal) / tss
            costs += alpha * (same[old] - same) / mass
            costs[old] = 0.0
            new = int(np.argmin(costs))
            if costs[new] < -1e-14:
                centers[old] -= (x[i] - centers[old]) / (counts[old] - 1)
                centers[new] += (x[i] - centers[new]) / (counts[new] + 1)
                z[i] = new
                counts[old] -= 1
                counts[new] += 1
                changed += 1
        current = _components(x, a, z, tss, mass, alpha)
        if current['objective'] > trace[-1]['objective'] + 1e-10:
            raise ArithmeticError('Joint objective increased')
        trace.append(current)
        moves.append(changed)
        if not changed:
            converged = True
            break
    status = ('baseline_reused_without_optimization' if alpha == 0 else
              'coordinate_fixed_point' if converged else 'sweep_cap_reached')
    # The reuse ablation preserves literal labels, including nonconsecutive IDs.
    if alpha == 0:
        z = np.asarray(initial_labels).copy()
    return z, {'algorithm': 'potts_graph_regularized_kmeans',
               'algorithm_revision': ALGORITHM_REVISION, 'alpha': float(alpha),
               'initialization': 'supplied_attribute_baseline', 'seed': int(seed),
               'max_sweeps': int(max_sweeps), 'converged': converged,
               'convergence_status': status,
               'sweeps': len(moves), 'moves_per_sweep': moves, 'trace': trace,
               'warnings': ['sweep_cap_reached'] if status == 'sweep_cap_reached' else [],
               'optimality': ('not optimized; supplied baseline reused' if alpha == 0 else
                              'coordinate local only; nonempty clusters enforced'),
               **trace[-1]}
