from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score


def partition_labels(x, labels):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if len(labels) != len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid features or labels length")
    unique, encoded = np.unique(labels, return_inverse=True)
    if not 2 <= len(unique) < len(x):
        raise ValueError("Metrics require 2 <= K < N")
    return x, encoded, len(unique)


def s_dbw(x, labels):
    """Halkidi-style S_Dbw: population variance, ball density <= radius.

    Radius = sqrt(sum(norm(var(Ck))))/K. Zero density denominator is
    undefined (raise), not an artificial perfect score. See docs/METRICS.md.
    """
    x, z, k = partition_labels(x, labels)
    groups = [x[z == c] for c in range(k)]
    centers = [g.mean(axis=0) for g in groups]
    norms = np.array([np.linalg.norm(g.var(axis=0, ddof=0)) for g in groups])
    total_norm = np.linalg.norm(x.var(axis=0, ddof=0))
    if total_norm == 0:
        raise ValueError("S_Dbw undefined for constant data")
    radius = np.sqrt(norms.sum()) / k
    density = [np.count_nonzero(np.linalg.norm(g - c, axis=1) <= radius) for g, c in zip(groups, centers)]
    between = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            den = max(density[i], density[j])
            if den == 0:
                raise ValueError("S_Dbw undefined: zero centroid density")
            mid = (centers[i] + centers[j]) / 2
            num = np.count_nonzero(np.linalg.norm(np.vstack([groups[i], groups[j]]) - mid, axis=1) <= radius)
            between += 2 * num / den
    return float(norms.mean() / total_norm + between / (k * (k - 1)))



def s_dbw_diagnostics(x, labels):
    """Expose empty density balls without substituting another S_Dbw variant."""
    x, z, k = partition_labels(x, labels)
    groups = [x[z == c] for c in range(k)]
    norms = np.array([np.linalg.norm(g.var(axis=0, ddof=0)) for g in groups])
    radius = float(np.sqrt(norms.sum()) / k)
    densities = [int(np.count_nonzero(np.linalg.norm(g - g.mean(axis=0), axis=1) <= radius)) for g in groups]
    empty = [c for c,d in enumerate(densities) if d == 0]
    return {'convention': 'S_Dbw-HV01-var/mean-centroid', 'radius': radius,
            'centroid_densities': densities, 'zero_density_clusters': empty,
            'undefined_pairs': len(empty)*(len(empty)-1)//2}


def network_indices(adjacency, labels):
    """AVI/AVU equations 18-21, binary undirected graph. Ordered AVU sum / K.

    Strict undefined denominators return None with counts. Do not conflate
    this with Newman modularity or silently substitute a weighted extension.
    """
    a = csr_matrix(adjacency, dtype=float)
    z = np.asarray(labels)
    if a.shape != (len(z), len(z)) or (a != a.T).nnz or np.any(a.diagonal() != 0) or np.any(a.data < 0):
        raise ValueError("Undirected loop-free nonnegative graph required")
    _, z = np.unique(z, return_inverse=True)
    k = len(np.unique(z))
    if k < 2:
        return {"AVI": None, "AVU": None, "undefined_reason": "K < 2"}
    a.data = (a.data > 0).astype(float)
    h = csr_matrix((np.ones(len(z)), (np.arange(len(z)), z)), shape=(len(z), k))
    b = (h.T @ a @ h).toarray()
    inside = b.diagonal()
    outside = b.sum(axis=1) - inside
    den_i = inside + outside
    undef_i = int(np.count_nonzero(den_i == 0))
    avi = None if undef_i else float(np.mean(inside / den_i))
    total, undef_u = 0.0, 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            den = outside[i] + outside[j] - b[i, j]
            if den <= 0:
                undef_u += 1
            else:
                total += b[i, j] / den
    return {"AVI": avi, "AVU": None if undef_u else float(total / k),
            "AVI_undefined_clusters": undef_i, "AVU_undefined_ordered_pairs": undef_u,
            "network_convention": "binary; symmetric adjacency; internal edges counted twice; AVU ordered sum divided by K"}


def all_metrics(x, labels, adjacency):
    x, z, k = partition_labels(x, labels)
    result = {"n": len(x), "k": k, "min_cluster_size": int(np.bincount(z).min()),
              "SW": float(silhouette_score(x, z)), "CH": float(calinski_harabasz_score(x, z)),
              "MQ": None, "MQ_status": "definition_pending_organizer"}
    try:
        result["S_Dbw"] = s_dbw(x, z)
    except ValueError as exc:
        result.update({"S_Dbw": None, "S_Dbw_status": str(exc)})
    result["S_Dbw_diagnostics"] = s_dbw_diagnostics(x, z)
    result.update(network_indices(adjacency, z))
    return result


def aligned_stability(ids_a, labels_a, ids_b, labels_b):
    a, b = dict(zip(ids_a, labels_a)), dict(zip(ids_b, labels_b))
    if len(a) != len(ids_a) or len(b) != len(ids_b):
        raise ValueError("Duplicate IDs")
    common = sorted(a.keys() & b.keys())
    if len(common) < 2:
        return {"common_n": len(common), "ARI": None}
    return {"common_n": len(common), "ARI": float(adjusted_rand_score([a[i] for i in common], [b[i] for i in common]))}
