from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score


def _encode_labels(labels, n=None):
    original_labels = labels
    labels = np.asarray(labels)
    if labels.ndim != 1 or (n is not None and len(labels) != n):
        raise ValueError("Expected one label per observation")
    if np.issubdtype(labels.dtype, np.inexact) and not np.isfinite(labels).all():
        raise ValueError("Labels must not contain missing or nonfinite values")
    if labels.dtype.kind in 'mM' and np.isnat(labels).any():
        raise ValueError("Labels must not contain missing or nonfinite values")
    # Inspect the original scalars too: NumPy otherwise coerces a NaN next to
    # string identifiers into the literal string "nan" before validation.
    if labels.dtype.kind in 'OUS' and any(
            value is None or (isinstance(value, (float, complex, np.inexact))
                              and not np.isfinite(value))
            for value in np.asarray(original_labels, dtype=object)):
        raise ValueError("Labels must not contain missing or nonfinite values")
    try:
        unique, encoded = np.unique(labels, return_inverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be comparable scalar identifiers") from exc
    return unique, encoded


def partition_labels(x, labels):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a finite node-by-feature matrix")
    unique, encoded = _encode_labels(labels, len(x))
    if not 2 <= len(unique) < len(x):
        raise ValueError("Metrics require 2 <= K < N")
    return x, encoded, len(unique)


def calinski_harabasz_indices(x, labels):
    """Return CH and CH/N without assigning a finite value to zero SSW.

    The mathematical CH ratio is positive infinity when SSW is zero and SSB
    is positive, and is undefined when both dispersions are zero.  JSON-facing
    values are ``None`` in both cases; the status and ``CH_is_infinite`` retain
    the distinction.  This avoids sklearn's historical finite fallback of 1.0.
    """
    x, z, k = partition_labels(x, labels)
    overall = x.mean(axis=0)
    ssw = 0.0
    ssb = 0.0
    for cluster in range(k):
        group = x[z == cluster]
        center = group.mean(axis=0)
        ssw += float(np.sum((group - center) ** 2))
        ssb += float(len(group) * np.sum((center - overall) ** 2))

    if ssw == 0.0:
        if ssb > 0.0:
            status = "positive_infinity_zero_within_dispersion"
            is_infinite = True
        else:
            status = "undefined_zero_within_and_between_dispersion"
            is_infinite = False
        return {
            "CH": None,
            "CH_per_n": None,
            "CH_status": status,
            "CH_is_infinite": is_infinite,
        }

    ch = (ssb / (k - 1)) / (ssw / (len(x) - k))
    return {
        "CH": float(ch),
        "CH_per_n": float(ch / len(x)),
        "CH_status": "ok",
        "CH_is_infinite": False,
    }


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


def s_dbw_paper(x, labels):
    """Literal implementation of Shalileh et al. (2025), Eq. 14.

    Eq. 14 prints f(x, c)=0 inside the radius and 1 outside it, the reverse of
    the within-radius density indicator in Halkidi--Vazirgiannis; both
    conventions are reported side by side.
    """
    x, z, k = partition_labels(x, labels)
    groups = [x[z == c] for c in range(k)]
    centers = [g.mean(axis=0) for g in groups]
    norms = np.array([np.linalg.norm(g.var(axis=0, ddof=0)) for g in groups])
    total_norm = np.linalg.norm(x.var(axis=0, ddof=0))
    if total_norm == 0:
        raise ValueError("S_Dbw_paper undefined for constant data")
    radius = np.sqrt(norms.sum()) / k
    density = [np.count_nonzero(np.linalg.norm(g - c, axis=1) > radius) for g, c in zip(groups, centers)]
    between = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            den = max(density[i], density[j])
            if den == 0:
                raise ValueError("S_Dbw_paper undefined: zero literal Eq. 14 centroid density")
            mid = (centers[i] + centers[j]) / 2
            num = np.count_nonzero(np.linalg.norm(np.vstack([groups[i], groups[j]]) - mid, axis=1) > radius)
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


def _network_block_sums(adjacency, labels, binary):
    """Validate a graph and return its cluster-to-cluster weight matrix."""
    a = csr_matrix(adjacency, dtype=float, copy=True)
    a.sum_duplicates()
    unique, z = _encode_labels(labels)
    if (a.shape != (len(z), len(z)) or not np.isfinite(a.data).all()
            or (a != a.T).nnz or np.any(a.diagonal() != 0) or np.any(a.data < 0)):
        raise ValueError("Undirected loop-free finite nonnegative graph required")
    a.eliminate_zeros()
    k = len(unique)
    if binary:
        a = a.copy()
        a.data = (a.data > 0).astype(float)
        a.eliminate_zeros()
    elif a.nnz:
        # All network ratios are invariant to uniform weight scaling. Keep
        # block sums and configuration-model volume finite on extreme scales.
        a.data /= a.data.max()
    if k < 2:
        return a, z, k, None
    h = csr_matrix((np.ones(len(z)), (np.arange(len(z)), z)), shape=(len(z), k))
    return a, z, k, (h.T @ a @ h).toarray()


def network_indices(adjacency, labels, *, binary=True):
    """AVI/AVU equations 18-21. Ordered AVU sum divided by K.

    Strict undefined denominators return None with counts. Do not conflate
    these indices with Newman modularity.  ``binary=True`` is the primary
    common comparison used by Shalileh et al.; ``binary=False`` retains the
    nonnegative edge weights allowed by the original AVI/AVU definition.
    """
    _, _, k, b = _network_block_sums(adjacency, labels, binary)
    if k < 2:
        return {"AVI": None, "AVU": None, "undefined_reason": "K < 2"}
    inside = b.diagonal()
    cross = b.copy()
    np.fill_diagonal(cross, 0)
    outside = cross.sum(axis=1)
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
    weight_convention = "binary" if binary else "weighted"
    return {"AVI": avi, "AVU": None if undef_u else float(total / k),
            "AVI_undefined_clusters": undef_i, "AVU_undefined_ordered_pairs": undef_u,
            "network_convention": f"{weight_convention}; symmetric adjacency; internal edges counted twice; AVU ordered sum divided by K"}


def network_quality_candidates(adjacency, labels, *, binary=True):
    """Newman modularity (reported as MQ) and the Mitchell--Mancoridis TurboMQ.

    Both use the graph convention selected by ``binary``.
    """
    a, _, k, b = _network_block_sums(adjacency, labels, binary)
    convention = "binary" if binary else "weighted"
    if k < 2:
        return {
            "TurboMQ": None,
            "TurboMQ_status": "undefined_K_less_than_2",
            "NewmanQ": None,
            "NewmanQ_status": "undefined_K_less_than_2",
            "network_quality_convention": convention,
        }

    inside = b.diagonal()
    cross = b.copy()
    np.fill_diagonal(cross, 0)
    outside = cross.sum(axis=1)
    cluster_factors = np.zeros(k, dtype=float)
    has_inside = inside > 0
    cluster_factors[has_inside] = inside[has_inside] / (inside[has_inside] + outside[has_inside])
    turbo_mq = float(cluster_factors.sum())

    total_weight_twice = float(a.sum())
    if total_weight_twice == 0.0:
        newman_q = None
        newman_status = "undefined_zero_total_edge_weight"
    else:
        cluster_strength = b.sum(axis=1)
        newman_q = float(np.sum(inside / total_weight_twice - (cluster_strength / total_weight_twice) ** 2))
        newman_status = "ok"

    return {
        "TurboMQ": turbo_mq,
        "TurboMQ_status": "ok",
        "NewmanQ": newman_q,
        "NewmanQ_status": newman_status,
        "network_quality_convention": (
            f"{convention}; symmetric adjacency; TurboMQ sum of cluster factors; "
            "NewmanQ configuration-model modularity"
        ),
    }


def all_metrics(x, labels, adjacency):
    x, z, k = partition_labels(x, labels)
    result = {"n": len(x), "k": k, "min_cluster_size": int(np.bincount(z).min()),
              "SW": float(silhouette_score(x, z)),
              "MQ": None, "MQ_status": None}
    result.update(calinski_harabasz_indices(x, z))
    try:
        result["S_Dbw"] = s_dbw(x, z)
        result["S_Dbw_status"] = "ok"
    except ValueError as exc:
        result.update({"S_Dbw": None, "S_Dbw_status": str(exc)})
    result["S_Dbw_diagnostics"] = s_dbw_diagnostics(x, z)
    result["S_Dbw_paper_convention"] = (
        "Shalileh2025-Eq14-literal-outside-radius; differs from the HV01 density convention"
    )
    try:
        result["S_Dbw_paper"] = s_dbw_paper(x, z)
        result["S_Dbw_paper_status"] = "ok"
    except ValueError as exc:
        result.update({"S_Dbw_paper": None, "S_Dbw_paper_status": str(exc)})
    result.update(network_indices(adjacency, z))
    result.update(network_quality_candidates(adjacency, z))
    result["MQ"] = result["NewmanQ"]
    result["MQ_status"] = "Modularity Quality = Newman modularity on the binary graph" if result["MQ"] is not None else result["NewmanQ_status"]
    return result


def aligned_stability(ids_a, labels_a, ids_b, labels_b):
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    if ids_a.ndim != 1 or ids_b.ndim != 1:
        raise ValueError("Expected one-dimensional ID sequences")
    _encode_labels(ids_a)
    _encode_labels(ids_b)
    _encode_labels(labels_a, len(ids_a))
    _encode_labels(labels_b, len(ids_b))
    a, b = dict(zip(ids_a, labels_a)), dict(zip(ids_b, labels_b))
    if len(a) != len(ids_a) or len(b) != len(ids_b):
        raise ValueError("Duplicate IDs")
    common = sorted(a.keys() & b.keys())
    if len(common) < 2:
        return {"common_n": len(common), "ARI": None}
    return {"common_n": len(common), "ARI": float(adjusted_rand_score([a[i] for i in common], [b[i] for i in common]))}
