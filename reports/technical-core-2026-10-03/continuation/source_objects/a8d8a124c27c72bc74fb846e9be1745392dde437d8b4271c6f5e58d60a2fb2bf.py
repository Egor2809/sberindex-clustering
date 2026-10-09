"""Explicitly gated clustering backends; never called by audit or preparation."""
from __future__ import annotations
from datetime import date
from numbers import Real
import re
import numpy as np
from scipy.sparse import csr_matrix, triu


def require_execution(cfg, execute=False):
    if execute is not True or cfg.get("execution", {}).get("allow_clustering") is not True:
        raise PermissionError("Clustering disabled. Both --execute-clustering and execution.allow_clustering=true are required.")


def _validated_ids(ids, n=None):
    values = np.asarray(ids, dtype=object)
    if values.ndim != 1 or not len(values) or (n is not None and len(values) != n):
        raise ValueError('Expected one nonempty unique ID per feature row')
    for value in values:
        if (value is None or (isinstance(value, str) and not value.strip())
                or (isinstance(value, (float, complex, np.inexact)) and not np.isfinite(value))):
            raise ValueError('IDs must be complete scalar identifiers')
        try:
            if value != value:
                raise ValueError('IDs must be complete scalar identifiers')
        except TypeError as exc:
            # pandas.NA has no truth value; NaT/NaN are not self-equal.
            raise ValueError('IDs must be complete scalar identifiers') from exc
    try:
        if len(set(values)) != len(values):
            raise ValueError('Duplicate vertex IDs')
    except TypeError as exc:
        raise ValueError('IDs must be hashable scalar identifiers') from exc
    return values.tolist()


def _validated_graph(adjacency, n):
    a = csr_matrix(adjacency, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    if (a.shape != (n, n) or not np.isfinite(a.data).all() or np.any(a.data < 0)
            or np.any(a.diagonal()) or (a != a.T).nnz):
        raise ValueError('Expected an aligned symmetric finite nonnegative loop-free graph')
    with np.errstate(over='ignore'):
        mass = float(a.sum())
    if not np.isfinite(mass):
        raise ValueError('Graph volume overflowed; rescale its weights')
    return a


def _validated_features(x, n):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != n or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Expected a finite aligned node-by-feature matrix')
    return x


def _integer(value, name, lower=0, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < lower or (upper is not None and value > upper)):
        raise ValueError(f'{name} must be an integer in its valid range')
    return int(value)


def _leiden_settings(cfg):
    settings = cfg['clustering']
    resolution = settings['resolution']
    if (isinstance(resolution, (bool, np.bool_)) or not isinstance(resolution, Real)
            or not np.isfinite(resolution) or resolution < 0):
        raise ValueError('resolution must be finite and nonnegative')
    iterations = settings['leiden_iterations']
    # Leiden documents negative values as "until no improvement". Zero runs
    # no optimisation; reject it instead of recording an unoptimised fit.
    if (isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer))
            or iterations == 0):
        raise ValueError('leiden_iterations must be a nonzero integer')
    return float(resolution), int(iterations), _integer(cfg['seed'], 'seed', 0, 2**32-1)


def _validated_membership(labels, n):
    labels = np.asarray(labels)
    if labels.shape != (n,) or labels.dtype.kind not in 'iu' or np.any(labels < 0):
        raise ValueError('Clustering backend returned invalid aligned integer memberships')
    return labels


def temporal_scope(cfg):
    """Validate the same boundary/sensitivity gate for every temporal caller."""
    contract = cfg['data_contract']
    stable = contract.get('stable_territories_verified') is True
    descriptive = (contract.get('temporal_scope') == 'retrospective_constant_id_sensitivity'
                   and bool(contract.get('boundary_review_evidence')))
    if not stable and not descriptive:
        raise ValueError('Temporal coupling requires boundary review or an explicitly limited retrospective sensitivity analysis')
    return stable


def _month(period):
    if not isinstance(period, str) or re.fullmatch(r'\d{4}-\d{2}-01', period) is None:
        raise ValueError('Temporal periods must be canonical month starts YYYY-MM-01')
    try:
        parsed = date.fromisoformat(period)
    except ValueError as exc:
        raise ValueError('Invalid temporal calendar month') from exc
    return parsed.year * 12 + parsed.month


def _igraph(a, ids):
    ids = _validated_ids(ids)
    a = _validated_graph(a, len(ids))
    import igraph as ig
    coo = triu(a, k=1).tocoo()
    g = ig.Graph(n=len(ids), edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
    g.vs["id"] = ids
    g.es["weight"] = coo.data.tolist()
    return g


def fit_static(method, x, a, ids, cfg, execute=False):
    require_execution(cfg, execute)
    if method not in ('kmeans', 'ward', 'leiden_static'):
        raise ValueError(f'Unknown static method: {method}')
    ids = _validated_ids(ids)
    x = _validated_features(x, len(ids))
    if a is not None:
        a = _validated_graph(a, len(ids))
    settings = cfg["clustering"]
    seed = _integer(cfg['seed'], 'seed', 0, 2**32-1)
    if method == "kmeans":
        k = _integer(settings['n_clusters'], 'n_clusters', 1, len(ids))
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x)
    elif method == "ward":
        k = _integer(settings['n_clusters'], 'n_clusters', 1, len(ids))
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
    else:
        resolution, iterations, seed = _leiden_settings(cfg)
        if a is None:
            raise ValueError('Leiden requires an aligned graph')
        import leidenalg as la
        p = la.find_partition(_igraph(a, ids), la.RBConfigurationVertexPartition,
             weights="weight", resolution_parameter=resolution,
             seed=seed, n_iterations=iterations)
        labels = p.membership
    return _validated_membership(labels, len(ids))


def fit_temporal(slices, graphs, cfg, execute=False):
    require_execution(cfg, execute)
    stable = temporal_scope(cfg)
    if not slices or len(slices) != len(graphs):
        raise ValueError('One graph is required for every time slice')
    if any(not isinstance(s, (list, tuple)) or len(s) != 3 for s in slices):
        raise ValueError('Expected period, IDs and features for every temporal slice')
    ids = _validated_ids(slices[0][1])
    dimensions, validated_graphs, periods = [], [], []
    for (period, current_ids, features), graph in zip(slices, graphs):
        if _validated_ids(current_ids) != ids:
            raise ValueError('Temporal sensitivity analysis requires the same ordered, unique territory IDs')
        dimensions.append(_validated_features(features, len(ids)).shape[1])
        validated_graphs.append(_validated_graph(graph, len(ids)))
        periods.append(_month(period))
    if len(set(dimensions)) != 1:
        raise ValueError('Temporal feature dimensions must agree across slices')
    if any(b != a+1 for a, b in zip(periods, periods[1:])):
        raise ValueError('Temporal coupling requires ordered consecutive calendar months')
    cutoff = _month(cfg['validation']['development_end'])
    resolution, iterations, seed = _leiden_settings(cfg)
    relative = cfg['clustering']['temporal_coupling_relative']
    if (isinstance(relative, (bool, np.bool_)) or not isinstance(relative, Real)
            or not np.isfinite(relative) or relative < 0):
        raise ValueError('temporal_coupling_relative must be finite and nonnegative')
    # A fixed fraction of development-period median vertex strength. Not a per-month tuned value.
    strengths = [np.asarray(a.sum(axis=1)).ravel() for p, a in zip(periods, validated_graphs) if p <= cutoff]
    if not strengths:
        raise ValueError("No development graphs for temporal coupling calibration")
    omega = relative * float(np.median(np.concatenate(strengths)))
    if not np.isfinite(omega):
        raise ValueError('Calibrated temporal coupling overflowed; rescale the graph weights')
    import leidenalg as la
    gs = [_igraph(a, ids) for a in validated_graphs]
    memberships, improvement = la.find_partition_temporal(gs, la.RBConfigurationVertexPartition,
        interslice_weight=omega, vertex_id_attr="id", weight_attr="weight",
        resolution_parameter=resolution, seed=seed, n_iterations=iterations)
    if len(memberships) != len(slices) or not np.isfinite(improvement):
        raise ValueError('Temporal backend returned invalid slice count or improvement')
    memberships = [_validated_membership(labels, len(ids)) for labels in memberships]
    return memberships, {"omega": omega, "optimisation_improvement": float(improvement),
            "inference_scope": "retrospective_joint_optimisation; NOT an online early-warning system",
            "boundary_stability_verified": stable,
            "boundary_caveat": None if stable else "2024 within-year changes not independently cleared; memberships are descriptive, not verified structural transitions"}
