"""Explicitly gated clustering backends; never called by audit or preparation."""
from __future__ import annotations
import numpy as np


def require_execution(cfg, execute=False):
    if not execute or cfg.get("execution", {}).get("allow_clustering") is not True:
        raise PermissionError("Clustering disabled. Both --execute-clustering and execution.allow_clustering=true are required.")


def _igraph(a, ids):
    import igraph as ig
    from scipy.sparse import triu
    coo = triu(a, k=1).tocoo()
    g = ig.Graph(n=len(ids), edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
    g.vs["id"] = ids
    g.es["weight"] = coo.data.tolist()
    return g


def fit_static(method, x, a, ids, cfg, execute=False):
    require_execution(cfg, execute)
    settings = cfg["clustering"]
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=settings["n_clusters"], n_init=20, random_state=cfg["seed"]).fit_predict(x)
    if method == "ward":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=settings["n_clusters"], linkage="ward").fit_predict(x)
    if method == "leiden_static":
        import leidenalg as la
        p = la.find_partition(_igraph(a, ids), la.RBConfigurationVertexPartition,
             weights="weight", resolution_parameter=settings["resolution"],
             seed=cfg["seed"], n_iterations=settings["leiden_iterations"])
        return np.asarray(p.membership)
    raise ValueError(f"Unknown static method: {method}")


def fit_temporal(slices, graphs, cfg, execute=False):
    require_execution(cfg, execute)
    if cfg["data_contract"]["stable_territories_verified"] is not True:
        raise ValueError("Temporal coupling requires documented boundary stability review")
    import leidenalg as la
    gs = [_igraph(a, ids) for (_, ids, _), a in zip(slices, graphs)]
    # A fixed fraction of development-period median vertex strength. Not a per-month tuned value.
    strengths = [np.asarray(a.sum(axis=1)).ravel() for (p, _, _), a in zip(slices, graphs) if p <= cfg["validation"]["development_end"]]
    if not strengths:
        raise ValueError("No development graphs for temporal coupling calibration")
    omega = cfg["clustering"]["temporal_coupling_relative"] * float(np.median(np.concatenate(strengths)))
    memberships, improvement = la.find_partition_temporal(gs, la.RBConfigurationVertexPartition,
        interslice_weight=omega, vertex_id_attr="id", weight_attr="weight",
        resolution_parameter=cfg["clustering"]["resolution"], seed=cfg["seed"],
        n_iterations=cfg["clustering"]["leiden_iterations"])
    return [np.asarray(x) for x in memberships], {"omega": omega, "optimisation_improvement": float(improvement),
            "inference_scope": "retrospective_joint_optimisation; NOT an online early-warning system"}
