"""Distinct attribute, road-distance and residual-trajectory graph layers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def graph_summary(a):
    a = csr_matrix(a)
    return {'edges': a.nnz // 2, 'components': int(connected_components(a, directed=False)[0]),
            'isolates': int(np.count_nonzero(np.asarray(a.sum(axis=1)).ravel() == 0)),
            'mean_degree': float(a.nnz / a.shape[0])}


def unit_mean_strength(a):
    """Match total graph mass, so alpha has the same meaning across layers."""
    a = csr_matrix(a, dtype=float, copy=True)
    if a.shape[0] != a.shape[1] or (a != a.T).nnz or np.any(a.diagonal()) or not np.isfinite(a.data).all() or np.any(a.data < 0):
        raise ValueError('Expected symmetric, finite, nonnegative, loop-free adjacency')
    mass = float(a.sum())
    if mass <= 0:
        raise ValueError('Cannot normalize an empty graph')
    return a * (a.shape[0] / mass)


def mix_layers(attribute, other, alpha):
    if not 0 <= alpha <= 1 or attribute.shape != other.shape:
        raise ValueError('Invalid mixture')
    result = alpha * unit_mean_strength(attribute) + (1-alpha) * unit_mean_strength(other)
    result.eliminate_zeros()
    return result


def distance_knn(distance, k=15):
    distance = np.array(distance, dtype=float, copy=True)
    n = len(distance)
    if distance.shape != (n,n) or not 0 < k < n or np.isnan(distance).any() or np.any(distance < 0):
        raise ValueError('Invalid distance matrix; use infinity for missing links')
    if not np.array_equal(distance, distance.T):
        raise ValueError('Road distances must be symmetric')
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1, kind='stable')[:, :k].copy()
    selected = np.take_along_axis(distance, order, axis=1)
    finite = np.isfinite(selected)
    positive = selected[finite & (selected > 0)]
    if not len(positive):
        raise ValueError('No positive road distances to calibrate the kernel')
    bandwidth = float(np.median(positive))
    rows = np.broadcast_to(np.arange(n)[:,None], order.shape)[finite]
    cols = order[finite]
    weights = np.exp(-0.5*(selected[finite]/bandwidth)**2)
    directed = csr_matrix((weights,(rows,cols)),shape=(n,n))
    a = directed.maximum(directed.T)
    a.eliminate_zeros()
    return a, order, selected, {'bandwidth_km':bandwidth,'k':k,**graph_summary(a)}


def read_road_graph(path, ids, k=15):
    """Use observed highway pairs only. Missing rail/road links are never zero."""
    import polars as pl
    tids = np.array([int(s.removeprefix('tid_')) for s in ids])
    if len(np.unique(tids)) != len(tids) or tids.min() < 0 or tids.max() > 10_000_000:
        raise ValueError('Invalid or duplicate territory identifiers')
    frame = (pl.scan_parquet(Path(path)).filter((pl.col('type')=='highway') &
             pl.col('territory_id_x').is_in(tids.tolist()) & pl.col('territory_id_y').is_in(tids.tolist()))
             .select('territory_id_x','territory_id_y','distance').collect())
    x, y, d = (frame[c].to_numpy() for c in frame.columns)
    if not np.isfinite(d).all() or np.any(d < 0) or np.any(x == y):
        raise ValueError('Invalid observed road distances')
    # The source stores one unordered record per pair; conflicts must not be averaged.
    keys = np.minimum(x,y).astype(np.int64)*(int(tids.max())+1) + np.maximum(x,y)
    if len(np.unique(keys)) != len(keys):
        raise ValueError('Duplicate unordered road pair')
    lookup = np.full(int(tids.max())+1,-1,dtype=int)
    lookup[tids] = np.arange(len(tids))
    i,j = lookup[x],lookup[y]
    distance = np.full((len(ids),len(ids)),np.inf)
    distance[i,j] = distance[j,i] = d
    observed_count = len(d)
    del frame,x,y,d,keys,i,j,lookup
    a, order, selected, info = distance_knn(distance,k)
    neighbors={key:[{'id':ids[int(idx)],'distance_km':float(km),
               'weight':float(np.exp(-.5*(km/info['bandwidth_km'])**2))}
               for idx,km in zip(order[row],selected[row]) if np.isfinite(km)] for row,key in enumerate(ids)}
    return a, neighbors, {**info,'observed_unordered_pairs':observed_count,
        'as_of':'2024-12-31','meaning':'road accessibility, not observed mobility or monetary flows'}


def trajectory_graph(monthly, k=15):
    """Cosine of centered residual trajectories, with common monthly pattern removed."""
    x = np.asarray(monthly,dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all() or len(x) < 3 or not 0 < k < x.shape[1]:
        raise ValueError('Expected time x territory x attribute array')
    residual = x - np.median(x,axis=1,keepdims=True)
    residual -= residual.mean(axis=0,keepdims=True)
    vectors = residual.transpose(1,0,2).reshape(x.shape[1],-1)
    norms = np.linalg.norm(vectors,axis=1)
    valid = norms > 1e-12
    vectors[valid] /= norms[valid,None]
    vectors[~valid] = 0
    similarity = np.clip(vectors @ vectors.T,0,1)
    np.fill_diagonal(similarity,0)
    order = np.argsort(-similarity,axis=1,kind='stable')[:,:k].copy()
    selected = np.take_along_axis(similarity,order,axis=1)
    positive = selected > 0
    rows = np.broadcast_to(np.arange(len(vectors))[:,None],order.shape)[positive]
    a = csr_matrix((selected[positive],(rows,order[positive])),shape=(len(vectors),len(vectors)))
    a = a.maximum(a.T)
    return a, {'months':len(x),'k':k,'zero_residual_territories':int((~valid).sum()),
        'window':'2023 calendar year; retrospective trajectory similarity',**graph_summary(a)}
