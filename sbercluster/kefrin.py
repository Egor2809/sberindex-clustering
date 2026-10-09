"""KEFRiNe (Shalileh & Mirkin, Entropy 2022, doi:10.3390/e24050626): K-means over features and network rows.

The criterion sums squared Euclidean distances to cluster centres in the feature block and in the block of
adjacency rows. Each block is z-scored by column and scaled to unit total variance, so the two blocks get equal
weight (rho = xi). Centres are initialised by k-means++ with restarts instead of the anomalous-cluster procedure.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans


def _block(m):
    m = np.asarray(m.toarray() if issparse(m) else m, dtype=float)
    sd = m.std(axis=0)
    m = np.divide(m - m.mean(axis=0), sd, out=np.zeros_like(m), where=sd > 0)
    return m / np.sqrt((m ** 2).sum())


def kefrin(x, a, k, seed=1729, n_init=50):
    data = np.hstack([_block(x), _block(a)])
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(data)
