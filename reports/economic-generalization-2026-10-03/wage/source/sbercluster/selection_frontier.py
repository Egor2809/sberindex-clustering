"""Prototype challengers with an explicit inductive prediction contract.

All model selection scores belong to the original frozen feature space. The
optional covariance transform is fitted on the training profile only. A model
stores its transform, centres and additive squared-distance biases; prediction
never refits any of them. These are exploratory challengers, not a replacement
for the historical frozen model.
"""
from __future__ import annotations

from numbers import Integral, Real
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

REVISION = "prototype_frontier_v1"


def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite feature matrix")
    return x


def predict_frontier(x, model):
    """Apply the complete saved distance rule, with stable first-index ties."""
    x = _matrix(x)
    if model.get("revision") != REVISION:
        raise ValueError("Unsupported frontier model revision")
    transform = np.asarray(model["transform"], dtype=float)
    centers = np.asarray(model["centers"], dtype=float)
    biases = np.asarray(model["biases"], dtype=float)
    if (transform.shape != (x.shape[1], x.shape[1]) or centers.ndim != 2
            or centers.shape[1] != x.shape[1] or not len(centers)
            or biases.shape != (len(centers),) or not np.isfinite(transform).all()
            or not np.isfinite(centers).all() or not np.isfinite(biases).all()):
        raise ValueError("Invalid aligned frontier model")
    distances = cdist(x @ transform, centers, metric="sqeuclidean") + biases
    if not np.isfinite(distances).all():
        raise ValueError("Distance rule overflowed")
    return distances.argmin(axis=1).astype(np.int32)


def silhouette_from_distances(distances, labels):
    """Exact Euclidean SW and samples, matching sklearn singleton convention."""
    labels = np.asarray(labels)
    distances = np.asarray(distances, dtype=float)
    n = len(labels)
    if (distances.shape != (n, n) or not np.isfinite(distances).all()
            or np.any(distances < 0) or np.any(np.diag(distances) != 0)):
        raise ValueError("Expected a finite nonnegative zero-diagonal distance matrix")
    _, z = np.unique(labels, return_inverse=True)
    k = z.max() + 1
    if not 2 <= k < n:
        raise ValueError("SW requires 2 <= occupied K < N")
    counts = np.bincount(z, minlength=k)
    sums = np.column_stack([distances[:, z == group].sum(axis=1) for group in range(k)])
    own = sums[np.arange(n), z] / np.maximum(counts[z] - 1, 1)
    means = sums / counts
    means[np.arange(n), z] = np.inf
    other = means.min(axis=1)
    denominator = np.maximum(own, other)
    samples = np.divide(other - own, denominator, out=np.zeros(n), where=denominator > 0)
    samples[counts[z] == 1] = 0
    return float(samples.mean()), samples


def fit_frontier(x, spec, seed=1729, *, initial_labels=None, distances=None):
    """Fit KMeans, radial-Huber or shrinkage prototypes, optionally tune SW.

    Silhouette tuning adjusts only parameters of the deployable Voronoi rule.
    A hard minimum occupancy prevents a high SW obtained by tiny groups. The
    bounded coordinate search establishes no global optimum or stationarity.
    """
    x = _matrix(x)
    k = spec.get("k", 4)
    if isinstance(k, bool) or not isinstance(k, Integral) or not 2 <= k < len(x):
        raise ValueError("Expected integer 2 <= K < N")
    min_fraction = spec.get("min_fraction", .03)
    shrinkage = spec.get("shrinkage", 0.)
    quantile = spec.get("huber_quantile")
    min_ch_ratio = spec.get("min_ch_ratio")
    for name, value in [("min_fraction", min_fraction), ("shrinkage", shrinkage)]:
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + " must be finite in [0,1]")
    if min_fraction * k > 1:
        raise ValueError("Impossible minimum cluster fraction")
    if min_ch_ratio is not None and (isinstance(min_ch_ratio, bool) or not isinstance(min_ch_ratio, Real)
                                     or not np.isfinite(min_ch_ratio) or not 0 < min_ch_ratio <= 1):
        raise ValueError("Minimum CH ratio must be finite in (0,1]")
    if quantile is not None and (isinstance(quantile, bool) or not isinstance(quantile, Real)
                                 or not np.isfinite(quantile) or not 0 < quantile < 1):
        raise ValueError("Huber quantile must be finite in (0,1)")
    tune = spec.get("tune_silhouette", False)
    if not isinstance(tune, bool):
        raise ValueError("tune_silhouette must be boolean")
    passes = spec.get("max_passes", 8)
    if isinstance(passes, bool) or not isinstance(passes, Integral) or not 1 <= passes <= 30:
        raise ValueError("max_passes must be an integer in [1,30]")
    transform = np.eye(x.shape[1])
    if shrinkage:
        covariance = np.cov(x, rowvar=False).reshape(x.shape[1], x.shape[1])
        # Fixed identity shrinkage in the already standardized original space.
        covariance = (1 - shrinkage) * np.eye(x.shape[1]) + shrinkage * covariance
        values, vectors = np.linalg.eigh(covariance)
        if values.min() <= 0:
            raise ValueError("Covariance transform is singular")
        transform = (vectors * (1 / np.sqrt(values))) @ vectors.T
    transformed = x @ transform
    if initial_labels is not None:
        labels = np.asarray(initial_labels)
        if labels.shape != (len(x),) or labels.dtype.kind not in 'iu' or set(labels.tolist()) != set(range(k)):
            raise ValueError("Expected aligned labels occupying every requested cluster")
        centers = np.array([transformed[labels == c].mean(axis=0) for c in range(k)])
        estimator = KMeans(n_clusters=k, n_init=1, init=centers, random_state=seed).fit(transformed)
    else:
        estimator = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(transformed)
    centers = estimator.cluster_centers_.copy()
    biases = np.zeros(k)
    model = {"revision": REVISION, "spec": dict(spec), "transform": transform.tolist(),
             "centers": centers.tolist(), "biases": biases.tolist()}
    labels = predict_frontier(x, model)
    maximum_sse = None
    if min_ch_ratio is not None:
        baseline = (np.asarray(initial_labels) if initial_labels is not None else
                    KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(x))
        reference_sse = sum(float(np.square(x[baseline == c] - x[baseline == c].mean(axis=0)).sum())
                            for c in range(k))
        tss = float(np.square(x - x.mean(axis=0)).sum())
        # CH ratio is monotone in SSE at fixed N,K,TSS. This is the exact bound.
        maximum_sse = tss / (1 + min_ch_ratio * (tss / reference_sse - 1))
    huber_iterations = 0
    if quantile is not None:
        radius = float(np.quantile(np.linalg.norm(transformed - centers[labels], axis=1), quantile))
        if radius <= 0:
            raise ValueError("Huber influence radius is zero")
        for iteration in range(100):
            previous = centers.copy()
            for c in range(k):
                group = transformed[labels == c]
                if not len(group):
                    raise ValueError("Huber fit emptied a requested group")
                radial = np.linalg.norm(group - centers[c], axis=1)
                weights = np.minimum(1., radius / np.maximum(radial, np.finfo(float).tiny))
                centers[c] = np.average(group, axis=0, weights=weights)
            model["centers"] = centers.tolist()
            labels = predict_frontier(x, model)
            huber_iterations = iteration + 1
            if np.max(np.abs(centers - previous)) < 1e-7:
                break
        model["huber_radius"] = radius
    minimum = max(2, int(np.ceil(min_fraction * len(x))))
    occupancy = np.bincount(labels, minlength=k)
    info = {"huber_iterations": huber_iterations, "min_required_size": minimum,
            "eligible_occupancy": bool(occupancy.min() >= minimum), "trace": [],
            "maximum_common_space_sse": maximum_sse}
    if tune and occupancy.min() >= minimum:
        if distances is None:
            distances = cdist(x, x)
        score, _ = silhouette_from_distances(distances, labels)
        # Cluster standard deviations define dimensional coordinate step sizes.
        step = np.array([np.maximum(transformed[labels == c].std(axis=0) * .15, 1e-5)
                         for c in range(k)])
        bias_step = max(float(np.median(np.min(cdist(transformed, centers, 'sqeuclidean'), axis=1))) * .2, 1e-5)
        info["trace"].append({"pass": 0, "SW": score, "accepted": 0})
        for sweep in range(passes):
            accepted = 0
            for c in range(k):
                for axis in range(x.shape[1] + 1):
                    best = None
                    for sign in (-1, 1):
                        trial_centers, trial_biases = centers.copy(), biases.copy()
                        if axis == x.shape[1]:
                            trial_biases[c] += sign * bias_step
                            trial_biases -= trial_biases.mean()
                        else:
                            trial_centers[c, axis] += sign * step[c, axis]
                        trial_model = {**model, "centers": trial_centers.tolist(), "biases": trial_biases.tolist()}
                        trial_labels = predict_frontier(x, trial_model)
                        if np.array_equal(labels, trial_labels) or np.bincount(trial_labels, minlength=k).min() < minimum:
                            continue
                        if maximum_sse is not None:
                            trial_sse = sum(float(np.square(x[trial_labels == group] -
                                            x[trial_labels == group].mean(axis=0)).sum()) for group in range(k))
                            if trial_sse > maximum_sse + 1e-12:
                                continue
                        trial_score, _ = silhouette_from_distances(distances, trial_labels)
                        if trial_score > score + 1e-10 and (best is None or trial_score > best[0]):
                            best = (trial_score, trial_centers, trial_biases, trial_labels)
                    if best is not None:
                        score, centers, biases, labels = best
                        accepted += 1
            info["trace"].append({"pass": sweep + 1, "SW": score, "accepted": accepted})
            if not accepted:
                step *= .5
                bias_step *= .5
                if max(float(step.max()), bias_step) < 1e-4:
                    break
        model.update(centers=centers.tolist(), biases=biases.tolist())
    info.update(occupied_k=int(len(np.unique(labels))), cluster_sizes=np.bincount(labels, minlength=k).tolist())
    return labels, model, info
