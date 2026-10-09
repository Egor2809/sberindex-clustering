"""Small building blocks for later independent validation, with explicit cutoffs."""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def heldout_peer_concordance(past_x, future_change, k=15):
    """Retrospective concordance of held-out changes, NOT a forecast.

    Peers are chosen from pre-cutoff features, but their later outcomes are used
    to assess shared behavior after those outcomes are observed.
    This function does not choose/tune features; future outcomes never enter distances.
    Baselines must be evaluated on this same finite common cohort.
    """
    x, y = np.asarray(past_x, float), np.asarray(future_change, float)
    if y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all() or not 1 <= k < len(x):
        raise ValueError("Invalid common cohort")
    distance = cdist(x, x)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
    peer_reference = np.median(y[neighbors], axis=1)
    return {"peer_reference": peer_reference, "absolute_discrepancy": np.abs(peer_reference - y), "neighbors": neighbors}


def paired_bootstrap(error_model, error_baseline, seed=1729, repetitions=1000):
    """Positive mean difference means lower model error; paired entities resampled."""
    a, b = np.asarray(error_model, float), np.asarray(error_baseline, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a + b).all():
        raise ValueError("Paired finite errors required")
    delta = b - a
    rng = np.random.default_rng(seed)
    means = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repetitions)])
    return {"mean_improvement": float(delta.mean()), "ci_95": np.quantile(means, [.025, .975]).tolist(),
            "n": len(delta), "resampling_unit": "municipality; spatial dependence not corrected"}
