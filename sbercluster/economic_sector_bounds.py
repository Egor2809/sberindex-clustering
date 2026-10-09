"""Exact finite-cohort missing-share bounds for fixed paired predictions."""
from __future__ import annotations
import numpy as np


def squared_loss_difference_bounds(y, candidate, reference, regions):
    """Bound mean loss(reference)-loss(candidate), no missing-at-random assumption.

    Every fixed eligible case must have BOTH predictions. Unknown targets may
    independently range over [0,1]. The result is an exact identification
    interval for this finite cohort, not a sampling confidence interval.
    """
    y, a, b, regions = map(np.asarray, [y, candidate, reference, regions])
    y, a, b = y.astype(float), a.astype(float), b.astype(float)
    if y.ndim != 1 or not len(y) or any(x.shape != y.shape for x in [a, b, regions]):
        raise ValueError("Nonempty aligned target, predictions and fixed regions required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not ((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)).all():
        raise ValueError("All eligible cases require both finite predictions in[0,1]")
    if np.isinf(y).any() or regions.dtype.kind not in "iu":
        raise ValueError("Targets may be NaN, never infinite; integer region identities required")
    observed = np.isfinite(y)
    if not ((y[observed] >= 0) & (y[observed] <= 1)).all():
        raise ValueError("Observed sector shares must be in[0,1]; never clip observed data")
    at_zero, at_one = b ** 2 - a ** 2, (1 - b) ** 2 - (1 - a) ** 2
    lower, upper = np.minimum(at_zero, at_one), np.maximum(at_zero, at_one)
    realized = (y[observed] - b[observed]) ** 2 - (y[observed] - a[observed]) ** 2
    lower[observed], upper[observed] = realized, realized
    per_region = []
    for region in np.unique(regions):
        ix = regions == region
        known = ix & observed
        per_region.append({"region_code": int(region), "eligible_n": int(ix.sum()),
                "observed_n": int(known.sum()), "missing_n": int((ix & ~observed).sum()),
                "mean_loss_difference_lower": float(lower[ix].mean()),
                "mean_loss_difference_upper": float(upper[ix].mean()),
                "observed_mean_loss_difference": float(((y[known] - b[known]) ** 2 - (y[known] - a[known]) ** 2).mean()) if known.any() else None})
    equal_lower = float(np.mean([r["mean_loss_difference_lower"] for r in per_region]))
    equal_upper = float(np.mean([r["mean_loss_difference_upper"] for r in per_region]))
    conclusion = "candidate_better_for_every_missing_fill" if equal_lower > 0 else (
                  "candidate_worse_for_every_missing_fill" if equal_upper < 0 else "not_identified_under_arbitrary_missingness")
    return {"eligible_n": len(y), "observed_n": int(observed.sum()), "missing_n": int((~observed).sum()),
            "eligible_regions": len(per_region), "observed_regions": int(len(np.unique(regions[observed]))),
            "regions_with_no_observed_target": [r["region_code"] for r in per_region if not r["observed_n"]],
            "equal_region_lower": equal_lower, "equal_region_upper": equal_upper,
            "municipality_lower": float(lower.mean()), "municipality_upper": float(upper.mean()),
            "conclusion": conclusion, "per_region": per_region,
            "endpoint_note": "Sharp for independently unrestricted missing shares in[0,1]; joint sector constraints not imposed; not a confidence interval"}
