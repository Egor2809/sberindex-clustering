"""Transitions retain full contingency tables; a label renaming is not a change."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .metrics import aligned_stability


def transitions(ids_before, labels_before, ids_after, labels_after):
    for ids, labels in [(ids_before, labels_before), (ids_after, labels_after)]:
        values = np.asarray(labels)
        if values.ndim != 1 or len(ids) != len(values):
            raise ValueError("One label is required for each entity ID")
        if not all(isinstance(value, str) and value.strip() for value in ids):
            raise ValueError("Entity IDs must be nonempty strings")
        if values.dtype.kind not in 'iu' and len(values):
            raise ValueError("Cluster labels must be integers")
    a, b = dict(zip(ids_before, labels_before)), dict(zip(ids_after, labels_after))
    if len(a) != len(ids_before) or len(b) != len(ids_after):
        raise ValueError("Duplicate entity IDs")
    common = sorted(a.keys() & b.keys())
    if not common:
        return {"common_n": 0, "entered": len(b), "exited": len(a),
                "events": [], "ARI": None, "matched_churn": None}
    ua, ub = sorted({a[i] for i in common}), sorted({b[i] for i in common})
    table = np.zeros((len(ua), len(ub)), dtype=int)
    ia, ib = {v:i for i,v in enumerate(ua)}, {v:i for i,v in enumerate(ub)}
    for key in common:
        table[ia[a[key]], ib[b[key]]] += 1
    rows, cols = linear_sum_assignment(-table)
    events = []
    for i in range(len(ua)):
        for j in range(len(ub)):
            if table[i,j]:
                events.append({"from": int(ua[i]), "to": int(ub[j]), "count": int(table[i,j]),
                    "share_of_origin": float(table[i,j]/table[i].sum()), "share_of_destination": float(table[i,j]/table[:,j].sum())})
    return {**aligned_stability(ids_before, labels_before, ids_after, labels_after),
            "matched_churn": float(1 - table[rows,cols].sum()/len(common)),
            "entered": len(b.keys()-a.keys()), "exited": len(a.keys()-b.keys()), "events": events,
            "interpretation": "overlap accounting only; neither boundary validation nor proof of economic change"}
