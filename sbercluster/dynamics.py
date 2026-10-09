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


def cluster_events(ids_before, labels_before, ids_after, labels_after, threshold=0.3, size_change=0.1):
    """Greene et al. (2010) step matching: Jaccard >= threshold links groups of consecutive partitions.

    One-to-one links are continue, grow or shrink; one-to-many is split, many-to-one is merge;
    unmatched groups are death (before) or birth (after).
    """
    if not 0 < threshold <= 1 or size_change < 0:
        raise ValueError("Threshold in (0,1] and nonnegative size change required")
    before, after = {}, {}
    for key, label in zip(ids_before, labels_before):
        before.setdefault(int(label), set()).add(key)
    for key, label in zip(ids_after, labels_after):
        after.setdefault(int(label), set()).add(key)
    links = [(i, j, len(a & b) / len(a | b)) for i, a in before.items() for j, b in after.items() if a & b]
    links = [(i, j, s) for i, j, s in links if s >= threshold]
    out = {i: [j for a, j, _ in links if a == i] for i in before}
    into = {j: [i for i, b, _ in links if b == j] for j in after}
    events = []
    for i, targets in sorted(out.items()):
        if not targets:
            events.append({"event": "death", "from": [i], "to": [], "size_before": len(before[i]), "size_after": 0})
        elif len(targets) > 1:
            events.append({"event": "split", "from": [i], "to": sorted(targets), "size_before": len(before[i]),
                           "size_after": sum(len(after[j]) for j in targets)})
    for j, sources in sorted(into.items()):
        if not sources:
            events.append({"event": "birth", "from": [], "to": [j], "size_before": 0, "size_after": len(after[j])})
        elif len(sources) > 1:
            events.append({"event": "merge", "from": sorted(sources), "to": [j],
                           "size_before": sum(len(before[i]) for i in sources), "size_after": len(after[j])})
    for i, j, similarity in sorted(links):
        if len(out[i]) == 1 and len(into[j]) == 1:
            ratio = len(after[j]) / len(before[i])
            kind = "grow" if ratio > 1 + size_change else "shrink" if ratio < 1 - size_change else "continue"
            events.append({"event": kind, "from": [i], "to": [j], "size_before": len(before[i]),
                           "size_after": len(after[j]), "jaccard": similarity})
    return events
