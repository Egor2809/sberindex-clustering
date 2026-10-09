"""Describe stable temporal groups by Mirkin's rule and measure how much of external indicators they explain."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sbercluster.temporal_groups import CATEGORIES, _read_panel, event_owners, rejoin, track_key, track_labels

OUTCOMES = {
    "wage_2023": ("Зарплата, 2023 (log)", True),
    "population": ("Население (log)", True),
    "urban_share_2024": ("Доля городского населения, 2024", False),
    "above_working_age_share_2023": ("Доля старше трудоспособного, 2023", False),
    "market_access": ("Индекс доступности рынков, 2024", False),
}

NORTH = ("Республика Саха (Якутия)", "Магаданская область", "Чукотский автономный округ", "Мурманская область",
         "Ненецкий автономный округ", "Камчатский край", "Ямало-Ненецкий автономный округ")
NORTH_GROUPS = ("Восток и Север", "Малые районы Якутии")


def eta_squared(values: list[float], labels: list) -> float:
    mean = sum(values) / len(values)
    total = sum((v - mean) ** 2 for v in values)
    groups: dict = {}
    for v, g in zip(values, labels):
        groups.setdefault(g, []).append(v)
    between = sum(len(vs) * (sum(vs) / len(vs) - mean) ** 2 for vs in groups.values())
    return between / total


def epsilon_squared(values: list[float], labels: list) -> float:
    k, n = len(set(labels)), len(values)
    mean = sum(values) / n
    total = sum((v - mean) ** 2 for v in values)
    eta = eta_squared(values, labels)
    within = total * (1 - eta)
    return (total * eta - (k - 1) * within / (n - k)) / total


def bootstrap_difference(values, a, b, regions, reps=1000, seed=1729):
    import numpy as np
    rng = np.random.default_rng(seed)
    names = sorted(set(regions))
    index = {r: [i for i, x in enumerate(regions) if x == r] for r in names}
    out = []
    for _ in range(reps):
        rows = [i for r in rng.choice(names, len(names)) for i in index[r]]
        v = [values[i] for i in rows]
        out.append(eta_squared(v, [a[i] for i in rows]) - eta_squared(v, [b[i] for i in rows]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def kmeans7_labels(ids):
    import numpy as np
    from sklearn.cluster import KMeans
    from sbercluster.edge_study import load_panel
    qcfg = json.loads((ROOT / "configs/temporal_quality.json").read_text("utf-8"))
    panel_ids, periods, monthly, _, _ = load_panel(qcfg, ROOT)
    months = sum(p <= qcfg["features"]["calibration_end"] for p in periods)
    annual = np.median(np.asarray(monthly)[:months], axis=0)
    labels = KMeans(7, n_init=50, random_state=qcfg["seed"]).fit_predict(annual)
    return dict(zip(panel_ids, (str(z) for z in labels)))


def mirkin(group_mean: float, overall_mean: float) -> float:
    return (group_mean - overall_mean) / overall_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/temporal_groups.json")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text("utf-8"))
    with gzip.open(ROOT / cfg["temporal_labels"], "rt", encoding="utf-8") as stream:
        temporal = json.load(stream)
    events = json.loads((ROOT / cfg["events"]).read_text("utf-8"))["temporal_leiden"]
    ids, periods, labels = temporal["ids"], temporal["periods"], temporal["labels"]
    owners, _ = event_owners(events["sizes"], events["steps"])
    tracks = rejoin(track_labels(labels, owners), cfg["rejoin_jaccard"])
    alive = Counter(track for row in tracks for track in set(row))
    stable = sorted(k for k, n in alive.items() if n >= cfg["min_months"])
    names = {k: cfg["names"][track_key(k, owners, periods)] for k in stable}
    ratios, _ = _read_panel(ROOT / cfg["panel"], set(ids), periods)

    cells = [(i, t, tracks[t][i]) for t in range(len(periods)) for i in range(len(ids))]
    overall = [sum(ratios[ids[i], periods[t]][j] for i, t, _ in cells) / len(cells) for j in range(len(CATEGORIES))]
    profiles = []
    for k in stable:
        own = [(i, t) for i, t, track in cells if track == k]
        means = [sum(ratios[ids[i], periods[t]][j] for i, t in own) / len(own) for j in range(len(CATEGORIES))]
        deviations = {name: round(mirkin(m, o), 4) for name, m, o in zip(CATEGORIES, means, overall)}
        profiles.append({"name": names[k], "cells": len(own), "mean_shares": {n: round(m, 4) for n, m in zip(CATEGORIES, means)},
                         "deviation": deviations,
                         "salient": [n for n, d in sorted(deviations.items(), key=lambda x: -abs(x[1])) if abs(d) >= args.threshold]})

    modal = {}
    for i, key in enumerate(ids):
        counts = Counter(tracks[t][i] for t in range(len(periods)) if tracks[t][i] in names)
        if counts:
            modal[key] = names[max(counts, key=lambda c: (counts[c], -c))]
    with (ROOT / cfg["indicators"]).open(encoding="utf-8", newline="") as stream:
        rows = {row["entity_id"]: row for row in csv.DictReader(stream)}
    with (ROOT / "data/processed/market_access_2024.csv").open(encoding="utf-8", newline="") as stream:
        access = {f"tid_{row['territory_id']}": row["market_access"] for row in csv.DictReader(stream)}
    with (ROOT / cfg["reference"]).open(encoding="utf-8", newline="") as stream:
        kmeans = {row["entity_id"]: row["cluster"] for row in csv.DictReader(stream)}
    kmeans7 = kmeans7_labels(ids)
    explained = []
    for column, (label, log) in OUTCOMES.items():
        data = []
        for key in modal:
            raw = access.get(key) if column == "market_access" else rows.get(key, {}).get(column)
            if raw not in (None, ""):
                value = float(raw)
                if not log or value > 0:
                    data.append((math.log(value) if log else value, modal[key], kmeans[key], rows[key]["region_name"], kmeans7[key]))
        values = [d[0] for d in data]
        explained.append({"outcome": column, "label": label, "n": len(data),
                          "eta2_groups": round(eta_squared(values, [d[1] for d in data]), 4),
                          "eta2_kmeans4": round(eta_squared(values, [d[2] for d in data]), 4),
                          "eta2_region": round(eta_squared(values, [d[3] for d in data]), 4),
                          "eta2_kmeans7": round(eta_squared(values, [d[4] for d in data]), 4),
                          "epsilon2_groups": round(epsilon_squared(values, [d[1] for d in data]), 4),
                          "epsilon2_kmeans7": round(epsilon_squared(values, [d[4] for d in data]), 4),
                          "groups_minus_kmeans7_ci95": bootstrap_difference(values, [d[1] for d in data], [d[4] for d in data], [d[3] for d in data])})

    ever: dict = {}
    for i, key in enumerate(ids):
        for t in range(len(periods)):
            if tracks[t][i] in names:
                ever.setdefault(names[tracks[t][i]], set()).add(key)
    north = {key for key in ids if rows[key]["region_name"] in NORTH}
    reached = {key for key in north if any(key in ever[name] for name in NORTH_GROUPS)}
    north_check = {"regions": list(NORTH), "groups": list(NORTH_GROUPS), "north_municipalities": len(north),
                   "north_in_north_groups": len(reached),
                   "north_months_in_north_groups": round(sum(tracks[t][i] in names and names[tracks[t][i]] in NORTH_GROUPS for i, key in enumerate(ids) if key in north for t in range(len(periods))) / (len(north) * len(periods)), 4), "north_share_overall": round(len(north) / len(ids), 4),
                   "north_share_by_group": {name: round(len(members & north) / len(members), 4) for name, members in ever.items()}}

    out = ROOT / cfg["output"] / "profiles.json"
    out.write_text(json.dumps({"rule": "Mirkin: (group mean share - overall mean share) / overall mean share over municipality-months",
                               "threshold": args.threshold, "overall_shares": {n: round(o, 4) for n, o in zip(CATEGORIES, overall)},
                               "groups": profiles, "assignment": "modal stable group over 24 months",
                               "assigned": len(modal), "explained": explained, "north": north_check}, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({"groups": [(p["name"], p["salient"]) for p in profiles], "explained": explained, "north": north_check}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
