"""Count group switches and flip-backs on the real 24 months: temporal Leiden against monthly and frozen KMeans."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sbercluster.edge_study import load_panel, reference_partition
from sbercluster.synthetic_transitions import align, kmeans_monthly
from sbercluster.temporal_groups import event_owners, rejoin, track_labels
from sbercluster.temporal_quality import fixed_assignments

RULE = ("Временной Leiden выигрывает, если у него меньше возвратов на одно МО (МО уходит из группы и возвращается в неё "
        "не позже чем через три месяца), чем у каждого варианта KMeans: обучаемого заново каждый месяц с K = 4 и K = 7 "
        "и с фиксированными центрами KMeans4 2023 года (с сезонной поправкой и без неё).")
WINDOW = 3


def stats(labels):
    paths = np.asarray(labels).T
    switches = (paths[:, 1:] != paths[:, :-1]).sum(axis=1)
    flips = np.zeros(len(paths), dtype=int)
    for i, path in enumerate(paths):
        for t in range(1, len(path)):
            if path[t] != path[t - 1]:
                back = path[t + 1:t + 1 + WINDOW]
                flips[i] += bool((back == path[t - 1]).any())
    total = int(switches.sum())
    return {"switches": total, "switches_per_mo": round(total / len(paths), 3), "moved_mo": int((switches > 0).sum()),
            "flip_backs": int(flips.sum()), "flip_backs_per_mo": round(flips.sum() / len(paths), 3),
            "flip_back_share": round(flips.sum() / total, 3) if total else 0.0, "groups_per_month": [int(len(set(z))) for z in labels]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/temporal_quality.json")
    parser.add_argument("--groups", default="configs/temporal_groups.json")
    parser.add_argument("--output", default="reports/temporal-stability")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text("utf-8"))
    gcfg = json.loads((ROOT / args.groups).read_text("utf-8"))
    ids, periods, monthly, raw, _ = load_panel(cfg, ROOT)
    with gzip.open(ROOT / gcfg["temporal_labels"], "rt", encoding="utf-8") as stream:
        temporal = json.load(stream)
    if temporal["ids"] != ids:
        raise ValueError("Temporal labels are not aligned with the panel")
    events = json.loads((ROOT / gcfg["events"]).read_text("utf-8"))["temporal_leiden"]
    owners, _ = event_owners(events["sizes"], events["steps"])
    leiden = rejoin(track_labels(temporal["labels"], owners), gcfg["rejoin_jaccard"])
    months = sum(p <= cfg["features"]["calibration_end"] for p in periods)
    centers, _ = reference_partition(ROOT, ids, np.median(monthly[:months], axis=0))
    fixed_raw, fixed, _ = fixed_assignments(monthly, centers)
    x = np.asarray(monthly)
    methods = {
        "temporal_leiden": leiden,
        "kmeans4_monthly": kmeans_monthly(x, 4, cfg["seed"], 20),
        "kmeans7_monthly": kmeans_monthly(x, 7, cfg["seed"], 20),
        "kmeans4_fixed_seasonal": fixed,
        "kmeans4_fixed_raw": fixed_raw,
    }
    result = {name: stats([np.asarray(z) for z in labels]) for name, labels in methods.items()}
    base = result["temporal_leiden"]["flip_backs_per_mo"]
    wins = all(base < r["flip_backs_per_mo"] for name, r in result.items() if name != "temporal_leiden")
    summary = {"rule": RULE, "window_months": WINDOW, "n": len(ids), "months": len(periods), "methods": result,
               "leiden_wins": wins}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({k: (v["switches_per_mo"], v["flip_backs_per_mo"], v["flip_back_share"], v["moved_mo"]) for k, v in result.items()}, ensure_ascii=False))
    print("leiden_wins", wins)


if __name__ == "__main__":
    main()
