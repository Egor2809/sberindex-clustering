"""Build deterministic attributed monthly graphs, no clustering/training."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbercluster.features import make_slices
from sbercluster.graph import knn_graph
from sbercluster.io import write_json, sha256

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/pilot.json").read_text(encoding="utf-8"))
    panel_path = root / "data/processed/panel.csv"
    manifest = json.loads((panel_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Prepared panel changed")
    panel = pd.read_csv(panel_path, dtype={"entity_id":str, "period":str})
    started = time.perf_counter()
    slices, scaler = make_slices(panel, cfg)
    out = root / "data/processed/graphs"
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for period, ids, x in slices:
        a, info = knn_graph(x, cfg["graph"]["k"], cfg["graph"]["symmetry"])
        save_npz(out / f"{period}.npz", a)
        np.save(out / f"features_{period}.npy", x, allow_pickle=False)
        write_json(out / f"nodes_{period}.json", ids)
        summaries.append({"period":period,"nodes":len(ids),**info})
    write_json(root / "artifacts/preparation/graph_preparation.json", {
        "graphs":summaries, "scaler":scaler, "seconds":time.perf_counter()-started,
        "panel_sha256":sha256(panel_path), "config":cfg,
        "source_hashes":{str(p.relative_to(root)):sha256(p) for p in [Path(__file__),root/"sbercluster/features.py",root/"sbercluster/graph.py"]},
        "clustering_executed":False, "economic_types_found":False,
        "scope":"all observed months; model-only run_periods selection does not restrict graph preparation",
        "status":"deterministic similarity graphs; no partition optimisation",
        "boundary_scope":"source-defined IDs; transitions require additional review"})
    print(f"Prepared {len(slices)} sparse graphs; no clustering performed.")
