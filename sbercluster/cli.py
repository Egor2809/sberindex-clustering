from __future__ import annotations
import argparse
import json
import subprocess
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from .data import audit_and_prepare
from .features import make_slices, validate_contract
from .graph import knn_graph
from .io import write_json, sha256
from .input_contracts import validate_prepared_panel
from .metrics import all_metrics
from .dynamics import transitions
from .models import require_execution, fit_static, fit_temporal, temporal_scope


def plan(cfg, root):
    blockers = []
    try:
        validate_contract(cfg)
    except ValueError as exc:
        blockers.append(str(exc))
    if not cfg["execution"]["allow_clustering"]:
        blockers.append("Clustering is disabled in the current configuration")
    if 'leiden_temporal' in cfg['clustering']['methods']:
        try:
            temporal_scope(cfg)
        except ValueError as exc:
            blockers.append(str(exc))
    return {"status": "prepared_not_run", "model_training_executed": False, "blockers": blockers,
            "methods": cfg["clustering"]["methods"], "seed": cfg["seed"],
            "next": "Review the data requirements and experiment plan before enabling the pilot configuration",
            "full_comparison_plan": "configs/comparison_plan.json"}


def run(cfg, root, execute):
    # First operation, before reading data or importing optional fit backends.
    require_execution(cfg, execute)
    root = Path(root)
    started = perf_counter()
    validate_contract(cfg)
    methods = cfg['clustering']['methods']
    if (not isinstance(methods, (list, tuple)) or not methods
            or any(not isinstance(method, str) or method not in
                   ('kmeans', 'ward', 'leiden_static', 'leiden_temporal') for method in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError('Expected distinct supported clustering methods')
    if 'leiden_temporal' in methods:
        temporal_scope(cfg)
    panel_path = root / "data/processed/panel.csv"
    panel = pd.read_csv(panel_path, dtype={"entity_id": str, "period": str})
    manifest = json.loads((panel_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Prepared panel hash changed; re-audit first")
    validate_prepared_panel(panel, manifest, cfg)
    slices, scaler = make_slices(panel, cfg)
    if cfg.get("run_periods") is not None:
        periods = cfg['run_periods']
        if (not isinstance(periods, (list, tuple)) or not periods
                or any(not isinstance(period, str) for period in periods)
                or len(set(periods)) != len(periods)):
            raise ValueError('run_periods must contain distinct requested months')
        requested = set(periods)
        if not requested.issubset({s[0] for s in slices}):
            raise ValueError("Requested period absent from panel")
        slices = [s for s in slices if s[0] in requested]
    if "leiden_temporal" in cfg["clustering"]["methods"]:
        periods = [s[0] for s in slices]
        expected = pd.date_range(periods[0], periods[-1], freq="MS").strftime("%Y-%m-%d").tolist()
        if periods != expected:
            raise ValueError("Temporal coupling requires consecutive calendar months")
    if cfg["graph"]["bandwidth"] != "median_knn_distance":
        raise ValueError("Unsupported bandwidth configuration")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    from .research import tracked_run
    with tracked_run(out, stage='cli'):
        return _run_created(cfg, root, out, run_id, panel_path, slices, scaler, started)


def _run_created(cfg, root, out, run_id, panel_path, slices, scaler, started):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    code_files = list((root / "sbercluster").glob("*.py"))
    write_json(out / "provenance.json", {"config": cfg, "panel_sha256": sha256(panel_path), "scaler": scaler,
         "git_commit": commit, "source_hashes": {p.name: sha256(p) for p in code_files}, "started_at": run_id})
    graphs, graph_info, timings = [], [], []
    labels_by_method = {m: [] for m in cfg["clustering"]["methods"]}
    for period, ids, x in slices:
        a, info = knn_graph(x, cfg["graph"]["k"], cfg["graph"]["symmetry"])
        graphs.append(a)
        graph_info.append({"period": period, **info})
        save_npz(out / f"graph_{period}.npz", a)
        write_json(out / f"nodes_{period}.json", ids)
        np.save(out / f"features_{period}.npy", x, allow_pickle=False)
        for method in labels_by_method:
            if method != "leiden_temporal":
                fit_started = perf_counter()
                labels_by_method[method].append(fit_static(method, x, a, ids, cfg, execute=True))
                timings.append({"method": method, "period": period, "stage": "fit", "seconds": perf_counter() - fit_started})
                print(json.dumps({"stage": "fit_completed", **timings[-1]}), flush=True)
    if "leiden_temporal" in labels_by_method:
        labels_by_method["leiden_temporal"], temporal_info = fit_temporal(slices, graphs, cfg, execute=True)
        write_json(out / "temporal_metadata.json", temporal_info)
    metrics, partitions, change = [], [], []
    for method, label_slices in labels_by_method.items():
        for index, ((period, ids, x), a, labels) in enumerate(zip(slices, graphs, label_slices)):
            metric_started = perf_counter()
            try:
                metric = all_metrics(x, labels, a)
            except ValueError as exc:
                metric = {"status": "undefined", "reason": str(exc)}
            metrics.append({"method": method, "period": period, **metric})
            timings.append({"method": method, "period": period, "stage": "metrics", "seconds": perf_counter() - metric_started})
            partitions.extend({"method": method, "period": period, "entity_id": entity, "cluster": int(label)} for entity, label in zip(ids, labels))
            if index:
                change.append({"method": method, "from_period": slices[index-1][0], "to_period": period,
                    **transitions(slices[index-1][1], label_slices[index-1], ids, labels)})
    write_json(out / "metrics.json", metrics)
    write_json(out / "transitions.json", change)
    write_json(out / "graphs.json", graph_info)
    write_json(out / "timings.json", {"stages": timings, "total_seconds": perf_counter() - started})
    pd.DataFrame(partitions).to_csv(out / "partitions.csv", index=False)
    write_json(out / "status.json", {"status": "completed", "MQ_complete": True,
        "partition_metrics_complete": all(row.get('status') != 'undefined' for row in metrics),
        "undefined_partitions": sum(row.get('status') == 'undefined' for row in metrics),
        "economic_interpretation_verified": False, "competition_ready": False})
    return {"run": str(out), "status": "computed_not_economically_validated"}


def main(argv=None):
    p = argparse.ArgumentParser(description="SberIndex preparation is the default; models are explicitly gated.")
    p.add_argument("command", choices=["audit", "plan", "run"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--input", type=Path)
    p.add_argument("--execute-clustering", action="store_true")
    args = p.parse_args(argv)
    cfg = json.loads((args.root / args.config).read_text(encoding="utf-8"))
    if args.command == "audit":
        candidates = list(args.root.glob(cfg["input_glob"])) if args.input is None else [args.input]
        if len(candidates) != 1:
            p.error("Specify exactly one source with --input; duplicate exports are not concatenated")
        result = audit_and_prepare(candidates[0], args.root)
    elif args.command == "plan":
        result = plan(cfg, args.root)
        write_json(args.root / "reports/execution_plan.json", result)
    else:
        result = run(cfg, args.root, args.execute_clustering)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
