"""Audit network indices against a hash-pinned independent implementation.

No fitting, and no claim that Pattern is the contest's official specification.
The reference is supplied explicitly, read only, and verified before import.
Undefined ratios are recorded separately from numerical discrepancies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sbercluster.features import make_slices
from sbercluster.graph import knn_graph
from sbercluster.metrics import network_indices, network_quality_candidates

PATTERN_COMMIT = "e5ff50b34a1bbd723b9ae2e46c122a06e782daea"
PATTERN_SHA256 = "f0de1764428b6d99f36dd85a3d17b64516ba6fcc1b4543c98883cb2dff4c90bb"
PATTERN_URL = (
    f"https://raw.githubusercontent.com/Sorooshi/Pattern/{PATTERN_COMMIT}"
    "/metrics/clustering_metrics.py"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path: Path):
    if digest(path) != PATTERN_SHA256:
        raise ValueError("Pattern source differs from the explicitly audited SHA-256")
    spec = importlib.util.spec_from_file_location("_audited_pattern_metrics", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the supplied Pattern source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdjacencyClusteringMetrics().get_metric


def compare_partition(adjacency, labels, reference, *, binary=True, tolerance=5e-12):
    """Compare on the common mathematical domain; preserve strict 0/0 status.

    Pattern returns zero for a zero denominator. Our published contract returns
    None for AVI/AVU in that case, so those rows must not pass a numerical gate.
    TurboMQ is checked by its algebraic identity with K*AVI, not by pretending
    Pattern implements MQ (its audited source has no such field).
    """
    a = np.asarray(adjacency.toarray() if hasattr(adjacency, "toarray") else adjacency, dtype=float)
    z = np.asarray(labels)
    if binary:
        a = (a > 0).astype(float)
    actual = {**network_indices(a, z, binary=False), **network_quality_candidates(a, z, binary=False)}
    expected = reference(a, z)
    comparisons = {}
    failures = []
    for ours, theirs in (("AVI", "AVI"), ("AVU", "AVU"), ("NewmanQ", "modularity")):
        value, target = actual[ours], float(expected[theirs])
        if value is None:
            comparisons[ours] = {"status": "undefined_by_strict_contract", "reference_zero_convention": target}
        else:
            error = abs(value - target)
            passed = bool(np.isfinite(error) and error <= tolerance)
            comparisons[ours] = {"status": "pass" if passed else "fail", "actual": value,
                                 "reference": target, "absolute_error": error}
            if not passed:
                failures.append(ours)
    k = len(np.unique(z))
    if actual["AVI"] is not None and actual["TurboMQ"] is not None:
        target = k * actual["AVI"]
        error = abs(actual["TurboMQ"] - target)
        passed = bool(np.isfinite(error) and error <= tolerance)
        comparisons["TurboMQ_identity"] = {"status": "pass" if passed else "fail", "actual": actual["TurboMQ"],
                                           "K_times_AVI": target, "absolute_error": error}
        if not passed:
            failures.append("TurboMQ_identity")
    else:
        comparisons["TurboMQ_identity"] = {"status": "undefined_AVI_or_K"}
    return {"binary": binary, "n": len(z), "occupied_k": k, "comparisons": comparisons,
            "failures": failures, "AVI_undefined_clusters": actual.get("AVI_undefined_clusters"),
            "AVU_undefined_ordered_pairs": actual.get("AVU_undefined_ordered_pairs")}


def annual_audit(reference, root=ROOT):
    panel_path = root / "data/processed/panel.csv"
    manifest = json.loads((panel_path.parent / "manifest.json").read_text("utf-8"))
    if digest(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Panel differs from the recorded immutable snapshot")
    config_path = root / "configs/research_annual.json"
    assignments_path = root / "reports/experiments/2026-09-23-v2/screen_partitions.csv.gz"
    cfg = json.loads(config_path.read_text("utf-8"))
    panel = pd.read_csv(panel_path, dtype={"entity_id": str, "period": str, "oktmo": str})
    slices, _ = make_slices(panel[panel.period.str.startswith("2023")], cfg)
    ids = slices[0][1]
    annual = np.median(np.stack([s[2] for s in slices]), axis=0)
    graph, _ = knn_graph(annual, 15)
    assignments = pd.read_csv(assignments_path, dtype={"entity_id": str})
    assignments = assignments[assignments.representation.eq("annual_2023")]
    results = []
    for candidate, part in assignments.groupby("candidate", sort=True):
        if part.entity_id.duplicated().any() or set(part.entity_id) != set(ids):
            raise ValueError("Published labels do not join exactly: " + candidate)
        labels = part.set_index("entity_id").reindex(ids).cluster.to_numpy(dtype=int)
        row = compare_partition(graph, labels, reference)
        row["candidate"] = candidate
        results.append(row)
    inputs = {str(p.relative_to(root)): digest(p) for p in (panel_path, config_path, assignments_path)}
    return results, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-source", type=Path, required=True, help=f"Download and inspect {PATTERN_URL}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args(argv)
    reference = load_reference(args.pattern_source)
    path_graph = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float)
    fixtures = []
    for binary in (True, False):
        fixture = compare_partition(path_graph, [5, 5, 9, 9], reference, binary=binary)
        fixture["name"] = "four_node_path"
        fixtures.append(fixture)
    disconnected = path_graph.copy()
    disconnected[1, 2] = disconnected[2, 1] = 0
    fixture = compare_partition(disconnected, [0, 0, 1, 1], reference)
    fixture["name"] = "zero_between_edges_explicit_undefined_AVU"
    fixtures.append(fixture)
    random = np.random.default_rng(20261003)
    for k in (2, 3, 4, 7):
        for binary in (True, False):
            a = random.uniform(0.05, 3, size=(4 * k, 4 * k))
            a = np.triu(a, 1)
            a += a.T
            fixture = compare_partition(a, np.repeat(np.arange(k), 4), reference, binary=binary)
            fixture["name"] = f"positive_weighted_complete_K{k}"
            fixtures.append(fixture)
    annual, inputs = ([], {}) if args.fixtures_only else annual_audit(reference)
    rows = fixtures + annual
    failures = [{"name": r.get("candidate", r.get("name")), "metrics": r["failures"]}
                for r in rows if r["failures"]]
    errors = [v["absolute_error"] for r in rows for v in r["comparisons"].values() if "absolute_error" in v]
    report = {
        "schema_version": 1, "status": "pass" if not failures else "fail", "scope": "independent_reference_audit_no_fitting",
        "official_MQ_definition_confirmed": False,
        "reference": {"url": PATTERN_URL, "commit": PATTERN_COMMIT, "sha256": PATTERN_SHA256,
                      "relation_to_contest": "author software; not an official contest specification",
                      "MQ_implemented_in_reference": False},
        "conventions": {"strict_undefined_ratios": True, "comparison_tolerance": 5e-12,
                        "annual_graph": "common annual2023 attribute union15NN; binary"},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                          (Path(__file__), ROOT / "sbercluster/metrics.py", ROOT / "sbercluster/features.py", ROOT / "sbercluster/graph.py")},
        "input_hashes": inputs, "fixtures": fixtures, "annual_partitions": annual,
        "summary": {"fixture_count": len(fixtures), "annual_partition_count": len(annual), "failures": failures,
                    "max_absolute_error": max(errors, default=0.0),
                    "undefined_comparisons": sum(v["status"].startswith("undefined") for r in rows for v in r["comparisons"].values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["summary"], "output": str(args.output)}, ensure_ascii=False))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
