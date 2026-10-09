"""Build the self-contained research atlas from existing, read-only results.

This command does no fitting and does not modify any source artifact.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import csv
import itertools
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from statistics import median
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbercluster.temporal_groups import LIFECYCLE, event_owners, flows, rejoin, track_key, track_labels
CATEGORIES = ("Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт")
TOTAL = "Все категории"
PILOT_PERIOD = "2023-12-01"
VALIDATION_FILES = (
    "frozen_prototypes.json", "reference_assignments.csv", "monthly_assignments.csv",
    "cluster_profiles.csv", "consensus_neighbors.csv", "peer_validation.json",
    "external_market_access.json", "confirmation_metrics.json",
)


def _finite(value: str, context: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {context}: {value!r}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Expected {'positive ' if positive else ''}finite number in {context}")
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def read_panel(path: Path, scaler_path: Path) -> dict:
    report = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = report.get("scaler", {})
    center = scaler.get("ratio_center")
    iqr = scaler.get("ratio_iqr")
    if not isinstance(center, list) or not isinstance(iqr, list) or len(center) != 5 or len(iqr) != 5:
        raise ValueError("Graph report has no five-dimensional ratio scaler")
    center = [_finite(v, "ratio_center") for v in center]
    iqr = [_finite(v, "ratio_iqr", positive=True) for v in iqr]

    required = {"entity_id", "territory_id", "period", "mo", "region_name", TOTAL, *CATEGORIES}
    rows = _read_csv(path)
    entities: dict[str, dict] = {}
    periods: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for line, row in enumerate(rows, 2):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Panel missing columns: {', '.join(sorted(missing))}")
        entity_id, period = row["entity_id"].strip(), row["period"].strip()
        key = (entity_id, period)
        if not entity_id or key in seen:
            raise ValueError(f"Blank or duplicate panel key at line {line}: {key}")
        if entity_id != f"tid_{row['territory_id'].strip()}":
            raise ValueError(f"Canonical ID mismatch at line {line}")
        seen.add(key); periods.add(period)
        total = _finite(row[TOTAL], f"{key}/{TOTAL}", positive=True)
        values = [_finite(row[name], f"{key}/{name}", positive=True) for name in CATEGORIES]
        ratios = [value / total for value in values]
        features = [(math.log(ratio) - center[i]) / iqr[i] / math.sqrt(5) for i, ratio in enumerate(ratios)]
        entity = entities.setdefault(entity_id, {
            "id": entity_id, "tid": row["territory_id"].strip(), "name": row["mo"].strip(),
            "region": row["region_name"].strip(), "rows": {},
        })
        if entity["name"] != row["mo"].strip() or entity["region"] != row["region_name"].strip():
            raise ValueError(f"Entity metadata changes across panel: {entity_id}")
        entity["rows"][period] = {"total": total, "ratios": ratios, "features": features}
    periods_sorted = sorted(periods)
    if len(periods_sorted) != 24 or periods_sorted[0] != "2023-01-01" or periods_sorted[-1] != "2024-12-01":
        raise ValueError("Expected the complete January 2023–December 2024 panel")
    if len(entities) != 2016:
        raise ValueError(f"Expected 2016 territories, found {len(entities)}")
    for entity_id, entity in entities.items():
        if set(entity["rows"]) != set(periods_sorted):
            raise ValueError(f"Incomplete period join for {entity_id}")
    output = []
    for entity in sorted(entities.values(), key=lambda e: (e["name"].casefold(), e["region"].casefold(), e["id"])):
        ordered = [entity["rows"][period] for period in periods_sorted]
        reference = entity["rows"][PILOT_PERIOD]
        development = [entity["rows"][period] for period in periods_sorted if period < "2024-01-01"]
        output.append({
            "id": entity["id"], "tid": entity["tid"], "name": entity["name"], "region": entity["region"],
            "totals": [round(row["total"], 6) for row in ordered],
            "ratios": [[round(100 * value, 6) for value in row["ratios"]] for row in ordered],
            "annual_ratios": [round(100 * median(row["ratios"][i] for row in development), 6) for i in range(5)],
            "features": [round(value, 8) for value in reference["features"]],
            "annual_features": [round(median(row["features"][i] for row in development), 8) for i in range(5)],
        })
    return {"periods": periods_sorted, "entities": output, "scaler": scaler}


def read_pilot(partitions_path: Path, metrics_path: Path, entity_ids: set[str]) -> dict:
    chosen: dict[str, int] = {}
    for line, row in enumerate(_read_csv(partitions_path), 2):
        if row.get("method") == "kmeans" and row.get("period") == PILOT_PERIOD:
            entity_id = row.get("entity_id", "").strip()
            if entity_id in chosen:
                raise ValueError(f"Duplicate pilot assignment at line {line}: {entity_id}")
            chosen[entity_id] = int(row["cluster"])
    if set(chosen) != entity_ids:
        raise ValueError("Pilot assignments do not join exactly to the panel")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    matching = [row for row in metrics if row.get("method") == "kmeans" and row.get("period") == PILOT_PERIOD]
    if len(matching) != 1:
        raise ValueError("Expected one December 2023 KMeans metric record")
    metric = matching[0]
    for name in ("SW", "CH", "AVI", "AVU"):
        _finite(metric[name], f"pilot metric {name}")
    return {
        "period": PILOT_PERIOD, "method": "KMeans", "labels": chosen,
        "metric": {key: metric.get(key) for key in ("n", "k", "min_cluster_size", "SW", "CH", "AVI", "AVU", "S_Dbw", "S_Dbw_status", "MQ", "MQ_status")},
        "cluster_sizes": dict(sorted(Counter(chosen.values()).items())),
    }


def _require_columns(rows: list[dict], columns: set[str], name: str) -> None:
    if not rows or not columns.issubset(rows[0]):
        raise ValueError(f"{name} missing columns: {', '.join(sorted(columns.difference(rows[0] if rows else {})))}")


def read_validation(directory: Path, entity_ids: set[str], periods: set[str]) -> dict:
    missing = [name for name in VALIDATION_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Validation run is incomplete; missing: {', '.join(missing)}")
    frozen = json.loads((directory / "frozen_prototypes.json").read_text("utf-8"))
    ids = frozen.get("ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != entity_ids:
        raise ValueError("Frozen prototype IDs do not join exactly to the panel")
    labels = frozen.get("reference_labels")
    if not isinstance(labels, list) or len(labels) != len(ids):
        raise ValueError("Frozen reference labels are not aligned to IDs")
    centers = frozen.get("centers")
    if not centers or any(len(row) != 5 for row in centers):
        raise ValueError("Frozen prototype centers must be five-dimensional")
    for row in centers:
        for value in row: _finite(value, "prototype center")

    reference = _read_csv(directory / "reference_assignments.csv")
    _require_columns(reference, {"entity_id", "cluster"}, "reference_assignments.csv")
    ref_ids = [row["entity_id"] for row in reference]
    if len(ref_ids) != len(set(ref_ids)) or set(ref_ids) != entity_ids:
        raise ValueError("Reference assignments do not join exactly to panel IDs")
    reference_map = {row["entity_id"]: int(row["cluster"]) for row in reference}

    monthly = _read_csv(directory / "monthly_assignments.csv")
    _require_columns(monthly, {"entity_id", "period", "cluster", "prototype_margin"}, "monthly_assignments.csv")
    monthly_seen: set[tuple[str, str]] = set()
    monthly_map: dict[str, list[int]] = {entity_id: [] for entity_id in sorted(entity_ids)}
    by_key = {}
    for row in monthly:
        key = (row["entity_id"], row["period"])
        if key in monthly_seen or row["entity_id"] not in entity_ids or row["period"] not in periods:
            raise ValueError(f"Invalid monthly assignment key: {key}")
        monthly_seen.add(key); by_key[key] = int(row["cluster"])
        _finite(row["prototype_margin"], f"prototype margin {key}")
    expected = {(entity, period) for entity in entity_ids for period in periods}
    if monthly_seen != expected:
        raise ValueError("Monthly assignments are not a complete panel join")
    for entity_id in monthly_map:
        monthly_map[entity_id] = [by_key[(entity_id, period)] for period in sorted(periods)]

    neighbor_rows = _read_csv(directory / "consensus_neighbors.csv")
    _require_columns(neighbor_rows, {"entity_id", "neighbor_id", "rank", "month_edge_frequency", "profile_distance_2023", "distance_km"}, "consensus_neighbors.csv")
    neighbors: dict[str, list[dict]] = {entity_id: [] for entity_id in sorted(entity_ids)}
    neighbor_keys = set()
    for row in neighbor_rows:
        source, target = row["entity_id"], row["neighbor_id"]
        key = (source, int(row["rank"]))
        if source not in entity_ids or target not in entity_ids or source == target or key in neighbor_keys:
            raise ValueError(f"Invalid consensus neighbor row: {source}/{target}")
        neighbor_keys.add(key)
        neighbors[source].append({"id": target, "rank": int(row["rank"]),
            "frequency": _finite(row["month_edge_frequency"], "month edge frequency"),
            "profile_distance": _finite(row["profile_distance_2023"], "profile distance"),
            "distance_km": _finite(row["distance_km"], "distance km")})

    profiles = _read_csv(directory / "cluster_profiles.csv")
    _require_columns(profiles, {"cluster", "n", "category", "median_standardized_log_ratio", "q25", "q75"}, "cluster_profiles.csv")
    peer = json.loads((directory / "peer_validation.json").read_text("utf-8"))
    validate_neighbor_coverage(neighbors, peer)
    if reference_map != dict(zip(ids, labels)):
        raise ValueError("Reference assignments differ from frozen labels")
    market = json.loads((directory / "external_market_access.json").read_text("utf-8"))
    confirmation = json.loads((directory / "confirmation_metrics.json").read_text("utf-8"))
    if confirmation.get("candidate") != frozen.get("candidate"):
        raise ValueError("Confirmation candidate differs from frozen model")
    return {"confirmation": confirmation, "status": "validated", "candidate": frozen.get("candidate"), "assignment_rule": frozen.get("assignment_rule"),
            "prototype_fidelity": frozen.get("prototype_fidelity"), "labels": reference_map,
            "monthly": monthly_map, "neighbors": neighbors, "cluster_profiles": profiles,
            "peer_validation": peer, "external_market_access": market}


def validate_neighbor_coverage(neighbors, peer):
    active = {key for key, values in neighbors.items() if values}
    if len(active) != peer.get('n_common') or len(neighbors)-len(active) != peer.get('n_excluded'):
        raise ValueError('Peer coverage differs from validation cohort')
    for source, values in neighbors.items():
        values.sort(key=lambda row: row['rank'])
        if not values:
            continue
        targets = [row['id'] for row in values]
        if len(values)!=15 or [row['rank'] for row in values]!=list(range(1,16)) or len(set(targets))!=15:
            raise ValueError(f'Expected 15 distinct ranked consensus neighbors for {source}')
        if source in targets or not set(targets).issubset(active):
            raise ValueError('Consensus neighbor is outside the common cohort')


def read_reference(directory: Path, entity_ids: set[str]) -> dict:
    membership_path = directory / "membership_stability.json"
    labels_path = directory / "labels/kmeans_k4__reference.npy"
    published_labels = directory / 'reference_assignments.csv'
    if not membership_path.is_file() or not (labels_path.is_file() or published_labels.is_file()):
        raise ValueError("Reference run lacks membership IDs or frozen kmeans_k4 labels")
    membership = json.loads(membership_path.read_text("utf-8"))
    ids = membership.get("ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != entity_ids:
        raise ValueError("Reference IDs do not join exactly to the panel")
    if labels_path.is_file():
        labels = np.load(labels_path, allow_pickle=False)
    else:
        rows = _read_csv(published_labels)
        mapping = {row['entity_id']: int(row['cluster']) for row in rows}
        if len(rows) != len(mapping) or set(mapping) != entity_ids:
            raise ValueError('Published reference assignments do not join exactly')
        labels = np.asarray([mapping[key] for key in ids], dtype=int)
    if labels.ndim != 1 or len(labels) != len(ids) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Reference labels are not an aligned integer vector")
    unique = sorted(int(value) for value in np.unique(labels))
    if unique != list(range(4)):
        raise ValueError(f"Expected contiguous kmeans_k4 labels 0..3, found {unique}")
    metrics_path = directory / "reference_metrics.json"
    if not metrics_path.is_file():
        raise ValueError("Reference run lacks reference_metrics.json")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    selected = [row for row in metrics if row.get("candidate") == "kmeans_k4"]
    if len(selected) != 1:
        raise ValueError("Expected one kmeans_k4 reference metric record")
    metric = selected[0]
    for name in ("SW", "CH", "AVI", "AVU"):
        _finite(metric[name], f"reference metric {name}")
    return {"candidate": "kmeans_k4", "groups": 4,
            "profile": "median monthly standardized log-ratio profile for 2023",
            "metric": {key: metric.get(key) for key in ("n", "k", "min_cluster_size", "SW", "CH", "AVI", "AVU", "S_Dbw", "MQ", "MQ_status", "negative_silhouette_fraction")},
            "labels": {entity_id: int(label) for entity_id, label in zip(ids, labels)}}

def read_stability(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    target = path / "stability.json" if path.is_dir() else path
    if not target.is_file():
        raise ValueError(f"Stability result not found: {target}")
    value = json.loads(target.read_text("utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Stability result must be a non-empty JSON array")
    for index, row in enumerate(value):
        if not isinstance(row, dict) or not {"candidate", "kind", "parameter", "ARI", "k"}.issubset(row):
            raise ValueError(f"Invalid stability row {index}")
        ari = _finite(row["ARI"], f"stability ARI row {index}")
        if not -1 <= ari <= 1:
            raise ValueError(f"Stability ARI outside [-1, 1] at row {index}")
    return value


def encode_payload(payload: dict) -> str:
    """Serialize data safely for a classic HTML script element."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


FONT_MARKER = "/*__ATLAS_FONTS__*/"


def inline_fonts(template: str, fonts_dir: Path = ROOT / "web/fonts") -> str:
    """Replace the font marker with @font-face rules whose WOFF2 files are embedded as data URIs."""
    if FONT_MARKER not in template:
        return template
    if template.count(FONT_MARKER) != 1:
        raise ValueError("Template must contain at most one font marker")
    css = (fonts_dir / "fonts.css").read_text(encoding="utf-8")

    def embed(match: "re.Match[str]") -> str:
        data = base64.b64encode((fonts_dir / match.group(1)).read_bytes()).decode("ascii")
        return f"url(data:font/woff2;base64,{data})"

    return template.replace(FONT_MARKER, re.sub(r"url\(([\w.-]+\.woff2)\)", embed, css).strip())


def package_payload(payload: dict, *, compressed: bool = True) -> str:
    """Deterministic offline transport; decompression does not change scientific data."""
    raw = encode_payload(payload)
    if not compressed:
        return raw
    packed = gzip.compress(raw.encode("utf-8"), compresslevel=6, mtime=0)
    # gzip's OS byte varies across Python/zlib platforms; fix it for byte reproducibility.
    packed = packed[:9] + b"\xff" + packed[10:]
    return encode_payload({"encoding": "gzip-base64", "data": base64.b64encode(packed).decode("ascii")})


def read_contest(path: Path, entity_ids: set[str], map_path: Path | None = None) -> dict:
    target = path / 'atlas_extension.json' if path.is_dir() else path
    value = json.loads(target.read_text('utf-8'))
    for mode in ('raw','relative'):
        labels = value['assignments'][mode]
        if set(labels) != entity_ids or any(type(z) is not int or not 0 <= z < 4 for z in labels.values()):
            raise ValueError('Contest memberships do not join exactly')
        matrix = value['dynamics'][mode]['transition_matrix']
        if len(matrix) != 4 or any(len(row)!=4 for row in matrix):
            raise ValueError('Expected 4 by 4 transition matrix')
        if any(type(x) is not int or x<0 for row in matrix for x in row) or sum(map(sum,matrix))!=len(entity_ids):
            raise ValueError('Transition counts do not conserve the cohort')
    for candidate, labels in value.get('network',{}).get('labels',{}).items():
        if set(labels) != entity_ids or any(type(z) is not int or z<0 for z in labels.values()):
            raise ValueError('Network memberships do not join exactly: '+candidate)
    for name, model in value.get('map_models', {}).items():
        labels = model['labels']
        groups = {profile['id'] for profile in model['profiles']}
        if set(labels) != entity_ids or any(type(z) is not int or z not in groups for z in labels.values()):
            raise ValueError('Map model memberships do not join exactly: ' + name)
        if sum(profile['n'] for profile in model['profiles']) != len(entity_ids):
            raise ValueError('Map model sizes do not conserve the cohort: ' + name)
    if map_path:
        geometry = json.loads(map_path.read_text('utf-8'))
        keys = [p['id'] for p in geometry['paths']]
        if len(keys)!=len(set(keys)) or not set(keys).issubset(entity_ids):
            raise ValueError('Map has duplicate or unknown territories')
        if geometry['matched'] != len(keys):
            raise ValueError('Map coverage differs from actual paths')
        value['map'] = geometry
    encode_payload(value)  # Reject nonfinite values before touching the published HTML.
    return value

EDGE_RULES = ("euclid_profile", "cosine_shares", "pearson_series", "lagged_correlation", "dtw_series", "road_distance")
ICVI_KEYS = ("SW", "CH", "S_Dbw", "S_Dbw_paper", "AVI", "AVU", "MQ")


def _rounded(value, digits: int = 6):
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Edge-rule results contain a nonfinite value")
        return round(value, digits)
    return value


def _aligned_groups(labels: list[int], reference: list[int]) -> tuple[list[int], list[dict]]:
    groups = sorted(set(labels))
    if groups != list(range(4)):
        raise ValueError(f"Expected four spectral groups 0..3, found {groups}")
    overlap = [[0] * 4 for _ in range(4)]
    for z, r in zip(labels, reference):
        overlap[z][r] += 1
    perm = max(itertools.permutations(range(4)), key=lambda p: (sum(overlap[g][p[g]] for g in range(4)), [-x for x in p]))
    sizes = Counter(labels)
    groups = []
    for g in range(4):
        top = max(range(4), key=lambda r: (overlap[g][r], -r))
        groups.append({"id": perm[g], "n": sizes[g], "share": round(overlap[g][perm[g]] / sizes[g], 4),
                       "top": top, "top_share": round(overlap[g][top] / sizes[g], 4)})
    return [perm[z] for z in labels], sorted(groups, key=lambda row: row["id"])


def lifecycle_tracks(block: dict, members: list[dict[str, list[str]]] | None = None) -> dict:
    sizes, steps = block["sizes"], block["steps"]
    owners, marks = event_owners(sizes, steps)
    tracks = []
    for owned in owners:
        row = {"start": min(owned), "sizes": [sizes[t][owned[t]] if t in owned else 0 for t in range(len(sizes))]}
        if members is not None:
            counts = Counter(profile for t, label in owned.items() for profile in members[t][label])
            profile, n = max(counts.items(), key=lambda item: (item[1], -item[0]))
            row.update(profile=profile, share=round(n / sum(counts.values()), 4))
        else:
            labels = set(owned.values())
            if len(labels) != 1:
                raise ValueError("Fixed profile changed identity across months")
            row.update(profile=int(labels.pop()), share=1.0)
        tracks.append(row)
    totals = Counter()
    for step in steps:
        totals.update(step["counts"])
    return {"tracks": tracks, "marks": marks, "counts": {key: totals[key] for key in sorted(LIFECYCLE)},
            "groups": [len(month) for month in sizes]}


def read_edge_rules(directory: Path, order: list[str], periods: list[str], validation: dict) -> dict:
    ids = set(order)
    report = json.loads((directory / "edge_rules.json").read_text("utf-8"))
    rules = {row["rule"]: row for row in report["rules"]}
    if report.get("n") != len(order) or set(rules) != set(EDGE_RULES) or len(report["rules"]) != len(EDGE_RULES):
        raise ValueError("Edge-rule comparison must cover the six registered rules on the full cohort")
    jaccard = [[report["edge_jaccard"][a][b] for b in EDGE_RULES] for a in EDGE_RULES]
    if any(abs(jaccard[i][j] - jaccard[j][i]) > 1e-12 or not 0 <= jaccard[i][j] <= 1 for i in range(6) for j in range(6)) or any(jaccard[i][i] != 1 for i in range(6)):
        raise ValueError("Edge overlap matrix must be a symmetric Jaccard matrix")
    keep = ("rule", "edges", "components", "isolates", "mean_degree", "transitivity", "within_region", "within_profile",
            "spectral", "leiden", "reference_on_graph", "edge_lag_counts", "sigma", "bandwidth_km")
    icvi = json.loads((directory / "icvi_table.json").read_text("utf-8"))
    for row in icvi:
        if not {"partition", "k", "min_cluster_size", "ARI_reference", *ICVI_KEYS}.issubset(row):
            raise ValueError(f"ICVI row is incomplete: {row.get('partition')}")
        row["k"], row["min_cluster_size"] = int(row["k"]), int(row["min_cluster_size"])
    if len({row["partition"] for row in icvi}) != len(icvi):
        raise ValueError("Duplicate partition in ICVI table")

    reference = validation["labels"]
    with gzip.open(directory / "edge_rule_partitions.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["entity_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != ids:
        raise ValueError("Edge-rule partitions do not join exactly to the panel")
    partitions = {}
    for rule in EDGE_RULES:
        labels, groups = _aligned_groups([int(by_id[key][f"{rule}_spectral"]) for key in order], [reference[key] for key in order])
        partitions[rule] = {"labels": "".join(map(str, labels)), "groups": groups}

    events = json.loads((directory / "cluster_events.json").read_text("utf-8"))
    config = json.loads((directory / "provenance.json").read_text("utf-8"))["config"]["events"]
    monthly = validation["monthly"]
    fixed_sizes = [dict(Counter(str(monthly[key][t]) for key in order)) for t in range(len(periods))]
    if events["fixed_profiles"]["sizes"] != fixed_sizes:
        raise ValueError("Fixed-profile lifecycle sizes differ from the frozen monthly assignments")
    with gzip.open(ROOT / config["temporal_labels"], "rt", encoding="utf-8") as stream:
        temporal = json.load(stream)
    if set(temporal["ids"]) != ids or len(temporal["ids"]) != len(ids) or temporal["periods"] != periods:
        raise ValueError("Temporal Leiden labels do not match the panel cohort")
    members = []
    for row in temporal["labels"]:
        month: dict[str, list[int]] = {}
        for key, label in zip(temporal["ids"], row):
            month.setdefault(str(label), []).append(reference[key])
        members.append(month)
    if events["temporal_leiden"]["sizes"] != [{label: len(values) for label, values in month.items()} for month in members]:
        raise ValueError("Temporal lifecycle sizes differ from the stored Leiden labels")
    deseasonalized = events["fixed_profiles_deseasonalized"]
    if len(deseasonalized["sizes"]) != len(periods) or any(sum(month.values()) != len(order) or not set(month) <= set(fixed_sizes[0]) for month in deseasonalized["sizes"]):
        raise ValueError("Deseasonalized fixed-profile sizes must cover the cohort with the frozen profile labels")
    agreement = json.loads((directory / "icvi_agreement.json").read_text("utf-8"))
    counts = {"all": len(icvi), "k4": sum(row["k"] == 4 for row in icvi)}
    for scope, n in counts.items():
        matrix = agreement[scope]["spearman_oriented"]
        if agreement[scope]["n"] != n or set(matrix) != set(ICVI_KEYS) or any(set(matrix[a]) != set(ICVI_KEYS) or matrix[a][a] != 1 or abs(matrix[a][b] - matrix[b][a]) > 1e-9 for a in ICVI_KEYS for b in ICVI_KEYS):
            raise ValueError(f"ICVI agreement matrix '{scope}' does not match the ICVI table")
    pareto = agreement["pareto_k4_sw_mq"]
    eligible = {row["partition"] for row in icvi if row["k"] == 4 and row["min_cluster_size"] >= pareto["min_cluster_size"]}
    if pareto["candidates"] != len(eligible) or not set(pareto["front"]) <= eligible or not pareto["front"]:
        raise ValueError("SW x MQ Pareto front does not match the ICVI table")
    icvi_agreement = {"keys": list(ICVI_KEYS), **{scope: {"n": agreement[scope]["n"], "rho": [[agreement[scope]["spearman_oriented"][a][b] for b in ICVI_KEYS] for a in ICVI_KEYS]} for scope in counts},
                      "pareto_k4_sw_mq": {key: pareto[key] for key in ("min_cluster_size", "candidates", "front")}}
    source = re.search(r"omega([\d.]+)_seed(\d+)", config["temporal_labels"])
    lifecycle = {"fixed_profiles": lifecycle_tracks(events["fixed_profiles"]),
                 "fixed_profiles_deseasonalized": lifecycle_tracks(deseasonalized),
                 "temporal_leiden": {**lifecycle_tracks(events["temporal_leiden"], members),
                                     "omega": float(source.group(1)), "seed": int(source.group(2))},
                 "jaccard_threshold": config["jaccard_threshold"]}
    value = _rounded({"n": report["n"], "k": report["k"], "series_months": report["series_months"], "rules": [
        {key: rules[name][key] for key in keep if key in rules[name]} for name in EDGE_RULES],
        "jaccard": jaccard, "k_sensitivity": json.loads((directory / "k_sensitivity.json").read_text("utf-8")),
        "icvi": icvi, "icvi_agreement": icvi_agreement, "partitions": partitions, "lifecycle": lifecycle})
    encode_payload(value)
    return value


def read_temporal_groups(directory: Path, order: list[str], periods: list[str], validation: dict) -> dict:
    summary = json.loads((directory / "summary.json").read_text("utf-8"))
    with gzip.open(ROOT / summary["source"], "rt", encoding="utf-8") as stream:
        temporal = json.load(stream)
    if sorted(temporal["ids"]) != sorted(order) or temporal["periods"] != periods or summary["periods"] != periods or summary["n"] != len(order):
        raise ValueError("Temporal group summary does not match the panel cohort")
    block = json.loads((ROOT / summary["events"]).read_text("utf-8"))["temporal_leiden"]
    owners, _ = event_owners(block["sizes"], block["steps"])
    tracks = rejoin(track_labels(temporal["labels"], owners), summary["rejoin_jaccard"])
    reference = [validation["labels"][key] for key in temporal["ids"]]
    ids = sorted({track for row in tracks for track in row})
    index = {track: k for k, track in enumerate(ids)}
    rows = []
    for track in ids:
        types = Counter(reference[i] for row in tracks for i, value in enumerate(row) if value == track)
        rows.append({"key": track_key(track, owners, periods), "sizes": [row.count(track) for row in tracks],
                     "type": max(types, key=lambda c: (types[c], -c)), "share": round(max(types.values()) / sum(types.values()), 4)})
    named = {group["key"]: group for group in summary["groups"]}
    for row in rows:
        group = named.pop(row["key"], None)
        if group:
            if group["sizes"] != row["sizes"] or group["top_type"] != row["type"]:
                raise ValueError(f"Temporal group {row['key']} differs from summary.json")
            row.update(name=group["name"], territories=group["territories"], core=group["core"])
    if named:
        raise ValueError(f"summary.json names groups that the labels do not contain: {sorted(named)}")
    stable = sorted((k for k, row in enumerate(rows) if row.get("name")), key=lambda k: (rows[k]["type"], -sum(rows[k]["sizes"]), k))
    if len(stable) > 9:
        raise ValueError("At most nine stable temporal groups fit the one-character monthly encoding")
    code = {ids[k]: str(position) for position, k in enumerate(stable)}
    where = {key: i for i, key in enumerate(temporal["ids"])}
    monthly = ["".join(code.get(row[where[key]], ".") for key in order) for row in tracks]
    value = {"omega": summary["omega"], "seed": summary["seed"], "min_months": summary["min_months"], "groups": rows,
             "stable": stable, "monthly": monthly,
             "flows": [[[index[a], index[b], n] for (a, b), n in step.items()] for step in flows(tracks)],
             **{key: summary[key] for key in ("stable_cover", "ari_reference", "multi_group", "multi_group_share", "switches", "groups_per_month")}}
    encode_payload(value)
    return value


NETWORK_GROUP_NAMES = {0: "Районы и округа", 1: "Города и крупные районы", 2: "Москва и Петербург"}
NETWORK_ROLES = ("selected", "kmeans4_reference", "kmeans3", "spectral3_a0.75", "leiden_a1_g0.3", "leiden_a0_g0.3")
NETWORK_KEYS = ("k", "min_size", "seed_ari", "oot_rerun_ari", "ARI_kmeans4", "oot_SW", "oot_CH", "oot_S_Dbw", "oot_P_MQ",
                "oot_C_AVI", "oot_C_AVU", "oot_C_MQ", "oot_R_MQ")


def read_network_core(directory: Path, order: list[str]) -> dict:
    summary = json.loads((directory / "summary.json").read_text("utf-8"))
    profiles = json.loads((directory / "profiles.json").read_text("utf-8"))
    selected = summary["selected"]
    if profiles["variant"] != selected["variant"]:
        raise ValueError("Network-core profiles describe a different variant than summary.json")
    rows = _read_csv(directory / "assignments.csv")
    _require_columns(rows, {"entity_id", "group"}, "assignments.csv")
    labels = {row["entity_id"]: int(row["group"]) for row in rows}
    if len(labels) != len(rows) or set(labels) != set(order):
        raise ValueError("Network-core assignments do not join exactly to the panel")
    sizes = Counter(labels.values())
    if [sizes[g] for g in range(len(selected["sizes"]))] != selected["sizes"] or set(sizes) != set(NETWORK_GROUP_NAMES):
        raise ValueError("Network-core group sizes differ from summary.json")
    groups = []
    for group in profiles["groups"]:
        g = int(group["group"])
        if group["size"] != sizes[g]:
            raise ValueError(f"Network-core profile size differs for group {g}")
        groups.append({"id": g, "name": NETWORK_GROUP_NAMES[g], "n": group["size"], "population": group["median_population"],
                       "wage": group["median_wage_2023"], "municipal_types": group["municipal_type_share"],
                       "top_regions": group["top_regions"], "kmeans4": {int(k): v for k, v in group["kmeans4_counts"].items()},
                       "shares": group["median_share"]})
    by_variant = {row["variant"]: row for row in summary["comparison"]}
    by_variant["selected"] = by_variant[selected["variant"]]
    missing = [name for name in NETWORK_ROLES if name not in by_variant]
    if missing:
        raise ValueError(f"Network-core comparison lacks: {', '.join(missing)}")
    chosen = [row for row in _read_csv(directory / "candidates.csv") if row["variant"] == selected["variant"] and row["scope"] == "all"]
    if len(chosen) != 1:
        raise ValueError("Network-core candidates.csv must hold the selected variant once")
    k_per_seed = json.loads(chosen[0]["k_per_seed"])
    value = _rounded({"variant": selected["variant"], "alpha": selected["alpha"], "gamma": selected["gamma"],
                      "k_min": summary["selection_rule"]["k_min"], "seed_ari_min": selected["seed_ari_min"],
                      "seeds_other_k": sum(k != selected["k"] for k in k_per_seed),
                      "seeds": len(summary["grid"]["seeds"]), "labels": "".join(str(labels[key]) for key in order),
                      "groups": sorted(groups, key=lambda g: g["id"]),
                      "level2_rerun_ari": [summary["level2"]["groups"][g]["selected"]["oot_rerun_ari"] for g in sorted(summary["level2"]["groups"])],
                      "comparison": [{"id": name, **{key: by_variant[name][key] for key in NETWORK_KEYS}} for name in NETWORK_ROLES]})
    encode_payload(value)
    return value


QUALITY_SERIES = {"leiden": ("temporal_leiden", "0.01", "1729"), "kmeans_fixed": ("kmeans4_fixed_2023", "", "")}
QUALITY_KEYS = ("MQ", "MQ_next", "MQ_mix", "MQ_mix_next", "ARI_prev", "churn_prev", "k")
TRADEOFF_KEYS = ("omega", "MQ_mean", "MQ_seed_sd", "MQ_drop_vs_omega0", "MQ_mix_mean", "MQ_next_over_own_next", "ARI_adjacent_mean", "churn_mean",
                 "seed_ARI_mean", "seed_ARI_min", "k_min", "k_max", "selected")


def read_temporal_quality(directory: Path, periods: list[str]) -> dict:
    summary = json.loads((directory / "summary.json").read_text("utf-8"))
    rows = _read_csv(directory / "monthly_icvi.csv")
    _require_columns(rows, {"method", "omega", "seed", "period", *QUALITY_KEYS}, "monthly_icvi.csv")
    series = {}
    for name, (method, omega, seed) in QUALITY_SERIES.items():
        chosen = {row["period"]: row for row in rows if row["method"] == method and row["omega"] == omega and row["seed"] == seed}
        if sorted(chosen) != periods:
            raise ValueError(f"monthly_icvi.csv does not cover 24 months for {name}")
        series[name] = {key: [None if chosen[p][key] == "" else _finite(chosen[p][key], f"{name}/{key}") for p in periods]
                        for key in QUALITY_KEYS}
    methods = {(row["method"], row.get("omega"), row.get("seed")): row for row in summary["methods"]}
    means = {"leiden": methods[("temporal_leiden", 0.01, 1729)], "kmeans_fixed": methods[("kmeans4_fixed_2023", None, None)],
             "kmeans_refit": methods[("kmeans4_monthly_refit", None, 1729)]}
    tradeoff = [{key: row[key] for key in TRADEOFF_KEYS} for row in summary["omega_tradeoff"]]
    if sum(row["selected"] for row in tradeoff) != 1 or next(row["omega"] for row in tradeoff if row["selected"]) != summary["omega_choice"]["selected"]:
        raise ValueError("omega trade-off does not mark the selected omega")
    value = _rounded({"series": series, "means": {name: {key: row[key] for key in ("SW", "CH", "AVI", "AVU", "MQ", "MQ_next", "MQ_mix", "ARI_prev", "churn_prev", "k_min", "k_max", "MQ_next_over_own_next")}
                                                  for name, row in means.items()},
                      "tradeoff": tradeoff,
                      "own_next_by_seed": [row["MQ_next_over_own_next"] for row in summary["methods"]
                                           if row["method"] == "temporal_leiden" and row.get("omega") == summary["omega_choice"]["selected"]],
                      "rule": summary["omega_rule"]["max_relative_drop"], "selected": summary["omega_choice"]["selected"],
                      "mixed_choice": summary["omega_choice_sensitivity_mixed_graph"]["selected"]})
    encode_payload(value)
    return value


EXTERNAL_OUTCOMES = {"population": "population", "wage_2023": "wage", "share_public_services": "public_services",
                     "above_working_age_share_2023": "older"}


def read_external_municipal(directory: Path, contest: dict | None = None) -> dict:
    rows = _read_csv(directory / "summary.csv")
    _require_columns(rows, {"outcome", "cluster", "type", "n_observed", "median", "q25", "q75"}, "summary.csv")
    results = json.loads((directory / "results.json").read_text("utf-8"))
    profiles: dict[int, dict] = {}
    for row in rows:
        key = EXTERNAL_OUTCOMES.get(row["outcome"])
        if key is None:
            continue
        entry = profiles.setdefault(int(row["cluster"]), {"id": int(row["cluster"]), "name": row["type"]})
        entry[key] = {"median": float(row["median"]), "q25": float(row["q25"]), "q75": float(row["q75"]),
                      "n": int(row["n_observed"])}
    if sorted(profiles) != [0, 1, 2, 3] or any(set(EXTERNAL_OUTCOMES.values()) - set(entry) for entry in profiles.values()):
        raise ValueError("summary.csv must cover four types and every external indicator")
    if contest:
        names = {int(profile["id"]): profile["name"].removesuffix(" профиль") for profile in contest.get("profiles", [])}
        if any(names.get(cluster) != entry["name"] for cluster, entry in profiles.items()):
            raise ValueError("external municipal types do not match atlas profiles")
    eta = {}
    for effect in results["effects"]:
        if effect["subset"] == "all" and effect["controls"] in ("none", "region"):
            eta.setdefault(effect["outcome"], {})[effect["controls"]] = round(float(effect["eta_squared"]), 4)
    for outcome in ("population", "wage_2023"):
        if set(eta.get(outcome, {})) != {"none", "region"}:
            raise ValueError(f"results.json lacks eta squared for {outcome}")
    value = {"profiles": [profiles[cluster] for cluster in sorted(profiles)], "eta_squared": eta}
    encode_payload(value)
    return value


def build(args: argparse.Namespace) -> None:
    if args.validation_run and not args.reference_run:
        raise ValueError("--validation-run requires --reference-run for matching development metrics")
    panel = read_panel(args.panel, args.graph_report)
    ids = {entity["id"] for entity in panel["entities"]}
    pilot = read_pilot(args.partitions, args.metrics, ids)
    reference = read_reference(args.reference_run, ids) if args.reference_run else None
    validation = read_validation(args.validation_run, ids, set(panel["periods"])) if args.validation_run else None
    for entity in panel["entities"]:
        entity["pilot_cluster"] = pilot["labels"][entity["id"]]
        if reference:
            entity["annual_cluster"] = reference["labels"][entity["id"]]
        if reference or validation:
            entity["features"] = entity["annual_features"]
        if validation:
            entity["reference_cluster"] = validation["labels"][entity["id"]]
            entity["monthly_clusters"] = validation["monthly"][entity["id"]]
    status = "validation" if validation else "reference" if reference else "pilot"
    groups = len(set(validation["labels"].values())) if validation else reference["groups"] if reference else pilot["metric"]["k"]
    payload = {
        "meta": {"entities": len(panel["entities"]), "months": len(panel["periods"]),
                 "status": status, "groups": groups, "ratio_iqr": panel["scaler"]["ratio_iqr"], "reference_period": PILOT_PERIOD,
                 "profile_label": "годовой медианный профиль 2023" if reference or validation else "декабрь 2023"},
        "categories": list(CATEGORIES), "periods": panel["periods"], "entities": panel["entities"],
        "pilot": {key: value for key, value in pilot.items() if key != "labels"},
        "reference": {key: value for key, value in reference.items() if key != "labels"} if reference else None,
        "validation": validation, "stability": read_stability(args.stability_run or args.reference_run),
    }
    if getattr(args,'contest_run',None):
        if not validation:
            raise ValueError('--contest-run requires the frozen validation contract')
        payload['contest'] = read_contest(args.contest_run,ids,getattr(args,'map_data',None))
    if getattr(args, "external_municipal", None):
        payload["external_municipal"] = read_external_municipal(args.external_municipal, payload.get("contest"))
    if getattr(args, "edge_rules", None):
        if not validation:
            raise ValueError("--edge-rules requires the frozen validation contract")
        payload["edge_rules"] = read_edge_rules(args.edge_rules, [entity["id"] for entity in panel["entities"]], panel["periods"], validation)
    if getattr(args, "temporal_groups", None):
        if not validation:
            raise ValueError("--temporal-groups requires the frozen validation contract")
        payload["temporal_groups"] = read_temporal_groups(args.temporal_groups, [entity["id"] for entity in panel["entities"]], panel["periods"], validation)
    if getattr(args, "network_core", None):
        payload["network_core"] = read_network_core(args.network_core, [entity["id"] for entity in panel["entities"]])
    if getattr(args, "temporal_quality", None):
        payload["temporal_quality"] = read_temporal_quality(args.temporal_quality, panel["periods"])
    template = inline_fonts(args.template.read_text(encoding="utf-8"))
    marker = "/*__ATLAS_DATA__*/null"
    if template.count(marker) != 1:
        raise ValueError("Template must contain exactly one atlas data marker")
    encoded = package_payload(payload, compressed=not getattr(args, "uncompressed", False))

    rendered = template.replace(marker, encoded)
    # Presentation sources remain inline so the atlas also opens without a server.
    for placeholder, source in (
        ("<!--__ACCEPTED_RESULTS__-->", ROOT / "web/accepted_results.html"),
        ("<!-- PRACTICAL_CASES -->", ROOT / "web/practical_cases.html"),
        ("<!-- ICVI_RESULTS -->", ROOT / "web/icvi_results.html"),
        ("<!-- METHOD_COMPARISON -->", ROOT / "web/method_comparison.html"),
        ("<!-- CLUSTER_STABILITY -->", ROOT / "web/cluster_stability.html"),
    ):
        if placeholder in rendered:
            rendered = rendered.replace(placeholder, source.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=args.output.parent,
                                         prefix=".atlas-", suffix=".html", delete=False) as stream:
            stream.write(rendered); temporary = stream.name
        os.replace(temporary, args.output)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)
    print(f"Created {args.output} ({len(panel['entities'])} territories, {len(panel['periods'])} months, {payload['meta']['status']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/processed/panel.csv")
    parser.add_argument("--graph-report", type=Path, default=ROOT / "reports/graph_preparation.json")
    parser.add_argument("--partitions", type=Path, default=ROOT / "reports/experiments/2026-09-23/partitions.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/experiments/2026-09-23/metrics.json")
    parser.add_argument("--validation-run", type=Path, help="Directory containing the complete frozen validation contract")
    parser.add_argument("--reference-run", type=Path, help="Run directory with frozen annual kmeans_k4 labels and IDs")
    parser.add_argument("--stability-run", type=Path, help="Optional stability.json file or containing directory")
    parser.add_argument('--contest-run',type=Path,help='Validated contest extension directory or JSON file')
    parser.add_argument('--map-data',type=Path,help='Derived municipal SVG geometry JSON')
    parser.add_argument("--edge-rules", type=Path, help="Directory with the edge-rule comparison, ICVI table and lifecycle events")
    parser.add_argument("--temporal-groups", type=Path, help="Directory with summary.json of the stable temporal Leiden groups")
    parser.add_argument("--external-municipal", type=Path, help="Directory with summary.csv and results.json of the open municipal statistics check")
    parser.add_argument("--network-core", type=Path, help="Directory with summary.json, profiles.json and assignments.csv of the annual network typology")
    parser.add_argument("--temporal-quality", type=Path, help="Directory with summary.json and monthly_icvi.csv of the monthly partition quality")
    parser.add_argument("--template", type=Path, default=ROOT / "web/research_template.html")
    parser.add_argument("--uncompressed", action="store_true", help="Embed plain JSON instead of the default lossless gzip package")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/atlas-preview/index.html")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
