"""Build a self-contained, read-only dashboard from the prepared SberIndex panel.

Run explicitly: python archive/scripts/build_dashboard.py
The command never prepares data, fits a model, or changes its input files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from statistics import median

from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = (
    "Здоровье",
    "Маркетплейсы",
    "Общественное питание",
    "Продовольствие",
    "Транспорт",
    "Все категории",
)
META_COLUMNS = {"territory_id", "region_name", "region_code", "municipal_district_name", "oktmo"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_panel(path: Path) -> dict:
    entities: dict[str, dict] = {}
    periods: set[str] = set()
    seen: set[tuple[str, str]] = set()
    columns = {"entity_id", "mo", "period", *CATEGORIES, *META_COLUMNS}

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Panel missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            entity_id = (row["entity_id"] or "").strip()
            name = (row["mo"] or "").strip()
            period = (row["period"] or "").strip()
            if not entity_id or not name or not DATE_RE.fullmatch(period):
                raise ValueError(f"Invalid identity or period at line {line}")
            key = (entity_id, period)
            if key in seen:
                raise ValueError(f"Repeated entity_id/period at line {line}: {key}")
            seen.add(key)
            periods.add(period)
            territory_id = (row["territory_id"] or "").strip()
            region = (row["region_name"] or "").strip()
            region_code = (row["region_code"] or "").strip()
            if entity_id != f"tid_{territory_id}" or not region or not region_code:
                raise ValueError(f"Invalid canonical territory identity at line {line}")
            entity = entities.setdefault(entity_id, {"id": entity_id, "territory_id": territory_id,
                "name": name, "region": region, "region_code": region_code, "rows": {}, "history": {}})
            if entity["name"] != name or entity["region"] != region or entity["region_code"] != region_code:
                raise ValueError(f"Entity ID maps to multiple names or regions: {entity_id}")
            year = period[:4]
            history = {"district": (row["municipal_district_name"] or "").strip(),
                       "oktmo": (row["oktmo"] or "").strip()}
            if year in entity["history"] and entity["history"][year] != history:
                raise ValueError(f"Changing registry record within year at line {line}")
            entity["history"][year] = history

            values = []
            for category in CATEGORIES:
                raw = (row[category] or "").strip()
                if not raw:
                    values.append(None)
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Invalid {category} at line {line}: {raw}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {category} at line {line}")
                values.append(value)
            entity["rows"][period] = values

    if not entities:
        raise ValueError("Panel is empty")
    dates = sorted(periods)
    medians: list[list[float | None]] = []
    counts: list[list[int]] = []
    missing_cells = 0
    for period in dates:
        row_medians = []
        row_counts = []
        for category_index in range(len(CATEGORIES)):
            sample = [entity["rows"][period][category_index]
                      for entity in entities.values() if period in entity["rows"]
                      and entity["rows"][period][category_index] is not None]
            row_medians.append(median(sample) if sample else None)
            row_counts.append(len(sample))
        medians.append(row_medians)
        counts.append(row_counts)
    output_entities = []
    incomplete_entities = 0
    for entity in sorted(entities.values(), key=lambda item: (item["name"].casefold(), item["region"].casefold(), item["id"])):
        values = [entity["rows"].get(period, [None] * len(CATEGORIES)) for period in dates]
        missing_cells += sum(value is None for row in values for value in row)
        incomplete_entities += len(entity["rows"]) != len(dates)
        output_entities.append({"id": entity["id"], "tid": entity["territory_id"],
                                "name": entity["name"], "region": entity["region"],
                                "region_code": entity["region_code"], "history": entity["history"],
                                "values": values})
    return {
        "categories": CATEGORIES,
        "periods": dates,
        "entities": output_entities,
        "median": medians,
        "median_counts": counts,
        "summary": {
            "entities": len(entities),
            "periods": len(dates),
            "rows": len(seen),
            "expected_rows": len(entities) * len(dates),
            "incomplete_entities": incomplete_entities,
            "missing_cells": missing_cells,
        },
    }


def read_neighbors(graph_dir: Path, entities: list[dict]) -> dict:
    period = "2024-12-01"
    nodes = json.loads((graph_dir / f"nodes_{period}.json").read_text(encoding="utf-8"))
    graph = load_npz(graph_dir / f"{period}.npz").tocsr()
    if graph.shape != (len(nodes), len(nodes)) or len(nodes) != len(entities) or len(set(nodes)) != len(nodes):
        raise ValueError("December graph dimensions or node identities do not match the panel")
    by_id = {entity["id"]: index for index, entity in enumerate(entities)}
    if set(nodes) != set(by_id):
        raise ValueError("December graph node IDs differ from panel entity IDs")
    neighbors = [[] for _ in entities]
    for source_index, source_id in enumerate(nodes):
        row = graph.getrow(source_index)
        ranked = sorted(((int(target), float(weight)) for target, weight in zip(row.indices, row.data)
                         if target != source_index and math.isfinite(weight) and weight > 0),
                        key=lambda pair: (-pair[1], nodes[pair[0]]))[:15]
        neighbors[by_id[source_id]] = [[by_id[nodes[target]], weight] for target, weight in ranked]
    return {"period": period, "top": neighbors}


def build(panel_path: Path, audit_path: Path, graph_dir: Path, graph_report_path: Path,
          template_path: Path, output_path: Path) -> None:
    payload = read_panel(panel_path)
    with audit_path.open("r", encoding="utf-8-sig") as stream:
        payload["audit"] = json.load(stream)
    with graph_report_path.open("r", encoding="utf-8-sig") as stream:
        report = json.load(stream)
    payload["neighbors"] = read_neighbors(graph_dir, payload["entities"])
    payload["graph_summary"] = next(item for item in report["graphs"]
                                    if item["period"] == payload["neighbors"]["period"])
    if report.get("clustering_executed") is not False:
        raise ValueError("Graph report does not confirm no clustering")
    template = template_path.read_text(encoding="utf-8")
    marker = "/*__DATA__*/null"
    if template.count(marker) != 1:
        raise ValueError("Template must contain exactly one data marker")
    # JSON is embedded in an HTML script element, so protect its closing tag.
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    rendered = template.replace(marker, data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output_path.parent,
                                         prefix=".dashboard-", suffix=".html", delete=False) as stream:
            stream.write(rendered)
            temporary_name = stream.name
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f"Created {output_path} ({payload['summary']['entities']} territories, {payload['summary']['periods']} months)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/processed/panel.csv")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/data_audit.json")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/processed/graphs")
    parser.add_argument("--graph-report", type=Path, default=ROOT / "reports/graph_preparation.json")
    parser.add_argument("--template", type=Path, default=ROOT / "web/template.html")
    parser.add_argument("--output", type=Path, default=ROOT / "web/index.html")
    args = parser.parse_args()
    build(args.panel, args.audit, args.graph_dir, args.graph_report, args.template, args.output)


if __name__ == "__main__":
    main()
