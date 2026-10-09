"""Prepare review-only 2024 municipal labour tables from official Rosstat XLSX.

The source workbooks do not contain OKTMO codes.  This script therefore keeps
the cleaned observations separate from a candidate crosswalk.  Candidates are
generated only by deterministic region + municipal type + exact/normalised
official-name rules and always require manual review before analytical use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

from openpyxl import load_workbook


ARK_URL = "https://29.rosstat.gov.ru/storage/mediabank/Trud12oktmo2024.xlsx"
BUR_HEADCOUNT_URL = (
    "https://03.rosstat.gov.ru/storage/mediabank/"
    "Среднесписочная_численность_работников%20МО_РБ_2024.xlsx"
)
BUR_WAGE_URL = (
    "https://03.rosstat.gov.ru/storage/mediabank/"
    "Среднемесячная_номинальная_з-п_МО_РБ_2024.xlsx"
)
SUPPRESSED_RE = re.compile(r"^\s*(?:…|\.{3})")

REVIEWED_MAPPING = {
    ("29", "Архангельск"): "tid_890",
    ("29", "Коряжма"): "tid_891",
    ("29", "Котлас"): "tid_892",
    ("29", "Новодвинск"): "tid_895",
    ("29", "Северодвинск"): "tid_896",
    ("29", "Верхнетоемский муниципальный округ"): "tid_898",
    ("29", "Вилегодский муниципальный округ"): "tid_899",
    ("29", "Виноградовский муниципальный округ"): "tid_900",
    ("29", "Каргопольский муниципальный округ"): "tid_901",
    ("29", "Котласский муниципальный округ"): "tid_903",
    ("29", "Мезенский муниципальный округ"): "tid_907",
    ("29", "Няндомский муниципальный округ"): "tid_908",
    ("29", "Плесецкий муниципальный округ"): "tid_911",
    ("29", "Устьянский муниципальный округ"): "tid_913",
    ("29", "Холмогорский муниципальный округ"): "tid_914",
    ("29", "Шенкурский муниципальный округ"): "tid_915",
    ("29", "Вельский муниципальный район"): "tid_897",
    ("29", "Коношский муниципальный район"): "tid_902",
    ("29", "Ленский муниципальный район"): "tid_905",
    ("29", "Онежский муниципальный район"): "tid_909",
    ("83", "Город Нарьян-Мар"): "tid_916",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_and_status(value: object) -> tuple[float | int | None, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value, "observed"
    if SUPPRESSED_RE.match(str(value)):
        return None, "suppressed"
    if str(value).strip() == "-":
        return None, "not_applicable_or_absent"
    raise ValueError(f"Unexpected numeric value: {value!r}")


def source_type(name: str) -> str:
    folded = name.casefold()
    if "муниципальный округ" in folded:
        return "муниципальный округ"
    if "муниципальный район" in folded:
        return "муниципальный район"
    if folded.startswith("г.") or folded.startswith("город "):
        return "город (тип уровня не указан в источнике)"
    return "городской округ"


def parse_arkhangelsk(path: Path) -> list[dict[str, object]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if workbook.sheetnames != ["oktmo_12"]:
            raise ValueError(f"Unexpected sheets: {workbook.sheetnames}")
        sheet = workbook["oktmo_12"]
        title = str(sheet["A1"].value or "")
        if "январь-декабрь 2024 года" not in title:
            raise ValueError(f"Unexpected period in title: {title!r}")
        region_name = ""
        records: list[dict[str, object]] = []
        for row_number, row in enumerate(sheet.iter_rows(values_only=True), 1):
            label = row[0]
            if not isinstance(label, str):
                continue
            if label.startswith("Муниципальные образования Архангельской области"):
                region_name = "Архангельская область"
                region_code = "29"
                continue
            if label.startswith("Муниципальные образования Ненецкого автономного округа"):
                region_name = "Ненецкий автономный округ"
                region_code = "83"
                continue
            if not label.endswith(" - всего"):
                continue
            if not region_name:
                raise ValueError("Municipality row precedes region marker")
            name = label.removesuffix(" - всего").strip()
            parsed = [value_and_status(value) for value in row[1:7]]
            records.append(
                {
                    "source": "Arkhangelskstat/NAOstat",
                    "source_region_code": region_code,
                    "source_region_name": region_name,
                    "source_municipality_name": name,
                    "source_municipality_type": source_type(name),
                    "source_row": row_number,
                    "year": 2024,
                    "period": "2024-01-01/2024-12-31",
                    "period_caption": "январь-декабрь 2024 года",
                    "organization_scope": "organizations_excluding_small_businesses",
                    "headcount_people": parsed[0][0],
                    "headcount_status": parsed[0][1],
                    "prior_year_headcount_people": None,
                    "headcount_yoy_percent": None,
                    "payroll_thousand_rubles": parsed[1][0],
                    "payroll_status": parsed[1][1],
                    "monthly_wage_rubles": parsed[2][0],
                    "wage_status": parsed[2][1],
                    "prior_year_monthly_wage_rubles": None,
                    "wage_yoy_percent": None,
                    "municipal_form_headcount_people": parsed[3][0],
                    "municipal_form_headcount_status": parsed[3][1],
                    "municipal_form_payroll_thousand_rubles": parsed[4][0],
                    "municipal_form_payroll_status": parsed[4][1],
                    "municipal_form_monthly_wage_rubles": parsed[5][0],
                    "municipal_form_wage_status": parsed[5][1],
                }
            )
    finally:
        workbook.close()
    if len(records) != 27:
        raise ValueError(f"Expected 27 Arkhangelsk/NAO municipality totals, got {len(records)}")
    return records


def parse_buryatia_pair(headcount_path: Path, wage_path: Path) -> list[dict[str, object]]:
    def annual_rows(path: Path) -> dict[str, tuple[int, tuple[object, ...]]]:
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            annual_names = [
                name for name in workbook.sheetnames if name.strip() == "январь-декабрь 2024 г."
            ]
            if len(annual_names) != 1:
                raise ValueError(f"Expected one Buryatia annual sheet, got {annual_names}")
            sheet = workbook[annual_names[0]]
            result = {}
            for row_number, row in enumerate(sheet.iter_rows(values_only=True), 1):
                name = row[0]
                if row_number < 6 or not isinstance(name, str) or not name.strip():
                    continue
                key = name.strip()
                if key in result:
                    raise ValueError(f"Duplicate Buryatia source name: {key!r}")
                result[key] = (row_number, row)
            return result
        finally:
            workbook.close()

    headcount = annual_rows(headcount_path)
    wage = annual_rows(wage_path)
    if headcount.keys() != wage.keys():
        raise ValueError("Buryatia annual workbooks have different exact municipality names")
    records: list[dict[str, object]] = []
    for name, (row_number, hrow) in headcount.items():
        _, wrow = wage[name]
        hvals = [value_and_status(value) for value in hrow[1:4]]
        wvals = [value_and_status(value) for value in wrow[1:4]]
        records.append(
            {
                "source": "Buryatstat",
                "source_region_code": "3",
                "source_region_name": "Республика Бурятия",
                "source_municipality_name": name,
                "source_municipality_type": source_type(name),
                "source_row": row_number,
                "year": 2024,
                "period": "2024-01-01/2024-12-31",
                "period_caption": "январь-декабрь 2024 г.; накопленным итогом",
                "organization_scope": "organizations_excluding_small_businesses",
                "headcount_people": hvals[0][0],
                "headcount_status": hvals[0][1],
                "prior_year_headcount_people": hvals[1][0],
                "headcount_yoy_percent": hvals[2][0],
                "payroll_thousand_rubles": None,
                "payroll_status": "not_provided",
                "monthly_wage_rubles": wvals[0][0],
                "wage_status": wvals[0][1],
                "prior_year_monthly_wage_rubles": wvals[1][0],
                "wage_yoy_percent": wvals[2][0],
                "municipal_form_headcount_people": None,
                "municipal_form_headcount_status": "not_provided",
                "municipal_form_payroll_thousand_rubles": None,
                "municipal_form_payroll_status": "not_provided",
                "municipal_form_monthly_wage_rubles": None,
                "municipal_form_wage_status": "not_provided",
            }
        )
    if len(records) != 27:
        raise ValueError(f"Expected 27 Buryatia municipality rows, got {len(records)}")
    return records


def simple_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).replace("ё", "е").casefold()
    text = re.sub(r"\bг\.(?=\s|$)", " город ", text)
    text = re.sub(r"[«»\"'(),.:;–—-]", " ", text)
    return " ".join(text.split())


def name_core(value: str) -> str:
    text = simple_name(value)
    phrases = (
        "муниципальный район",
        "муниципальный округ",
        "городской округ",
        "город",
    )
    for phrase in phrases:
        text = re.sub(rf"\b{phrase}\b", " ", text)
    return " ".join(text.split())


def type_compatible(source: str, registry: str) -> bool:
    return source == registry or source.startswith("город (тип уровня")


def load_registry(panel_path: Path, bridge_path: Path) -> list[dict[str, str]]:
    with panel_path.open(encoding="utf-8", newline="") as stream:
        panel_rows = list(csv.DictReader(stream))
    with bridge_path.open(encoding="utf-8", newline="") as stream:
        bridge = {row["territory_id"]: row for row in csv.DictReader(stream)}
    unique: dict[str, dict[str, str]] = {}
    for row in panel_rows:
        if row["region_code"] not in {"3", "29", "83"}:
            continue
        territory_id = row["territory_id"]
        if territory_id in unique:
            continue
        meta = bridge.get(territory_id)
        if meta is None or not (int(meta["year_from"]) <= 2024 < int(meta["year_to"])):
            continue
        unique[territory_id] = {
            "territory_id": territory_id,
            "entity_id": row["entity_id"],
            "region_code": row["region_code"],
            "region_name": row["region_name"],
            "registry_name": row["municipal_district_name"],
            "registry_type": meta["municipal_district_type"],
            "oktmo": row["oktmo"],
            "year_from": meta["year_from"],
            "year_to": meta["year_to"],
        }
    return list(unique.values())


def candidate_crosswalk(
    records: list[dict[str, object]], registry: list[dict[str, str]]
) -> list[dict[str, object]]:
    output = []
    for source in records:
        region = str(source["source_region_code"])
        source_name = str(source["source_municipality_name"])
        source_kind = str(source["source_municipality_type"])
        pool = [row for row in registry if row["region_code"] == region]
        exact = [
            row
            for row in pool
            if simple_name(row["registry_name"]) == simple_name(source_name)
            and type_compatible(source_kind, row["registry_type"])
        ]
        normalised = [
            row
            for row in pool
            if name_core(row["registry_name"]) == name_core(source_name)
            and type_compatible(source_kind, row["registry_type"])
        ]
        matches = exact if exact else normalised
        method = "exact_region_type_name" if exact else "normalized_region_type_core"
        if len(matches) != 1:
            method = "ambiguous" if matches else "unmatched"
        match = matches[0] if len(matches) == 1 else {}
        output.append(
            {
                "source_region_code": region,
                "source_region_name": source["source_region_name"],
                "source_municipality_name": source_name,
                "source_municipality_type": source_kind,
                "registry_name": match.get("registry_name", ""),
                "registry_type": match.get("registry_type", ""),
                "territory_id": match.get("territory_id", ""),
                "entity_id": match.get("entity_id", ""),
                "oktmo": match.get("oktmo", ""),
                "year_from": match.get("year_from", ""),
                "year_to": match.get("year_to", ""),
                "candidate_method": method,
                "candidate_count": len(matches),
                "manual_review_required": True,
                "ambiguity_candidates": " | ".join(
                    f"{row['registry_name']} [{row['oktmo']}]" for row in matches
                ),
            }
        )
    return output


PROFILE_NAMES = {
    "0": "Сбалансированный: средние МО",
    "1": "Продуктовый: малые районы",
    "2": "Городской сервисный",
    "3": "Удалённый: мало маркетплейсов",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_reviewed_crosswalk(
    records: list[dict[str, object]],
    candidates: list[dict[str, str]],
    bridge_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    if len(REVIEWED_MAPPING) != 21:
        raise ValueError(f"Expected exactly 21 reviewed mappings, got {len(REVIEWED_MAPPING)}")

    source_by_key = {
        (str(row["source_region_code"]), str(row["source_municipality_name"])): row
        for row in records
    }
    if len(source_by_key) != len(records):
        raise ValueError("Source municipality keys are not unique")

    candidate_by_key = {
        (row["source_region_code"], row["source_municipality_name"]): row
        for row in candidates
    }
    if len(candidate_by_key) != len(candidates):
        raise ValueError("Candidate municipality keys are not unique")

    bridge_by_entity = {row["entity_id"]: row for row in bridge_rows}
    if len(bridge_by_entity) != len(bridge_rows):
        raise ValueError("Bridge entity IDs are not unique")

    reviewed: list[dict[str, object]] = []
    for key, expected_entity_id in REVIEWED_MAPPING.items():
        source = source_by_key.get(key)
        candidate = candidate_by_key.get(key)
        bridge = bridge_by_entity.get(expected_entity_id)
        if source is None or candidate is None or bridge is None:
            raise ValueError(f"Missing source/candidate/bridge row for reviewed key {key!r}")
        if candidate["candidate_count"] != "1":
            raise ValueError(f"Reviewed key is not a unique candidate: {key!r}")
        if candidate["entity_id"] != expected_entity_id:
            raise ValueError(
                f"Reviewed entity mismatch for {key!r}: "
                f"{candidate['entity_id']!r} != {expected_entity_id!r}"
            )
        if not (int(bridge["year_from"]) <= 2024 < int(bridge["year_to"])):
            raise ValueError(f"Reviewed registry version is not valid in 2024: {key!r}")
        for field in ("territory_id", "entity_id", "oktmo", "year_from", "year_to"):
            if candidate[field] != bridge[field]:
                raise ValueError(f"Candidate/bridge {field} mismatch for {key!r}")
        if candidate["source_region_code"] != bridge["region_code"]:
            raise ValueError(f"Candidate/bridge region mismatch for {key!r}")

        reviewed.append(
            {
                **candidate,
                "manual_review_required": False,
                "manual_review_completed": True,
                "cluster": bridge["cluster"],
                "profile_name": PROFILE_NAMES[bridge["cluster"]],
                "review_status": "manual_name_type_region_version_verified",
                "review_basis": (
                    "manual review of unique region+type+official-name candidate "
                    "and registry validity in 2023-2024"
                ),
                "official_source_oktmo_present": False,
                "join_scope": "20 Arkhangelsk Oblast + 1 Nenets AO municipalities only",
            }
        )

    if sum(row["source_region_code"] == "29" for row in reviewed) != 20:
        raise ValueError("Reviewed scope must contain exactly 20 Arkhangelsk rows")
    if sum(row["source_region_code"] == "83" for row in reviewed) != 1:
        raise ValueError("Reviewed scope must contain exactly 1 Nenets AO row")
    if any(row["source_region_code"] == "3" for row in reviewed):
        raise ValueError("Buryatia must not enter the reviewed analytical join")
    return reviewed


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_profile_results(
    records: list[dict[str, object]],
    reviewed: list[dict[str, object]],
    bridge_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_by_key = {
        (str(row["source_region_code"]), str(row["source_municipality_name"])): row
        for row in records
    }
    joined: list[dict[str, object]] = []
    for link in reviewed:
        key = (str(link["source_region_code"]), str(link["source_municipality_name"]))
        source = source_by_key[key]
        joined.append({**source, **link})

    eligible_bridge = [
        row
        for row in bridge_rows
        if row["region_code"] in {"29", "83"}
        and int(row["year_from"]) <= 2024 < int(row["year_to"])
    ]
    rows: list[dict[str, object]] = []
    for cluster in ("0", "1", "2", "3"):
        group = [row for row in joined if row["cluster"] == cluster]
        wage = [
            float(row["monthly_wage_rubles"])
            for row in group
            if row["wage_status"] == "observed"
        ]
        headcount = [
            float(row["headcount_people"])
            for row in group
            if row["headcount_status"] == "observed"
        ]
        panel_count = sum(row["cluster"] == cluster for row in eligible_bridge)
        wage_summary = numeric_summary(wage)
        headcount_summary = numeric_summary(headcount)
        rows.append(
            {
                "cluster": int(cluster),
                "profile_name": PROFILE_NAMES[cluster],
                "reviewed_municipalities": len(group),
                "eligible_panel_municipalities_in_two_regions": panel_count,
                "panel_coverage_fraction": len(group) / panel_count if panel_count else None,
                "excluded_panel_municipalities": panel_count - len(group),
                "wage_count": wage_summary["count"],
                "wage_min_rubles": wage_summary["min"],
                "wage_median_rubles": wage_summary["median"],
                "wage_max_rubles": wage_summary["max"],
                "headcount_count": headcount_summary["count"],
                "headcount_min_people": headcount_summary["min"],
                "headcount_median_people": headcount_summary["median"],
                "headcount_max_people": headcount_summary["max"],
            }
        )

    excluded_source = [
        row
        for row in records
        if row["source_region_code"] in {"29", "83"}
        and (
            str(row["source_region_code"]),
            str(row["source_municipality_name"]),
        )
        not in REVIEWED_MAPPING
    ]
    result = {
        "scope": {
            "description": (
                "Descriptive comparison of 21 manually reviewed municipalities: "
                "20 in Arkhangelsk Oblast and 1 in Nenets AO"
            ),
            "year": 2024,
            "reviewed_municipalities": len(reviewed),
            "source_municipalities_in_two_region_workbook": 27,
            "source_coverage_fraction": len(reviewed) / 27,
            "eligible_panel_municipalities_in_two_regions": len(eligible_bridge),
            "panel_coverage_fraction": len(reviewed) / len(eligible_bridge),
            "excluded_source_municipalities": [
                {
                    "region": row["source_region_name"],
                    "name": row["source_municipality_name"],
                    "reason": "no manually approved unique panel candidate",
                }
                for row in excluded_source
            ],
            "excluded_panel_count": len(eligible_bridge) - len(reviewed),
            "buryatia": {
                "source_rows": sum(row["source_region_code"] == "3" for row in records),
                "bridge_rows": sum(row["region_code"] == "3" for row in bridge_rows),
                "included": False,
                "reason": "no Republic of Buryatia rows in the frozen panel bridge",
            },
        },
        "profiles": rows,
        "interpretation_limits": [
            "No p-values or inferential tests are reported.",
            "The 21 municipalities are a small non-national reviewed subset.",
            "Headcount is an absolute organization payroll scale, not an employment rate.",
            "Wage and headcount exclude small businesses and do not represent all residents.",
            "The join is manually name/type/region/version verified, not an official source OKTMO join.",
        ],
    }
    return rows, result


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=root / "artifacts/sources/round2")
    parser.add_argument("--panel", type=Path, default=root / "data/processed/panel.csv")
    parser.add_argument(
        "--bridge", type=Path, default=root / "reports/external-v4/municipality_bridge.csv"
    )
    parser.add_argument("--output", type=Path, default=root / "reports/external-round2")
    args = parser.parse_args()

    ark = args.source_dir / "arkhangelsk-nao-2024.xlsx"
    bur_head = args.source_dir / "buryatia-headcount-2024.xlsx"
    bur_wage = args.source_dir / "buryatia-wage-2024.xlsx"
    records = parse_arkhangelsk(ark) + parse_buryatia_pair(bur_head, bur_wage)
    args.output.mkdir(parents=True, exist_ok=True)
    clean_path = args.output / "rosstat_labor_2024.csv"
    candidate_path = args.output / "candidate_crosswalk.csv"
    write_csv(clean_path, records)

    # Preserve the primary candidate audit when it already exists. Rebuilding it
    # requires scanning the large panel; reviewed outputs only need this saved
    # candidate file and the compact frozen bridge.
    if candidate_path.exists():
        candidates = read_csv(candidate_path)
        if len(candidates) != len(records):
            raise ValueError(
                f"Saved candidate audit has {len(candidates)} rows; expected {len(records)}"
            )
    else:
        registry = load_registry(args.panel, args.bridge)
        candidates = candidate_crosswalk(records, registry)
        write_csv(candidate_path, candidates)
    candidate_sha256 = sha256_file(candidate_path)

    bridge_rows = read_csv(args.bridge)
    reviewed = build_reviewed_crosswalk(records, candidates, bridge_rows)
    reviewed_path = args.output / "reviewed_crosswalk.csv"
    write_csv(reviewed_path, reviewed)
    profile_rows, results = build_profile_results(records, reviewed, bridge_rows)
    write_csv(args.output / "results.csv", profile_rows)
    (args.output / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    metadata = {
        "scope": {
            "year": 2024,
            "raw_regions": [
                "Архангельская область",
                "Ненецкий автономный округ",
                "Республика Бурятия",
            ],
            "clean_rows": len(records),
            "reviewed_analytical_scope": {
                "regions": ["Архангельская область", "Ненецкий автономный округ"],
                "municipalities": 21,
                "arkhangelsk": 20,
                "nenets": 1,
                "national_generalisation": False,
            },
            "analytical_join_approved_for_reviewed_subset_only": True,
        },
        "sources": [
            {
                "url": ARK_URL,
                "page_url": "https://29.rosstat.gov.ru/main_indicators?print=1",
                "file": ark.name,
                "sha256": sha256_file(ark),
                "rows": 27,
                "title_period": "январь-декабрь 2024 года",
            },
            {
                "url": BUR_HEADCOUNT_URL,
                "page_url": "https://03.rosstat.gov.ru/resourse?print=1",
                "file": bur_head.name,
                "sha256": sha256_file(bur_head),
                "rows": 27,
                "title_period": "за январь-декабрь 2024 г.",
            },
            {
                "url": BUR_WAGE_URL,
                "page_url": "https://03.rosstat.gov.ru/trud_zp?print=1",
                "file": bur_wage.name,
                "sha256": sha256_file(bur_wage),
                "rows": 27,
                "title_period": "за январь-декабрь 2024 г.",
            },
        ],
        "schema": {
            "headcount": "average payroll headcount excluding external part-timers; people",
            "payroll": "accrued payroll; thousand rubles; Arkhangelsk/NAO only",
            "wage": "average monthly nominal accrued wage; rubles",
            "organization_scope": "organizations excluding small businesses",
            "period": "January-December cumulative/annual 2024",
        },
        "candidate_mapping": {
            "file": candidate_path.name,
            "sha256": candidate_sha256,
            "status": "primary_unreviewed_candidate_audit_preserved",
            "source_has_oktmo": False,
            "candidate_rows": len(candidates),
            "unique_candidates": sum(row["candidate_count"] == "1" for row in candidates),
            "ambiguous_candidates": sum(int(row["candidate_count"]) > 1 for row in candidates),
            "unmatched": sum(row["candidate_count"] == "0" for row in candidates),
            "methods": (
                "region + type + exact name, then deterministic type-token "
                "normalisation; no fuzzy matching"
            ),
        },
        "reviewed_mapping": {
            "file": reviewed_path.name,
            "sha256": sha256_file(reviewed_path),
            "rows": len(reviewed),
            "review_tag": "manual_name_type_region_version_verified",
            "official_source_oktmo_join": False,
            "approved_scope": "20 Arkhangelsk Oblast + 1 Nenets AO municipalities",
            "all_expected_entity_ids_verified": True,
            "buryatia_bridge_rows": sum(row["region_code"] == "3" for row in bridge_rows),
        },
        "suppression": {
            "source_token": "…2)",
            "meaning": (
                "not published to protect confidential primary statistical data "
                "under Federal Law 282-FZ"
            ),
            "handling": "null with status=suppressed; never converted to zero",
        },
        "cautions": [
            "The workbook name/sheet name containing 'oktmo' is not evidence of row-level OKTMO codes.",
            "The reviewed join was manually name/type/region/version verified; it is not an official source OKTMO join.",
            "The 21-municipality subset cannot support a national validation claim.",
            "Headcount is an absolute organization payroll scale, not an employment rate.",
            "No dataset-specific licence statement was found on the source pages or in the workbooks; cite the official source URL.",
        ],
    }
    (args.output / "source_mapping_audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
