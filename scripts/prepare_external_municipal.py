"""Join open municipal statistics to the 2 016 frozen municipalities by OKTMO.

Every source row is matched only by the exact eight-digit OKTMO of the
municipality.  A code with several differing source values is marked
ambiguous and left empty; no name matching or value substitution is done.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook


TOP_LEVEL = "Муниципальное образование верхнего уровня"
LABOUR_TOTAL = "Всего по обследуемым видам экономической деятельности"
ANOMALY = "Аномальное значение показателя"
AGE_GROUPS = {
    "Всего": "population_2023_bdpmo",
    "Моложе трудоспособного возраста": "below_working_age_2023",
    "Трудоспособный возраст": "working_age_2023",
    "Старше трудоспособного возраста": "above_working_age_2023",
}
SECTION_RE = re.compile(r"^Раздел\s+(\S)\s")
LOOKALIKE = str.maketrans("АВЕНСО", "ABEHCO")
SECTION_SHARES = {
    "share_agriculture": ("A",),
    "share_mining": ("B",),
    "share_manufacturing": ("C",),
    "share_trade": ("G",),
    "share_hospitality": ("I",),
    "share_public_services": ("O", "P", "Q"),
}
OUTPUT_FIELDS = [
    "entity_id",
    "territory_id",
    "cluster",
    "region_code",
    "region_name",
    "municipal_district_type",
    "intracity_moscow_petersburg",
    "oktmo",
    "oktmo8",
    "rosstat_status",
    "rosstat_name",
    "population_2024_rosstat",
    "urban_population_2024_rosstat",
    "urban_share_2024",
    "bdpmo_population_status",
    "bdpmo_population_name",
    "population_2023_bdpmo",
    "above_working_age_share_2023",
    "population",
    "population_source",
    "wage_status",
    "wage_2023",
    "headcount_status",
    "headcount_2023",
    "jobs_per_1000",
    *SECTION_SHARES,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def oktmo8(value: str) -> str:
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 8:
        return digits
    if len(digits) == 11 and digits.endswith("000"):
        return digits[:8]
    raise ValueError(f"Unexpected municipal OKTMO: {value!r}")


def section_letter(label: str) -> str | None:
    match = SECTION_RE.match(label)
    return match.group(1).translate(LOOKALIKE) if match else None


def unique_value(values: list[tuple[float, str]]) -> tuple[float | None, str, str]:
    if not values:
        return None, "no_code_match", ""
    distinct = {value for value, _ in values}
    if len(distinct) > 1:
        return None, "ambiguous", ""
    return values[0][0], "matched", values[0][1]


def read_rosstat_population(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    rows: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in workbook["Численность_по_МО"].iter_rows(min_row=8, values_only=True):
        code = str(row[0]).strip() if row[0] is not None else ""
        if not re.fullmatch(r"\d{10}", code):
            continue
        if not code[5:8] == "000" or not isinstance(row[2], (int, float)):
            continue
        rows[code[:8]].append((float(row[2]), float(row[3] or 0), str(row[1]).strip()))
    workbook.close()
    return rows


def read_bdpmo(path: Path, key_column: str) -> dict[tuple[str, str], list[tuple[float, str, str]]]:
    rows: dict[tuple[str, str], list[tuple[float, str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter=";"):
            if row["mun_level"] != TOP_LEVEL:
                continue
            rows[(row["oktmo"], row[key_column])].append(
                (
                    float(row["indicator_value"]) if row["indicator_value"].strip() else None,
                    row["municipality"],
                    row["comment"],
                )
            )
    return rows


def labour_value(rows: dict, code: str, label: str) -> tuple[float | None, str]:
    found = rows.get((code, label), [])
    clean = [(value, name) for value, name, comment in found if comment != ANOMALY and value is not None]
    if found and not clean:
        return None, "flagged_anomalous" if any(c == ANOMALY for _, _, c in found) else "empty_in_source"
    value, status, _ = unique_value(clean)
    return value, status


def build_rows(bridge: list[dict[str, str]], rosstat: dict, population: dict, headcount: dict, wage: dict) -> list[dict[str, object]]:
    sections_by_code: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for code, label in headcount:
        letter = section_letter(label)
        if letter:
            sections_by_code[code][letter].add(label)
    result = []
    for item in bridge:
        code = oktmo8(item["oktmo"])
        row: dict[str, object] = {key: item[key] for key in (
            "entity_id", "territory_id", "cluster", "region_code", "region_name",
            "municipal_district_type", "intracity_moscow_petersburg", "oktmo",
        )}
        row["oktmo8"] = code
        candidates = rosstat.get(code, [])
        total, status, name = unique_value([((v, u), n) for v, u, n in candidates])
        row["rosstat_status"], row["rosstat_name"] = status, name
        if total is not None:
            row["population_2024_rosstat"] = int(total[0])
            row["urban_population_2024_rosstat"] = int(total[1])
            row["urban_share_2024"] = total[1] / total[0] if total[0] > 0 else None
        ages = {}
        statuses = []
        for group, column in AGE_GROUPS.items():
            value, age_status, age_name = unique_value(
                [(v, n) for v, n, _ in population.get((code, group), []) if v is not None]
            )
            ages[column] = value
            statuses.append(age_status)
            if group == "Всего":
                row["bdpmo_population_name"] = age_name
        row["bdpmo_population_status"] = statuses[0]
        row["population_2023_bdpmo"] = ages["population_2023_bdpmo"]
        if ages["population_2023_bdpmo"] and ages["above_working_age_2023"] is not None:
            row["above_working_age_share_2023"] = ages["above_working_age_2023"] / ages["population_2023_bdpmo"]
        if row.get("population_2024_rosstat"):
            row["population"], row["population_source"] = row["population_2024_rosstat"], "rosstat_2024"
        elif row["population_2023_bdpmo"]:
            row["population"], row["population_source"] = row["population_2023_bdpmo"], "bdpmo_2023"
        else:
            row["population_source"] = "none"
        row["wage_2023"], row["wage_status"] = labour_value(wage, code, LABOUR_TOTAL)
        workers, row["headcount_status"] = labour_value(headcount, code, LABOUR_TOTAL)
        row["headcount_2023"] = workers
        denominator = row["population_2023_bdpmo"] or row.get("population_2024_rosstat")
        if workers is not None and denominator:
            row["jobs_per_1000"] = 1000 * workers / float(denominator)
        if workers:
            for share, letters in SECTION_SHARES.items():
                parts = []
                for letter in letters:
                    labels = sections_by_code[code].get(letter, set())
                    value = labour_value(headcount, code, next(iter(labels)))[0] if len(labels) == 1 else None
                    parts.append(value)
                if all(part is not None for part in parts):
                    row[share] = sum(parts) / workers
        result.append(row)
    return result


def formatted(value: object) -> object:
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 6)
    return value


def coverage(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = ["population", "population_2024_rosstat", "population_2023_bdpmo", "urban_share_2024",
              "above_working_age_share_2023", "wage_2023", "headcount_2023", "jobs_per_1000", *SECTION_SHARES]
    clusters = sorted({str(row["cluster"]) for row in rows})
    return {
        field: {
            "all": sum(row.get(field) is not None for row in rows),
            "by_cluster": {
                cluster: sum(row.get(field) is not None for row in rows if str(row["cluster"]) == cluster)
                for cluster in clusters
            },
        }
        for field in fields
    }


def status_counts(rows: list[dict[str, object]], column: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(column, ""))] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=root / "artifacts/sources/municipal")
    parser.add_argument("--bridge", type=Path, default=root / "reports/external-v4/municipality_bridge.csv")
    parser.add_argument("--output", type=Path, default=root / "reports/external-municipal")
    args = parser.parse_args()

    with args.bridge.open(encoding="utf-8", newline="") as stream:
        bridge = list(csv.DictReader(stream))
    if len({row["entity_id"] for row in bridge}) != len(bridge):
        raise ValueError("Duplicate entity_id in bridge")
    codes = [oktmo8(row["oktmo"]) for row in bridge]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate OKTMO in bridge")

    sources = {
        "rosstat": args.source_dir / "rosstat-bul-mo-2024.xlsx",
        "population": args.source_dir / "bdpmo-population-2023.csv",
        "headcount": args.source_dir / "bdpmo-headcount-2023.csv",
        "wage": args.source_dir / "bdpmo-wage-2023.csv",
    }
    rows = build_rows(
        bridge,
        read_rosstat_population(sources["rosstat"]),
        read_bdpmo(sources["population"], "vozr"),
        read_bdpmo(sources["headcount"], "okved2"),
        read_bdpmo(sources["wage"], "okved2"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "municipal_indicators.csv"
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: formatted(row.get(key)) for key in OUTPUT_FIELDS})
    audit = {
        "n": len(rows),
        "matching": "exact eight-digit OKTMO (panel code without dashes and the trailing 000); ambiguous codes are left empty",
        "sources": {name: {"file": path.name, "sha256": sha256_file(path)} for name, path in sources.items()},
        "status": {
            column: status_counts(rows, column)
            for column in ("rosstat_status", "bdpmo_population_status", "wage_status", "headcount_status", "population_source")
        },
        "coverage": coverage(rows),
        "output_sha256": sha256_file(output_path),
    }
    (args.output / "mapping_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["status"], ensure_ascii=False))


if __name__ == "__main__":
    main()
