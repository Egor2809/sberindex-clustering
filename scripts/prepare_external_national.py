"""Build exact-code year-valid independent Rosstat labour cohorts.

Raw archives are immutable IfTochno transformations of Rosstat PMO data. The
archive's current/stable code is deliberately not used for a longitudinal join.
Conflicting keys, ambiguous dates, names, units or municipal levels are excluded
and reported. Source database IDs are not Russian subject/region codes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.download_external_national import ROOT, cohort, sha256
from sbercluster.external_national import broad_sector, name_key, number_status

INDICATORS = {"wage": "Y48423007", "headcount": "Y48423005", "population": "Y48112027"}
PARENT_DATABASES = {83: "Архангельская область", 86: "Тюменская область",
                    89: "Тюменская область", 87: "Магаданская область"}


def read_source(raw: Path, kind: str, year: int) -> tuple[pd.DataFrame, dict]:
    path = raw / f"{kind}-iftochno.zip"
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if f"year{year}_" in name and name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one annual source member: {kind}/{year}")
        content = archive.read(names[0])
    import io
    frame = pd.read_csv(io.BytesIO(content), sep=";", dtype=str, keep_default_na=False)
    if not frame.indicator_code.eq(INDICATORS[kind]).all() or not frame.year.eq(str(year)).all():
        raise ValueError("Unexpected indicator or observation year")
    period = "На 1 января" if kind == "population" else "Январь-декабрь"
    frame = frame.loc[frame.indicator_period.eq(period)].copy()
    if kind == "population":
        frame = frame.loc[frame.mest.eq("Все население")].copy()
        frame["sector_code"] = "total"
    else:
        def section(text):
            if text == "Всего по обследуемым видам экономической деятельности":
                return "total"
            return broad_sector({"sector": text, "sector_path": [text]})
        frame["sector_code"] = frame.okved2.map(section)
        frame = frame.loc[frame.sector_code.notna()].copy()
    frame["source_row"] = frame.index + 2
    if frame.duplicated(["oktmo", "sector_code"]).any():
        conflicts = frame.loc[frame.duplicated(["oktmo", "sector_code"], keep=False), ["oktmo", "sector_code"]]
        frame["conflicting_key"] = frame.set_index(["oktmo", "sector_code"]).index.isin(conflicts.set_index(["oktmo", "sector_code"]).index)
    else:
        frame["conflicting_key"] = False
    frame["kind"] = kind
    return frame, {"kind": kind, "year": year, "archive_file": path.name,
                   "archive_sha256": sha256(path), "member": names[0],
                   "member_sha256": hashlib.sha256(content).hexdigest(),
                   "annual_broad_sector_rows": len(frame)}


def attach_observations(registry: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    merged = source.merge(registry, left_on="oktmo", right_on="source_oktmo8",
                          how="inner", suffixes=("_source", ""), validate="many_to_one")
    merged["name_verified"] = [name_key(a) in {name_key(b), name_key(c)}
                               for a, b, c in zip(merged.municipality, merged.municipal_district_name,
                                                 merged.municipal_district_name_short)]
    merged["region_verified"] = [name_key(a) in {name_key(b), name_key(PARENT_DATABASES.get(int(code), b))}
                                 for a, b, code in zip(merged.region_name_source, merged.region_name, merged.region_code)]
    start = pd.to_numeric(merged.oktmo_year_from, errors="coerce")
    end = pd.to_numeric(merged.oktmo_year_to, errors="coerce")
    merged["source_code_validity_verified"] = (start <= merged.year.astype(int)) & (merged.year.astype(int) <= end)
    merged["municipal_level_verified"] = merged.mun_level.eq("Муниципальное образование верхнего уровня")
    expected_units = merged.kind.map({"wage": "Рубль", "headcount": "Человек", "population": "Человек"})
    merged["unit_verified"] = merged.indicator_unit.eq(expected_units)
    values = [number_status(value) for value in merged.indicator_value]
    merged["value"] = [pair[0] for pair in values]
    merged["value_status"] = [pair[1] for pair in values]
    merged["join_verified"] = (merged.name_verified & merged.region_verified & merged.source_code_validity_verified
                                & merged.municipal_level_verified & merged.unit_verified & ~merged.conflicting_key)
    merged["boundary_history_flag"] = merged.oktmo_history.str.contains("территориаль|Выделение|Объединение", case=False, regex=True)
    return merged


def build_year(raw: Path, year: int, output: Path) -> dict:
    registry = cohort(year)
    observations = []
    sources = []
    for kind in INDICATORS:
        source, metadata = read_source(raw, kind, year)
        sources.append(metadata)
        observations.append(attach_observations(registry, source))
    all_rows = pd.concat(observations, ignore_index=True)
    # A source with duplicate observation keys cannot enter any numerator/total.
    valid = all_rows.loc[all_rows.join_verified & all_rows.value_status.eq("observed")].copy()
    wide = valid.pivot(index="entity_id", columns=["kind", "sector_code"], values="value")
    wide.columns = [f"{kind}_{sector}" for kind, sector in wide.columns]
    result = registry.merge(wide, on="entity_id", how="left", validate="one_to_one")
    history = all_rows.groupby("entity_id").boundary_history_flag.any()
    result["source_boundary_history_flag"] = result.entity_id.map(history).eq(True)
    earlier = cohort(2023).set_index("entity_id")
    result["registry_changed_2023_to_year"] = [earlier.loc[row.entity_id, "oktmo"] != row.oktmo or
                                              earlier.loc[row.entity_id, "year_from"] != row.year_from
                                              for row in result.itertuples()]
    result["intracity_moscow_petersburg"] = result.region_code.isin([77, 78]) & result.municipal_district_type.str.contains("внутригород", case=False)
    for col in ["wage_total", "headcount_total", "population_total", "headcount_A", "headcount_C", "headcount_O", "headcount_P", "headcount_Q"]:
        if col not in result:
            result[col] = np.nan
    result["log_annual_monthly_wage_rubles"] = np.log(result.wage_total.where(result.wage_total > 0))
    jobs_ratio = result.headcount_total / result.population_total.where(result.population_total > 0)
    result["covered_organization_jobs_per_resident"] = jobs_ratio.where(result.headcount_total >= 0)
    result["log_covered_organization_jobs_per_resident"] = np.log(jobs_ratio.where(jobs_ratio > 0))
    total = result.headcount_total.where(result.headcount_total > 0)
    for name, columns in {"manufacturing_headcount_share": ["headcount_C"],
                          "agriculture_headcount_share": ["headcount_A"],
                          "public_administration_education_health_headcount_share": ["headcount_O", "headcount_P", "headcount_Q"]}.items():
        numerator = result[columns].sum(axis=1, min_count=len(columns))
        share = numerator / total
        result[name] = share.where((numerator >= 0) & (share <= 1))
    # Every municipality, including excluded observations, remains in the audit.
    selected = ["entity_id", "oktmo_source", "year", "kind", "sector_code", "municipality", "region_name_source",
                "indicator_period", "indicator_unit", "indicator_value", "value", "value_status", "source_row",
                "oktmo_stable", "oktmo_year_from", "oktmo_year_to", "oktmo_history", "name_verified", "region_verified",
                "source_code_validity_verified", "municipal_level_verified", "unit_verified", "conflicting_key", "join_verified", "boundary_history_flag"]
    all_rows[selected].to_csv(output / f"join-audit-{year}.csv", index=False)
    columns = [c for c in result.columns if c not in {"shape", "shape_linked_oktmo"}]
    result[columns].to_csv(output / f"cohort-{year}.csv", index=False)
    reasons = {field: int((~all_rows[field]).sum()) for field in ["name_verified", "region_verified", "source_code_validity_verified", "municipal_level_verified", "unit_verified"]}
    outcomes = ["log_annual_monthly_wage_rubles", "log_covered_organization_jobs_per_resident", "manufacturing_headcount_share", "agriculture_headcount_share", "public_administration_education_health_headcount_share"]
    coverage = {field: {"n": int(result[field].notna().sum()), "regions": int(result.loc[result[field].notna(), "region_code"].nunique())} for field in outcomes}
    return {"year": year, "frozen_cohort": len(registry), "sources": sources, "exact_code_source_rows": len(all_rows),
            "verified_source_rows": int(all_rows.join_verified.sum()), "exclusion_failures_may_overlap": reasons,
            "conflicting_source_rows": int(all_rows.conflicting_key.sum()),
            "source_numeric_status": all_rows.value_status.value_counts().to_dict(), "outcome_coverage": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/sources/external-national-2026-10-03")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/external-national-2026-10-03")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"source_catalog": "https://tochno.st/datasets/bdmo", "source_license": "CC BY 4.0",
                "archive_sources": json.loads((args.raw / "archive-sources.json").read_text(encoding="utf-8")),
                "registry_sha256": sha256(ROOT / "artifacts/sources/acquisition/t_dict_municipal_districts.xlsx"),
                "frozen_reference_labels_sha256": sha256(ROOT / "reports/experiments/2026-09-23-v2/validation/reference_assignments.csv"),
                "attribution": "Муниципальная статистика России с 2005 года // Росстат; обработка Если быть точным, 2025.",
                "join": "Exact eight-digit observed OKTMO and year-valid SberIndex territory registry. No stable-code substitution or fuzzy name join.",
                "source_code_validity": "Inclusive source from/to years; inconsistent or missing source validity excluded conservatively; registry uses half-open year_from/year_to.",
                "years": [build_year(args.raw, year, args.output) for year in [2023, 2024]]}
    (args.output / "source-audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"years": [{"year": y["year"], "coverage": y["outcome_coverage"], "failures": y["exclusion_failures_may_overlap"]} for y in metadata["years"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
