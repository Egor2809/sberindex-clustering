"""Deterministic stratified spot checks against official PMO passport HTML.

These contemporary primary-source pages are checked against the independently
downloaded 2025 archive. Publication revisions are reported as discrepancies;
the historical archive is never overwritten to make a check agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.download_external_national import ROOT, cohort, sha256
from sbercluster.external_national import name_key, parse_passport, select_unique, total_sector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "artifacts/sources/external-national-2026-10-03")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/external-national-2026-10-03")
    parser.add_argument("--per-region", type=int, default=3)
    args = parser.parse_args()
    if args.per_region < 1 or args.per_region > 10:
        raise ValueError("Per-region check size must be between1 and10")
    registry = cohort(2023)
    registry["file"] = registry.source_oktmo8.map(lambda c: args.raw / f"{c}-2023.html")
    sample = registry.loc[registry.file.map(Path.exists)].groupby("region_code", sort=True).head(args.per_region)
    historical = pd.read_csv(args.report / "cohort-2023.csv").set_index("entity_id")
    rows, files = [], []
    for row in sample.itertuples():
        passport = parse_passport(row.file.read_bytes(), 2023)
        name_verified = bool(passport.source_name and name_key(passport.source_name) in
                             {name_key(row.municipal_district_name), name_key(row.municipal_district_name_short)})
        files.append({"entity_id": row.entity_id, "file": row.file.name, "sha256": sha256(row.file),
                      "source_status": passport.status, "source_name": passport.source_name,
                      "name_verified": name_verified,
                      "url": f"https://rosstat.gov.ru/scripts/db_inet2/passport/table.aspx?opt={int(row.source_oktmo8)}2023",
                      "tls_certificate_verified": False})
        for kind in ["wage", "headcount", "population"]:
            primary, status = select_unique(passport.observations, kind, total_sector)
            population_source = "population_estimate"
            if kind == "population" and status == "not_published":
                primary, status = select_unique(passport.observations, "population_age_sex",
                                                lambda r: r["sector_path"] == ["Всего", "Всего"])
                population_source = "population_all_ages_both_sexes"
            archived = historical.loc[row.entity_id, f"{kind}_total"]
            comparable = name_verified and status == "observed" and pd.notna(archived)
            tolerance = 1.0 if kind == "population" else .11
            difference = primary - archived if comparable else None
            rows.append({"entity_id": row.entity_id, "region_code": row.region_code, "kind": kind,
                         "primary_status": status, "primary_value": primary, "archive_value": archived,
                         "source_name_verified": name_verified, "comparable": comparable,
                         "difference_primary_minus_archive": difference,
                         "agrees_at_publication_precision": abs(difference) <= tolerance if comparable else None,
                         "population_primary_kind": population_source if kind == "population" else None})
    checks = pd.DataFrame(rows)
    checks.to_csv(args.report / "primary-source-checks.csv", index=False)
    summary = {"sample_rule": f"First{args.per_region} frozenIDs orderedbyterritoryID perregionamongalreadydownloaded2023PMOpages; no outcome/label selection",
               "sampled_municipalities": len(sample), "sampled_regions": sample.region_code.nunique(),
               "source_status_counts": pd.Series([r["source_status"] for r in files]).value_counts().to_dict(),
               "source_files": files, "by_kind": {},
               "transport_limit": "Official Rosstat HTTPS certificate chain could notbe verified locally; explicit public-data exception. IfTochno archive TLS verified. Noauth or globalTLS change.",
               "revision_policy": "Contemporary2026PMOpages may revisehistorical2023values; discrepancies reported, archive untouched."}
    summary["population_comparison_limit"] = "Age/sex all-resident counts provide a related indicator check when the exact population-estimate indicator is absent. Three Ryazan discrepancies are cross-indicator comparisons, not verified same-indicator errors."
    for kind in ["wage", "headcount", "population"]:
        comparable = checks.loc[checks.kind.eq(kind) & checks.comparable]
        agreement = comparable.agrees_at_publication_precision.astype(bool)
        summary["by_kind"][kind] = {"comparable": len(comparable), "agrees": int(agreement.sum()),
                                   "discrepancies": comparable.loc[~agreement].to_dict("records")}
    (args.report / "primary-source-checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_files", "transport_limit", "revision_policy"}}, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
