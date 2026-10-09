"""Prepare the official ID-coded snapshot; no model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
from .io import CATEGORIES, TOTAL, read_export, sha256, write_json

def read_parquet(path):
    return pd.DataFrame(pl.read_parquet(path).to_dict(as_series=False))

def active_registry(observations, registry):
    """Documented [year_from, year_to) intervals, exactly one match."""
    d = observations.copy()
    d["year"] = d["period"].str[:4].astype(int)
    merged = d.merge(registry, on="territory_id", how="left")
    selected = merged[(merged.year >= merged.year_from) & (merged.year < merged.year_to)].copy()
    keys = ["territory_id", "period"] + (["category"] if "category" in d else [])
    if len(selected) != len(d) or selected.duplicated(keys).any():
        raise ValueError("Missing or overlapping annual registry intervals")
    return selected.drop(columns="year")

def prepare(root):
    root = Path(root)
    source = root / "artifacts/sources/acquisition"
    pq = source / "hackathonlicence/consumption.parquet"
    registry_path = source / "t_dict_municipal_districts.xlsx"
    d = read_parquet(pq).drop(columns=["__index_level_0__"], errors="ignore")
    if set(d.columns) != {"date", "territory_id", "category", "value"}:
        raise ValueError("Official parquet schema changed")
    d["period"] = pd.to_datetime(d.date + "-01", format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    if d.duplicated(["territory_id", "period", "category"]).any():
        raise ValueError("Duplicate canonical key")
    if d.isna().any().any() or not np.isfinite(d.value).all() or (d.value <= 0).any():
        raise ValueError("Missing/nonpositive canonical data: review required")
    if set(d.category) != set(CATEGORIES + [TOTAL]):
        raise ValueError("Category schema changed")
    registry = pd.read_excel(registry_path, dtype={"oktmo": str})
    mapped = active_registry(d, registry)
    csvs = list((root / "data/raw").glob("*.csv"))
    if len(csvs) > 1:
        raise ValueError("Multiple CSV references: specify one vintage")
    out, reports = root / "data/processed", root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    csv_hash = None
    if csvs:
        csv = read_export(csvs[0]).rename(columns={"category_15": "category"})
        key = ["period", "category", "value"]
        mismatch = d.groupby(key).size().subtract(csv.groupby(key).size(), fill_value=0)
        if (mismatch != 0).any():
            raise ValueError("Parquet and supplied CSV differ; review vintage")
        csv_hash = sha256(csvs[0])
        write_json(reports / "source_alignment.json", {
            "csv_sha256": csv_hash, "parquet_sha256": sha256(pq),
            "compared": ["period", "category", "value", "multiplicity"],
            "differing_multiset_keys": 0, "csv_rows": len(csv), "parquet_rows": len(d),
            "canonical_key_duplicates": 0,
            "scope": "Full numeric payload equivalence; use source territory_id directly, no name inference"})
    else:
        write_json(reports / "source_alignment.json", {"status": "reference_CSV_not_supplied",
            "parquet_sha256": sha256(pq), "parquet_rows": len(d), "canonical_key_duplicates": 0})
    wide = d.pivot(index=["territory_id", "period"], columns="category", values="value")
    months = sorted(d.period.unique())
    calendar = pd.date_range(months[0], months[-1], freq="MS").strftime("%Y-%m-%d").tolist()
    if months != calendar:
        raise ValueError("Whole calendar months missing")
    complete = wide.notna().all(axis=1).groupby(level=0).sum()
    balanced_ids = complete[complete.eq(len(months))].index
    panel = wide.loc[balanced_ids].reset_index()
    meta = active_registry(panel[["territory_id", "period"]], registry)
    cols = ["territory_id", "period", "municipal_district_name", "region_name", "region_code", "oktmo",
            "municipal_district_center_lat", "municipal_district_center_lon"]
    panel = panel.merge(meta[cols], on=["territory_id", "period"], validate="one_to_one")
    latest = meta.sort_values("period").groupby("territory_id").tail(1).set_index("territory_id")
    panel["mo"] = panel.territory_id.map(latest.municipal_district_name)
    panel["entity_id"] = "tid_" + panel.territory_id.astype(str)
    panel["boundary_review_flag"] = np.where(panel.region_code.eq(20),
        "chechnya_registry_caveat", "annual_registry_2024_coverage_caveat")
    panel = panel.sort_values(["entity_id", "period"])
    panel.to_csv(out / "panel.csv", index=False, encoding="utf-8")
    mapped.to_csv(out / "canonical_long.csv", index=False, encoding="utf-8")
    excluded = pd.DataFrame({"territory_id": sorted(set(d.territory_id)-set(balanced_ids))})
    excluded["observed_complete_months"] = excluded.territory_id.map(complete)
    excluded["reason"] = "not_complete_all_months_retrospective_panel"
    excluded.to_csv(reports / "excluded_territories.csv", index=False)
    market_path = source / "hackathonlicence/market_access.parquet"
    market = read_parquet(market_path)
    if market.territory_id.duplicated().any():
        raise ValueError("Duplicate market-access territory")
    market.to_csv(out / "market_access_2024.csv", index=False)
    ratio = wide[CATEGORIES].sum(axis=1) / wide[TOTAL]
    audit_path = reports / "data_audit.json"
    prior = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    audit = {**prior, "source": "official consumption.parquet; see source_alignment.json for optional CSV comparison",
        "sha256": sha256(pq), "rows": len(d), "n_territories": int(d.territory_id.nunique()),
        "regions": int(mapped.region_code.nunique()), "n_months": len(months), "months": months,
        "n_categories_including_total": len(CATEGORIES)+1, "missing_cells": 0,
        "canonical_duplicate_keys": 0, "complete_panel_names": len(balanced_ids),
        "complete_panel_territories": len(balanced_ids), "panel_rows": len(panel),
        "incomplete_territories": len(excluded),
        "identity_status": "official territory_id; annual registry joined uniquely",
        "semantics_status": "mean noncash resident expenditure estimates; ratios are NOT proven exhaustive shares",
        "registry_interval": "year_from inclusive, year_to exclusive; coverage documented to 2024-01-01",
        "chechnya_panel_territories": int(panel.loc[panel.region_code.eq(20), "territory_id"].nunique()),
        "market_access_panel_coverage": int(len(set(balanced_ids) & set(market.territory_id))),
        "observed_category_sum_over_total_quantiles": {str(k):float(v) for k,v in ratio.quantile([0,.01,.5,.99,1]).items()},
        "observed_category_sum_above_total": int((ratio > 1).sum()),
        "model_training_executed": False,
        "csv_name_conflicts_resolved_by": "original ID-coded parquet, not guessed name matching",
        "panel_selection": "balanced over full 2023-2024: retrospective atlas, not an as-of forecast cohort"}
    csv_only = ["n_unique_names", "ambiguous_names", "rows_with_duplicate_key", "rows_excluded_due_to_ambiguous_name", "unique_unambiguous_names", "unambiguous_incomplete_names"]
    audit["csv_export_diagnostics"] = {key:audit.pop(key) for key in csv_only if key in audit}
    audit["csv_export_diagnostics"]["meaning"] = "name-only export diagnostics; these exclusions do NOT apply to the canonical ID panel"
    write_json(audit_path, audit)
    write_json(out / "manifest.json", {"raw_sha256": sha256(pq), "csv_sha256": csv_hash,
        "registry_sha256": sha256(registry_path), "panel_sha256": sha256(out / "panel.csv"),
        "identity_mode": "source_territory_id", "n_entities": len(balanced_ids), "months": months,
        "selection": "retrospective balanced panel"})
    return audit
