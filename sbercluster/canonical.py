"""Prepare the official ID-coded snapshot; no model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import polars as pl
from .io import CATEGORIES, TOTAL, read_export, sha256, write_json
from .data import preparation_stage, publish_preparation, _audit_and_prepare
from .input_contracts import _canonical_territory

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
    with preparation_stage(root) as stage:
        csvs = list((root / 'data/raw').glob('*.csv'))
        if len(csvs) > 1:
            raise ValueError('Multiple CSV references: specify one vintage')
        source = root / 'artifacts/sources/acquisition'
        paths = [source/'hackathonlicence/consumption.parquet',
                 source/'t_dict_municipal_districts.xlsx',
                 source/'hackathonlicence/market_access.parquet', *csvs]
        fingerprints = {path.relative_to(root).as_posix():sha256(path) for path in paths}
        if csvs:
            _audit_and_prepare(csvs[0], stage, source_sha256=fingerprints[csvs[0].relative_to(root).as_posix()])
        audit = _prepare(root, stage, fingerprints)
        if set((root/'data/raw').glob('*.csv')) != set(csvs):
            raise ValueError('CSV references changed during preparation')
        for name,expected in fingerprints.items():
            if sha256(root/name) != expected:
                raise ValueError('Source changed during preparation: '+name)
        publish_preparation(stage, root)
    return audit


def _prepare(root, output_root, fingerprints):
    source = root / "artifacts/sources/acquisition"
    pq = source / "hackathonlicence/consumption.parquet"
    registry_path = source / "t_dict_municipal_districts.xlsx"
    source_hash = lambda path: fingerprints[path.relative_to(root).as_posix()]
    d = read_parquet(pq).drop(columns=["__index_level_0__"], errors="ignore")
    if set(d.columns) != {"date", "territory_id", "category", "value"}:
        raise ValueError("Official parquet schema changed")
    if d.empty:
        raise ValueError('Official consumption snapshot is empty')
    d['territory_id'] = d.territory_id.map(_canonical_territory).map(int)
    d["period"] = pd.to_datetime(d.date + "-01", format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    if d.duplicated(["territory_id", "period", "category"]).any():
        raise ValueError("Duplicate canonical key")
    if d.isna().any().any() or not np.isfinite(d.value).all() or (d.value <= 0).any():
        raise ValueError("Missing/nonpositive canonical data: review required")
    if set(d.category) != set(CATEGORIES + [TOTAL]):
        raise ValueError("Category schema changed")
    registry = pd.read_excel(registry_path, dtype={"oktmo": str})
    needed = {'territory_id','year_from','year_to','municipal_district_name','region_name',
              'region_code','oktmo','municipal_district_center_lat','municipal_district_center_lon'}
    if not needed <= set(registry.columns) or registry.empty:
        raise ValueError('Official registry schema is incomplete')
    registry['territory_id'] = registry.territory_id.map(_canonical_territory).map(int)
    intervals = registry[['year_from','year_to']].to_numpy(dtype=float)
    if (not np.isfinite(intervals).all() or not np.equal(intervals,np.floor(intervals)).all()
            or not (intervals[:,0] < intervals[:,1]).all()):
        raise ValueError('Registry year intervals must be finite ordered integers')
    mapped = active_registry(d, registry)
    market_path = source / 'hackathonlicence/market_access.parquet'
    market = read_parquet(market_path)
    if not {'territory_id','market_access'} <= set(market.columns) or market.empty:
        raise ValueError('Official market-access schema is incomplete')
    market['territory_id'] = market.territory_id.map(_canonical_territory).map(int)
    if market.territory_id.duplicated().any():
        raise ValueError('Duplicate market-access territory')
    if not np.isfinite(market.market_access.to_numpy(dtype=float)).all():
        raise ValueError('Market-access values must be finite')
    csvs = list((root / "data/raw").glob("*.csv"))
    if len(csvs) > 1:
        raise ValueError("Multiple CSV references: specify one vintage")
    out, reports = output_root / "data/processed", output_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    csv_hash = None
    if csvs:
        csv = read_export(csvs[0]).rename(columns={"category_15": "category"})
        key = ["period", "category", "value"]
        mismatch = d.groupby(key).size().subtract(csv.groupby(key).size(), fill_value=0)
        if (mismatch != 0).any():
            raise ValueError("Parquet and supplied CSV differ; review vintage")
        csv_hash = source_hash(csvs[0])
        write_json(reports / "source_alignment.json", {
            "csv_sha256": csv_hash, "parquet_sha256": source_hash(pq),
            "compared": ["period", "category", "value", "multiplicity"],
            "differing_multiset_keys": 0, "csv_rows": len(csv), "parquet_rows": len(d),
            "canonical_key_duplicates": 0,
            "scope": "Full numeric payload equivalence; use source territory_id directly, no name inference"})
    else:
        write_json(reports / "source_alignment.json", {"status": "reference_CSV_not_supplied",
            "parquet_sha256": source_hash(pq), "parquet_rows": len(d), "canonical_key_duplicates": 0})
    wide = d.pivot(index=["territory_id", "period"], columns="category", values="value")
    months = sorted(d.period.unique())
    calendar = pd.date_range(months[0], months[-1], freq="MS").strftime("%Y-%m-%d").tolist()
    if months != calendar:
        raise ValueError("Whole calendar months missing")
    complete = wide.notna().all(axis=1).groupby(level=0).sum()
    balanced_ids = complete[complete.eq(len(months))].index
    if len(balanced_ids) == 0:
        raise ValueError('No complete source territories in the prepared panel')
    panel = wide.loc[balanced_ids].reset_index()
    meta = active_registry(panel[["territory_id", "period"]], registry)
    required_metadata = ['territory_id','municipal_district_name','region_name','region_code','oktmo']
    if meta[required_metadata].isna().any().any():
        raise ValueError('Retained territory identity and regional metadata is incomplete')
    # Missing centers are documented source metadata, never model inputs.
    coordinates = meta[['municipal_district_center_lat','municipal_district_center_lon']].to_numpy(dtype=float)
    if np.isinf(coordinates).any():
        raise ValueError('Provided retained-territory coordinates must be finite')
    cols = ["territory_id", "period", "municipal_district_name", "region_name", "region_code", "oktmo",
            "municipal_district_center_lat", "municipal_district_center_lon"]
    panel = panel.merge(meta[cols], on=["territory_id", "period"], validate="one_to_one")
    latest = meta.sort_values("period").groupby("territory_id").tail(1).set_index("territory_id")
    panel["mo"] = panel.territory_id.map(latest.municipal_district_name)
    panel["entity_id"] = "tid_" + panel.territory_id.astype(str)
    panel["boundary_review_flag"] = np.where(panel.region_code.eq(20),
        "chechnya_registry_caveat", "annual_registry_2024_coverage_caveat")
    panel = panel.sort_values(["entity_id", "period"])
    # RFC4180 line endings reproduce the original Windows snapshot on every OS.
    panel.to_csv(out / "panel.csv", index=False, encoding="utf-8", lineterminator="\r\n")
    mapped.to_csv(out / "canonical_long.csv", index=False, encoding="utf-8", lineterminator="\r\n")
    excluded = pd.DataFrame({"territory_id": sorted(set(d.territory_id)-set(balanced_ids))})
    excluded["observed_complete_months"] = excluded.territory_id.map(complete)
    excluded["reason"] = "not_complete_all_months_retrospective_panel"
    excluded.to_csv(reports / "excluded_territories.csv", index=False, lineterminator="\r\n")
    market.to_csv(out / "market_access_2024.csv", index=False, lineterminator="\r\n")
    ratio = wide[CATEGORIES].sum(axis=1) / wide[TOTAL]
    audit_path = reports / "data_audit.json"
    prior = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    if (prior.get('sha256') != source_hash(pq)
            and not (csv_hash is not None and prior.get('sha256') == csv_hash)):
        prior = {}
    audit = {**prior, "source": "official consumption.parquet; see source_alignment.json for optional CSV comparison",
        "sha256": source_hash(pq), "rows": len(d), "n_territories": int(d.territory_id.nunique()),
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
        "market_access_sha256": source_hash(market_path),
        "csv_name_conflicts_resolved_by": "original ID-coded parquet, not guessed name matching",
        "panel_selection": "balanced over full 2023-2024: retrospective atlas, not an as-of forecast cohort"}
    csv_only = ["n_unique_names", "ambiguous_names", "rows_with_duplicate_key", "rows_excluded_due_to_ambiguous_name", "unique_unambiguous_names", "unambiguous_incomplete_names"]
    diagnostics = dict(prior.get('csv_export_diagnostics', {}))
    diagnostics.update({key:audit.pop(key) for key in csv_only if key in audit})
    audit["csv_export_diagnostics"] = diagnostics
    audit["csv_export_diagnostics"]["meaning"] = "name-only export diagnostics; these exclusions do NOT apply to the canonical ID panel"
    write_json(audit_path, audit)
    write_json(out / "manifest.json", {"raw_sha256": source_hash(pq), "csv_sha256": csv_hash,
        "registry_sha256": source_hash(registry_path), "panel_sha256": sha256(out / "panel.csv"),
        "market_access_sha256": source_hash(market_path),
        "identity_mode": "source_territory_id", "n_entities": len(balanced_ids), "months": months,
        "selection": "retrospective balanced panel"})
    return audit
