from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from .io import CATEGORIES, TOTAL, KEY, read_export, sha256, write_json


def audit_and_prepare(input_path, root):
    """Audit raw values only; exclude whole ambiguous names, never average them."""
    root = Path(root)
    reports, processed = root / "reports", root / "data/processed"
    reports.mkdir(exist_ok=True, parents=True)
    processed.mkdir(exist_ok=True, parents=True)
    d = read_export(input_path)
    if set(d.category_15.unique()) != set(CATEGORIES + [TOTAL]):
        raise ValueError("Unexpected categories: review schema instead of silently ignoring fields")
    if not (d.obs_status.eq("A").all() and d.freq.eq("Месяц").all()
            and d.unit_measure.eq("руб.").all() and d.unit_mult.eq(0).all()):
        raise ValueError("Status, frequency or units changed: human measurement review required")
    bad_values = ~np.isfinite(d.value) | d.value.le(0)
    duplicate = d.duplicated(KEY, keep=False)
    ambiguous = sorted(d.loc[duplicate, "mo"].unique())
    invalid_names = set(d.loc[bad_values, "mo"])
    clean = d[~d.mo.isin(set(ambiguous) | invalid_names)].copy()
    wide = clean.pivot(index=["mo", "period"], columns="category_15", values="value")
    months = sorted(d.period.unique())
    expected = pd.date_range(months[0], months[-1], freq="MS").strftime("%Y-%m-%d").tolist()
    if months != expected:
        raise ValueError("Entire calendar months absent: review before panel preparation")
    complete_month = wide.notna().all(axis=1)
    coverage = complete_month.groupby(level="mo").sum()
    names = coverage[coverage.eq(len(months))].index
    panel = wide.loc[names].reset_index()
    panel.insert(0, "entity_id", panel.mo.map(lambda x: "name_" + hashlib.sha256(x.encode()).hexdigest()[:20]))
    if panel.groupby("entity_id").mo.nunique().max() != 1:
        raise ValueError("Name hash collision")
    panel = panel.sort_values(["entity_id", "period"])
    panel.to_csv(processed / "panel.csv", index=False, encoding="utf-8")
    conflicts = d.loc[duplicate].groupby("mo").agg(rows_with_duplicate_key=("value", "size"))
    conflicts["all_excluded_rows"] = d[d.mo.isin(ambiguous)].groupby("mo").size()
    conflicts.reset_index().to_csv(reports / "ambiguous_names.csv", index=False, encoding="utf-8")
    excluded = pd.DataFrame({"mo": sorted(set(d.mo) - set(names))})
    excluded["reason"] = excluded.mo.map(lambda x: "ambiguous_name" if x in ambiguous else "invalid_value" if x in invalid_names else "incomplete_24_month_panel")
    excluded.to_csv(reports / "excluded_names.csv", index=False, encoding="utf-8")
    p = wide[complete_month]
    ratio = p[CATEGORIES].sum(axis=1) / p[TOTAL]
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(input_path).name), "sha256": sha256(input_path),
        "rows": len(d), "months": months, "n_months": len(months),
        "n_unique_names": int(d.mo.nunique()), "n_categories_including_total": int(d.category_15.nunique()),
        "categories": CATEGORIES + [TOTAL], "missing_cells": int(d.isna().sum().sum()),
        "nonpositive_or_nonfinite_values": int(bad_values.sum()),
        "ambiguous_names": len(ambiguous), "rows_with_duplicate_key": int(duplicate.sum()),
        "rows_excluded_due_to_ambiguous_name": int(d.mo.isin(ambiguous).sum()),
        "unique_unambiguous_names": int(clean.mo.nunique()),
        "complete_panel_names": len(names), "panel_rows": len(panel),
        "unambiguous_incomplete_names": int(clean.mo.nunique() - len(names)),
        "observed_category_sum_over_total_quantiles": {str(k): float(v) for k, v in ratio.quantile([0, .01, .5, .99, 1]).items()},
        "observed_category_sum_above_total": int((ratio > 1).sum()),
        "identity_status": "provisional names; NOT verified municipal IDs or stable boundaries",
        "semantics_status": "numerical consistency is NOT proof of common denominator or disjoint categories",
        "panel_selection": "complete coverage over whole 2023-2024; retrospective selection, not a live forecasting cohort",
        "model_training_executed": False,
    }
    write_json(reports / "data_audit.json", audit)
    write_json(reports / "environment.json", {"python": platform.python_version(), "platform": platform.platform(), "packages": {n: version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "networkx", "polars", "openpyxl"]}})
    write_json(processed / "manifest.json", {"raw_sha256": audit["sha256"], "panel_sha256": sha256(processed / "panel.csv"), "identity_mode": "unique_name_provisional", "categories": CATEGORIES, "n_entities": len(names), "months": months})
    return audit
