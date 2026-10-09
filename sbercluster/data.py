from __future__ import annotations

import hashlib
import platform
from contextlib import contextmanager
import os
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from .io import CATEGORIES, TOTAL, KEY, read_export, sha256, write_json

PREPARED_FILES = {
    'data/processed/panel.csv', 'data/processed/manifest.json',
    'data/processed/canonical_long.csv', 'data/processed/market_access_2024.csv',
    'reports/ambiguous_names.csv', 'reports/excluded_names.csv',
    'reports/excluded_territories.csv', 'reports/source_alignment.json',
    'reports/data_audit.json', 'reports/environment.json',
}


class PreparationRollbackError(RuntimeError):
    """Publication failed and backups must be retained for manual recovery."""

    def __init__(self, stage, failures):
        self.recovery_directory = Path(stage)
        self.failures = failures
        super().__init__('Preparation rollback failed; recoverable files and lock retained at ' + str(stage))


@contextmanager
def preparation_stage(root):
    """Keep all generated preparation files private until validation succeeds."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.prepare.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(str(os.getpid()) + '\n')
    stage = None
    retain = False
    try:
        stage = Path(tempfile.mkdtemp(prefix='.prepare-', dir=root))
        prior = root / 'reports/data_audit.json'
        if prior.is_file():
            (stage / 'reports').mkdir()
            shutil.copyfile(prior, stage / 'reports/data_audit.json')
        yield stage
    except PreparationRollbackError:
        retain = True
        raise
    finally:
        if not retain:
            try:
                if stage is not None and stage.exists():
                    if stage.resolve().parent != root:
                        raise ValueError('Preparation staging escaped the requested root')
                    shutil.rmtree(stage)
            finally:
                lock.unlink()


def publish_preparation(stage, root):
    """Promote validated files, manifest last; roll back ordinary failures.

    Individual replacements are atomic. This is not a multi-file power-loss
    transaction; callers must wait for successful return before consuming it.
    """
    root, stage = Path(root).resolve(), Path(stage).resolve()
    names = [name for name in PREPARED_FILES if (stage / name).is_file()]
    if 'data/processed/manifest.json' not in names:
        raise ValueError('Prepared manifest required before publication')
    names.sort(key=lambda name: (name == 'data/processed/manifest.json', name))
    backups = {}
    for name in names:
        target = root / name
        if not target.resolve().is_relative_to(root) or target.is_symlink():
            raise ValueError('Prepared target escapes the requested root')
        if target.exists():
            if not target.is_file():
                raise ValueError('Prepared target must be a file: ' + name)
            backup = stage / '.backups' / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, backup)
            backups[name] = backup
    promoted = []
    try:
        for name in names:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
            promoted.append(name)
    except BaseException as original:
        failures = []
        for name in reversed(promoted):
            target = root / name
            try:
                if name in backups:
                    os.replace(backups[name], target)
                else:
                    target.unlink()
            except BaseException as error:
                failures.append((name, error))
        if failures:
            raise PreparationRollbackError(stage, failures) from original
        raise


def audit_and_prepare(input_path, root):
    with preparation_stage(root) as stage:
        expected = sha256(input_path)
        audit = _audit_and_prepare(input_path, stage, source_sha256=expected)
        if sha256(input_path) != expected:
            raise ValueError('CSV source changed during preparation')
        publish_preparation(stage, root)
    return audit


def _audit_and_prepare(input_path, root, *, source_sha256=None):
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
        "source": str(Path(input_path).name), "sha256": source_sha256 or sha256(input_path),
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
