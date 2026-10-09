from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

CATEGORIES = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
TOTAL = "Все категории"
KEY = ["mo", "period", "category_15"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical_period(values):
    """UTC API timestamps represent Moscow month starts; avoid a one-month shift."""
    series = pd.Series(values, dtype="string")
    if series.isna().any():
        raise ValueError("Missing period")
    if series.str.contains("T").any():
        stamp = pd.to_datetime(series, utc=True, errors="raise").dt.tz_convert("Europe/Moscow")
        if not (stamp.dt.day == 1).all():
            raise ValueError("API timestamps are not Moscow month starts")
        return stamp.dt.strftime("%Y-%m-01")
    stamp = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
    if not (stamp.dt.day == 1).all():
        raise ValueError("Expected month-start dates")
    return stamp.dt.strftime("%Y-%m-%d")


def read_export(path):
    d = pd.read_csv(path, sep=";", dtype={"mo": "string", "period": "string", "category_15": "string"})
    required = set(KEY + ["value", "obs_status", "unit_measure", "unit_mult", "freq"])
    if not required.issubset(d.columns):
        raise ValueError(f"Missing columns: {required - set(d.columns)}")
    if d[KEY].isna().any().any():
        raise ValueError("Missing observation key")
    d["period"] = canonical_period(d["period"])
    d["value"] = pd.to_numeric(d["value"], errors="raise")
    return d


def code_revision(root):
    """Describe a Git checkout or a source archive without inventing a clean state."""
    import os
    import subprocess
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip())
        return {'git_commit': commit, 'git_worktree_modified': dirty, 'revision_source': 'git'}
    except (OSError, subprocess.CalledProcessError):
        return {'git_commit': os.environ.get('SOURCE_REVISION') or None,
                'git_worktree_modified': None, 'revision_source': 'source_archive; module hashes are authoritative'}
