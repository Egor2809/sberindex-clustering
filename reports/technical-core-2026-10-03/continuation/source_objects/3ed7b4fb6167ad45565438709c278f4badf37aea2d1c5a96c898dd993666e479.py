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


def read_verified_artifact(directory, relative, manifest_name="SHA256.json"):
    """Read exactly the bytes bound by an archive's checksum manifest.

    The checked bytes are returned to the parser, avoiding a second read that
    could consume different content. The manifest itself is the trust anchor
    from the versioned source archive, not an authenticity signature.
    """
    directory = Path(directory).resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Artifact path must stay inside the archive")
    path = (directory / relative_path).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Artifact path escaped the archive")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate manifest field: " + key)
            result[key] = value
        return result
    manifest = json.loads((directory / manifest_name).read_bytes(), object_pairs_hook=unique_object)
    key = relative_path.as_posix()
    expected = manifest.get(key) if isinstance(manifest, dict) else None
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError("Missing or invalid artifact fingerprint: " + key)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Published artifact fingerprint differs: " + key)
    return raw
