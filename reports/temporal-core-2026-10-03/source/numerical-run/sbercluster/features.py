from __future__ import annotations
import numpy as np
import pandas as pd
from .io import CATEGORIES, TOTAL


def validate_contract(cfg):
    contract = cfg["data_contract"]
    needed = ["measurement_definition_verified"]
    if cfg["features"]["mode"] == "hellinger_with_residual":
        needed += ["common_denominator_verified", "categories_disjoint_verified"]
    missing = [k for k in needed if contract.get(k) is not True]
    if missing or not contract.get("evidence"):
        raise ValueError("Unresolved data contract: " + ", ".join(missing or ["evidence"]))
    if contract.get("identity_mode") == "unique_name_provisional":
        raise ValueError("Provisional names cannot support confirmed economic transitions")


def profile_matrix(frame):
    """Hellinger embedding including residual, only after semantic validation.

    Euclidean distance between returned rows equals Hellinger distance.
    The total is a denominator, never an additional spending category.
    """
    values = frame[CATEGORIES].to_numpy(dtype=float)
    total = frame[TOTAL].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(total).all() or (values < 0).any() or (total <= 0).any():
        raise ValueError("Finite nonnegative categories and positive totals required")
    residual = total - values.sum(axis=1)
    if (residual < 0).any():
        raise ValueError("Category sum exceeds total; do not clip inconsistent observations")
    p = np.column_stack([values, residual]) / total[:, None]
    return np.sqrt(p) / np.sqrt(2)


def transform_frozen(frame, scaler):
    """Apply the full fitted embedding, without recalibrating on new observations.

    Older published scalers omit level_weight and used the zero-level setting.
    """
    mode = scaler.get("mode", "log_ratios_to_total")
    weight = scaler.get("level_weight", 0.0)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("level_weight must be in [0,1]")
    total = frame[TOTAL].to_numpy(dtype=float)
    if mode == "hellinger_with_residual":
        profile = profile_matrix(frame)
    elif mode == "log_ratios_to_total":
        values = frame[CATEGORIES].to_numpy(dtype=float)
        center = np.asarray(scaler["ratio_center"], dtype=float)
        scale = np.asarray(scaler["ratio_iqr"], dtype=float)
        if (center.shape != (len(CATEGORIES),) or scale.shape != center.shape
                or not np.isfinite(center).all() or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError("Invalid frozen relative-intensity scaler")
        if (not np.isfinite(values).all() or not np.isfinite(total).all()
                or (values <= 0).any() or (total <= 0).any()):
            raise ValueError("Positive finite estimates required")
        profile = (np.log(values / total[:, None]) - center) / scale / np.sqrt(len(CATEGORIES))
    else:
        raise ValueError("Unsupported feature mode")
    x = np.sqrt(1 - weight) * profile
    if weight > 0:
        center, scale = scaler["level_center"], scaler["level_iqr"]
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid frozen level scaler")
        x = np.column_stack([x, np.sqrt(weight) * (np.log(total) - center) / scale])
    if not np.isfinite(x).all():
        raise ValueError("Feature transformation produced nonfinite values")
    return x


def scalers_equal(first, second):
    """Exact frozen-space agreement, with the historical zero-level default."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    return ({**first, "level_weight": first.get("level_weight", 0.0)}
            == {**second, "level_weight": second.get("level_weight", 0.0)})


def make_slices(panel, cfg):
    validate_contract(cfg)
    mode = cfg["features"]["mode"]
    if mode not in ("hellinger_with_residual", "log_ratios_to_total"):
        raise ValueError("Unsupported feature mode")
    if cfg["features"]["seasonality"] != "same_month_cross_section":
        raise ValueError("Unsupported seasonality handling")
    if panel.empty or panel[["entity_id", "period"]].isna().any().any():
        raise ValueError("Nonempty panel with complete entity-month keys required")
    if not panel.entity_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Entity IDs must be nonempty strings")
    periods = panel.period
    if not periods.map(lambda value: isinstance(value, str)).all() or not periods.str.fullmatch(r"\d{4}-\d{2}-01").all():
        raise ValueError("Expected canonical month-start periods YYYY-MM-01")
    pd.to_datetime(periods, format="%Y-%m-%d", errors="raise")
    cutoff = cfg["features"]["calibration_end"]
    if not isinstance(cutoff, str) or not pd.Series([cutoff]).str.fullmatch(r"\d{4}-\d{2}-01").all():
        raise ValueError("Expected canonical calibration month YYYY-MM-01")
    pd.to_datetime(cutoff, format="%Y-%m-%d", errors="raise")
    validation = cfg.get("validation", {})
    if cutoff > validation.get("development_end", cutoff) or cutoff >= validation.get("confirmation_start", "9999-12-01"):
        raise ValueError("Calibration must remain within the development period")
    if panel.duplicated(["entity_id", "period"]).any():
        raise ValueError("Duplicate entity-month")
    weight = cfg["features"]["level_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("level_weight must be in [0,1]")
    # Level standardisation frozen using development observations only.
    ref = panel[panel.period <= cfg["features"]["calibration_end"]]
    if ref.empty:
        raise ValueError("Empty calibration period")
    if mode == "hellinger_with_residual":
        profile_matrix(ref)
    elif not np.isfinite(ref[CATEGORIES+[TOTAL]].to_numpy(dtype=float)).all() or (ref[CATEGORIES+[TOTAL]] <= 0).any().any():
        raise ValueError("Calibration requires positive finite observations")
    logs = np.log(ref[TOTAL].to_numpy())
    center = float(np.median(logs))
    scale = float(np.quantile(logs, .75) - np.quantile(logs, .25))
    ratio_center, ratio_iqr = None, None
    if mode == "log_ratios_to_total":
        ref_ratios = np.log(ref[CATEGORIES].to_numpy(dtype=float) / ref[TOTAL].to_numpy(dtype=float)[:, None])
        ratio_center = np.median(ref_ratios, axis=0)
        ratio_iqr = np.quantile(ref_ratios, .75, axis=0) - np.quantile(ref_ratios, .25, axis=0)
        if not np.isfinite(ratio_iqr).all() or (ratio_iqr <= 0).any():
            raise ValueError("Constant or nonfinite relative-intensity feature during calibration")
    if weight > 0 and scale <= 0:
        raise ValueError("Constant level during calibration")
    scaler = {"level_center": center, "level_iqr": scale,
              "ratio_center": ratio_center.tolist() if ratio_center is not None else None,
              "ratio_iqr": ratio_iqr.tolist() if ratio_iqr is not None else None,
              "mode": mode, "level_weight": weight, "calibration_end": cutoff}
    slices = []
    for period, f in panel.groupby("period", sort=True):
        f = f.sort_values("entity_id")
        x = transform_frozen(f, scaler)
        slices.append((period, f.entity_id.tolist(), x))
    return slices, scaler
