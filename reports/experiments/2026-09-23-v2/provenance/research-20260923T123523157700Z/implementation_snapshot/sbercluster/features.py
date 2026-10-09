from __future__ import annotations
import numpy as np
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


def make_slices(panel, cfg):
    validate_contract(cfg)
    mode = cfg["features"]["mode"]
    if mode not in ("hellinger_with_residual", "log_ratios_to_total"):
        raise ValueError("Unsupported feature mode")
    if cfg["features"]["seasonality"] != "same_month_cross_section":
        raise ValueError("Unsupported seasonality handling")
    if panel.duplicated(["entity_id", "period"]).any():
        raise ValueError("Duplicate entity-month")
    weight = cfg["features"]["level_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("level_weight must be in [0,1]")
    # Level standardisation frozen using development observations only.
    ref = panel[panel.period <= cfg["features"]["calibration_end"]]
    if ref.empty:
        raise ValueError("Empty calibration period")
    if not np.isfinite(ref[CATEGORIES+[TOTAL]].to_numpy(dtype=float)).all() or (ref[CATEGORIES+[TOTAL]] <= 0).any().any():
        raise ValueError("Calibration requires positive finite observations")
    logs = np.log(ref[TOTAL].to_numpy())
    center = float(np.median(logs))
    scale = float(np.quantile(logs, .75) - np.quantile(logs, .25))
    ref_ratios = np.log(ref[CATEGORIES].to_numpy(dtype=float) / ref[TOTAL].to_numpy(dtype=float)[:, None])
    ratio_center = np.median(ref_ratios, axis=0)
    ratio_iqr = np.quantile(ref_ratios, .75, axis=0) - np.quantile(ref_ratios, .25, axis=0)
    if (ratio_iqr <= 0).any():
        raise ValueError("Constant relative-intensity feature during calibration")
    if weight > 0 and scale <= 0:
        raise ValueError("Constant level during calibration")
    slices = []
    for period, f in panel.groupby("period", sort=True):
        f = f.sort_values("entity_id")
        if mode == "hellinger_with_residual":
            profile = profile_matrix(f)
        else:
            # Ratios of published estimates, NOT asserted expenditure shares.
            raw = f[CATEGORIES].to_numpy(dtype=float)
            total = f[TOTAL].to_numpy(dtype=float)
            if (raw <= 0).any() or (total <= 0).any() or not np.isfinite(raw).all() or not np.isfinite(total).all():
                raise ValueError("Positive finite estimates required")
            profile = (np.log(raw / total[:, None]) - ratio_center) / ratio_iqr / np.sqrt(len(CATEGORIES))
        x = np.sqrt(1 - weight) * profile
        if weight > 0:
            x = np.column_stack([x, np.sqrt(weight) * (np.log(f[TOTAL]) - center) / scale])
        slices.append((period, f.entity_id.tolist(), x))
    return slices, {"level_center": center, "level_iqr": scale, "ratio_center": ratio_center.tolist(),
                    "ratio_iqr": ratio_iqr.tolist(), "mode": mode,
                    "calibration_end": cfg["features"]["calibration_end"]}
