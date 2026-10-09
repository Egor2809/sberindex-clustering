from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]
TOTAL = "Все категории"
SEAM = ("2023-12-01", "2024-01-01")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def median_shares(panel: pd.DataFrame) -> pd.DataFrame:
    shares = panel[CATEGORIES].div(panel[TOTAL], axis=0) * 100
    shares["period"] = panel["period"].to_numpy()
    return shares.groupby("period")[CATEGORIES].median().sort_index()


def trend_step(series: np.ndarray) -> dict:
    y = np.asarray(series, float)
    t = np.arange(len(y), dtype=float)
    step = (t >= 12).astype(float)
    design = np.column_stack([np.ones_like(t), t, step])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return {"monthly_trend": float(coef[1]), "step_at_january_2024": float(coef[2]),
            "trend_over_12_months": float(12 * coef[1]),
            "residual_sd": float(residual.std(ddof=3))}


def seam_summary(series: pd.Series) -> dict:
    values = series.sort_index()
    periods = list(values.index)
    changes = values.diff().dropna()
    seam = float(values[SEAM[1]] - values[SEAM[0]])
    inside = changes.drop(SEAM[1])
    larger = int((inside.abs() >= abs(seam)).sum())
    first, second = values.iloc[:12].to_numpy(), values.iloc[12:].to_numpy()
    return {
        "december_2023_to_january_2024": seam,
        "within_year_monthly_changes_abs_ge_seam": larger,
        "within_year_monthly_changes": int(len(inside)),
        "within_year_change_sd": float(inside.std(ddof=1)),
        "november_to_december_2023": float(first[11] - first[10]),
        "november_to_december_2024": float(second[11] - second[10]),
        "january_2024_minus_january_2023": float(second[0] - first[0]),
        "january_minus_february_2023": float(values[periods[0]] - values[periods[1]]),
        "january_minus_february_2024": float(values[periods[12]] - values[periods[13]]),
        "january_to_december_2023": float(first[-1] - first[0]),
        "january_to_december_2024": float(second[-1] - second[0]),
        "february_to_november_2023": float(first[10] - first[1]),
        "february_to_november_2024": float(second[10] - second[1]),
        "same_month_2024_minus_2023_mean": float((second - first).mean()),
        "same_month_2024_minus_2023_min": float((second - first).min()),
        **trend_step(values.to_numpy()),
    }


def territory_growth(panel: pd.DataFrame, category: str, start: str, end: str) -> dict:
    share = panel.assign(share=panel[category] / panel[TOTAL] * 100).pivot(
        index="entity_id", columns="period", values="share")
    delta = share[end] - share[start]
    return {"start": start, "end": end, "n": int(delta.notna().sum()),
            "share_of_territories_with_increase": float((delta > 0).mean()),
            "median_change_pp": float(delta.median())}


def type_sizes(assignments: pd.DataFrame) -> dict:
    counts = assignments.groupby(["period", "cluster"]).size().unstack(fill_value=0).sort_index()
    result = {}
    for cluster in counts.columns:
        series = counts[cluster].astype(float)
        summary = seam_summary(series)
        result[str(int(cluster))] = {"monthly_counts": {k: int(v) for k, v in series.items()},
                                     **{k: summary[k] for k in (
                                         "december_2023_to_january_2024",
                                         "november_to_december_2023", "november_to_december_2024",
                                         "january_2024_minus_january_2023",
                                         "within_year_monthly_changes_abs_ge_seam",
                                         "january_to_december_2023", "january_to_december_2024",
                                         "same_month_2024_minus_2023_mean")}}
    return result


def run(cfg: dict, root: Path = ROOT) -> dict:
    panel_path = root / cfg["panel"]
    assignments_path = root / cfg["monthly_assignments"]
    panel = pd.read_csv(panel_path, dtype={"entity_id": str, "period": str})
    assignments = pd.read_csv(assignments_path, dtype={"entity_id": str, "period": str})
    if panel.groupby("period").size().nunique() != 1 or panel["period"].nunique() != 24:
        raise ValueError("Balanced 24-month panel required")
    medians = median_shares(panel)
    categories = {c: {"monthly_median_share_pct": {k: float(v) for k, v in medians[c].items()},
                      **seam_summary(medians[c])} for c in CATEGORIES}
    market = cfg["focus_category"]
    within = [territory_growth(panel, market, a, b) for a, b in cfg["within_year_windows"]]
    return {
        "question": "Is the change at the 2023/2024 seam a break or January seasonality plus trend?",
        "inputs": {"panel": cfg["panel"], "panel_sha256": digest(panel_path),
                   "monthly_assignments": cfg["monthly_assignments"],
                   "monthly_assignments_sha256": digest(assignments_path)},
        "share_definition": "category / 'Все категории' * 100, median over territories per month",
        "categories": categories,
        "focus_category": market,
        "within_year_territory_change": within,
        "type_sizes_fixed_2023_prototypes": type_sizes(assignments),
        "external_reference": cfg["external_reference"],
        "documentation_check": cfg["documentation_check"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/january_check.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text("utf-8"))
    result = run(cfg)
    out = ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
