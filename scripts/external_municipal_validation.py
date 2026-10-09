"""Describe the four fixed consumption types with open municipal statistics.

Labels are fixed; nothing is refitted.  For every outcome the script reports
type medians and IQR, eta squared without controls and the within-stratum
partial R² after region (and region × municipal type) fixed effects, with
region-block bootstrap intervals and Kruskal–Wallis tests on raw values and on
within-region deviations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def eta_squared(values, groups) -> float | None:
    y = np.asarray(values, float)
    g = np.asarray(groups)
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 1e-12:
        return None
    between = sum(len(y[g == k]) * (y[g == k].mean() - y.mean()) ** 2 for k in np.unique(g))
    return float(between / total)


def center_by(matrix: np.ndarray, codes: np.ndarray) -> np.ndarray:
    n_groups = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=n_groups)
    sums = np.zeros((n_groups, matrix.shape[1]))
    np.add.at(sums, codes, matrix)
    means = sums / np.maximum(counts, 1)[:, None]
    return matrix - means[codes]


def within_eta_squared(values, groups, strata) -> float | None:
    """Share of within-stratum variance explained by group dummies."""
    y = np.asarray(values, float)
    levels = np.unique(groups)
    dummies = np.column_stack([(np.asarray(groups) == k).astype(float) for k in levels])
    codes = pd.factorize(np.asarray(strata), sort=True)[0]
    centered = center_by(np.column_stack([y, dummies]), codes)
    c = centered.T @ centered
    total = c[0, 0]
    if total <= 1e-12:
        return None
    explained = float(c[0, 1:] @ np.linalg.pinv(c[1:, 1:], rcond=1e-12) @ c[1:, 0])
    return float(np.clip(explained / total, 0, 1))


def strata_for(frame: pd.DataFrame, controls: str) -> pd.Series:
    if controls == "none":
        return pd.Series("all", index=frame.index)
    if controls == "region":
        return frame.region_code.astype(str)
    if controls == "region_type":
        return frame.region_code.astype(str) + "|" + frame.municipal_district_type
    raise ValueError(controls)


def transformed(frame: pd.DataFrame, column: str, transform: str) -> pd.Series:
    values = frame[column].astype(float)
    if transform == "log":
        return np.log(values.where(values > 0))
    if transform == "identity":
        return values
    raise ValueError(transform)


def region_bootstrap(
    frame: pd.DataFrame, y: np.ndarray, strata: pd.Series, draws: int, seed: int, by_region: bool
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    regions = frame.region_code.astype(str).to_numpy()
    unique = np.unique(regions)
    positions = {region: np.flatnonzero(regions == region) for region in unique}
    codes = pd.factorize(strata.astype(str), sort=True)[0]
    width = int(codes.max()) + 1
    clusters = frame.cluster.to_numpy()
    estimates = []
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[region] for region in chosen])
        block = np.concatenate([np.full(len(positions[region]), i) for i, region in enumerate(chosen)])
        strata_index = block * width + codes[index] if by_region else codes[index]
        value = within_eta_squared(y[index], clusters[index], strata_index)
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    return [float(v) for v in np.percentile(estimates, [2.5, 97.5])]


def within_region_deviation(frame: pd.DataFrame, y: pd.Series) -> pd.Series:
    return y - y.groupby(frame.region_code.astype(str)).transform("mean")


def kruskal_test(values: pd.Series, groups: pd.Series) -> dict[str, float] | None:
    samples = [values[groups == k].to_numpy() for k in sorted(groups.unique())]
    if len(samples) < 2 or any(len(s) < 2 for s in samples):
        return None
    statistic, p_value = kruskal(*samples)
    return {"H": float(statistic), "p_value": float(p_value)}


def describe(frame: pd.DataFrame, outcome: dict, names: dict, y: pd.Series, deviation: pd.Series) -> list[dict]:
    rows = []
    for cluster in sorted(frame.cluster.unique()):
        mask = frame.cluster == cluster
        raw = frame.loc[mask, outcome["column"]].astype(float)
        dev = deviation[mask].dropna()
        row = {
            "outcome": outcome["column"],
            "cluster": int(cluster),
            "type": names[str(cluster)],
            "n_total": int(mask.sum()),
            "n_observed": int(raw.notna().sum()),
            "median": float(raw.median()),
            "q25": float(raw.quantile(0.25)),
            "q75": float(raw.quantile(0.75)),
            "regions": int(frame.loc[mask & raw.notna(), "region_code"].nunique()),
        }
        if outcome["transform"] == "log":
            row["median_log"] = float(y[mask].median())
            row["within_region_median_pct"] = float(100 * (np.exp(dev.median()) - 1))
        else:
            row["within_region_median_pp"] = float(100 * dev.median())
        rows.append(row)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/external_municipal.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    indicators_path = root / config["indicators"]
    labels_path = root / config["labels"]
    output = root / config["output"]

    indicators = pd.read_csv(indicators_path, dtype={"region_code": str, "oktmo8": str})
    labels = pd.read_csv(labels_path)[["entity_id", "cluster"]]
    frame = indicators.drop(columns="cluster").merge(labels, on="entity_id", how="inner", validate="one_to_one")
    if len(frame) != len(labels) or len(frame) != len(indicators):
        raise ValueError("Indicators and frozen labels do not cover the same municipalities")
    if not (indicators.set_index("entity_id").cluster == labels.set_index("entity_id").cluster).all():
        raise ValueError("Cluster labels in indicators differ from the frozen reference assignments")

    names = config["cluster_names"]
    summary, effects = [], []
    for subset in config["subsets"]:
        part = frame if subset == "all" else frame[~frame.intracity_moscow_petersburg.astype(bool)]
        for outcome in config["outcomes"]:
            y_all = transformed(part, outcome["column"], outcome["transform"])
            observed = part[y_all.notna()]
            y = y_all[y_all.notna()]
            deviation = within_region_deviation(observed, y)
            if subset == "all":
                summary.extend(describe(part, outcome, names, y_all, deviation.reindex(part.index)))
            for controls in config["controls"]:
                strata = strata_for(observed, controls)
                effects.append({
                    "subset": subset,
                    "outcome": outcome["column"],
                    "transform": outcome["transform"],
                    "controls": controls,
                    "n": int(len(y)),
                    "regions": int(observed.region_code.nunique()),
                    "strata": int(strata.nunique()),
                    "eta_squared": within_eta_squared(y.to_numpy(), observed.cluster.to_numpy(), strata),
                    "region_bootstrap_percentile_95": region_bootstrap(
                        observed, y.to_numpy(), strata, config["bootstrap_draws"], config["seed"], controls != "none"
                    ),
                    "kruskal_wallis": kruskal_test(
                        y if controls == "none" else y - y.groupby(strata).transform("mean"), observed.cluster
                    ),
                })

    coverage = []
    for cluster in sorted(frame.cluster.unique()):
        mask = frame.cluster == cluster
        row = {"cluster": int(cluster), "type": names[str(cluster)], "n": int(mask.sum())}
        for outcome in config["outcomes"]:
            row[outcome["column"]] = int(frame.loc[mask, outcome["column"]].notna().sum())
        coverage.append(row)

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False, float_format="%.6g")
    results = {
        "scope": "Fixed four-type labels described by open municipal statistics; no clustering fitted, no causal claims.",
        "n": int(len(frame)),
        "inputs": {
            "indicators": {"path": config["indicators"], "sha256": digest(indicators_path)},
            "labels": {"path": config["labels"], "sha256": digest(labels_path)},
            "config": {"path": str(args.config.relative_to(root)), "sha256": digest(args.config)},
        },
        "script_sha256": digest(Path(__file__)),
        "outcomes": config["outcomes"],
        "coverage": coverage,
        "summary": summary,
        "effects": effects,
        "definitions": {
            "eta_squared_none": "between-type sum of squares / total sum of squares of the transformed outcome",
            "eta_squared_region": "partial R² of type dummies after removing region means (within-region share of variance explained)",
            "eta_squared_region_type": "same after removing region × municipal-type means",
            "kruskal_wallis_controlled": "test on deviations from the stratum mean; heuristic, deviations are not independent",
            "within_region_median": "median deviation of the type from its region mean: % for log outcomes, percentage points for shares",
        },
        "limitations": [
            "Wages and headcount cover large and medium organisations only and are counted where organisations are registered, not where people live.",
            "Population uses Rosstat 1 January 2024 where the 2023 OKTMO code is present in the bulletin and BD PMO 1 January 2023 otherwise.",
            "Sector shares are left empty when a section is absent or suppressed in the source, so their coverage is uneven and biased towards larger municipalities.",
            "Region-block bootstrap intervals are descriptive; partial R² is upward biased under a null.",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = pd.DataFrame([e for e in effects if e["subset"] == "all"]).pivot(index="outcome", columns="controls", values="eta_squared")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
