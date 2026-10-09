from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster.io import TOTAL
from sbercluster.research import annual_profile, load_development
from sbercluster.research_validation import (consensus_peers, nearest_indices,
                                             regional_bootstrap_difference, transform_frozen)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dummies(codes) -> np.ndarray:
    values = pd.Series(np.asarray(codes)).astype(str)
    return pd.get_dummies(values, drop_first=True).to_numpy(float)


def ols(design: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return coef, residual


def r_squared(residual: np.ndarray, y: np.ndarray) -> float:
    return float(1 - (residual @ residual) / ((y - y.mean()) @ (y - y.mean())))


def nested_f(rss_reduced: float, rss_full: float, q: int, n: int, k_full: int) -> float:
    return float(((rss_reduced - rss_full) / q) / (rss_full / (n - k_full)))


def cluster_wald(design: np.ndarray, residual: np.ndarray, clusters, columns) -> dict:
    n, k = design.shape
    bread = np.linalg.pinv(design.T @ design)
    codes = pd.factorize(pd.Series(np.asarray(clusters)).astype(str))[0]
    groups = int(codes.max() + 1)
    scores = np.zeros((groups, k))
    np.add.at(scores, codes, design * residual[:, None])
    meat = scores.T @ scores
    scale = groups / (groups - 1) * (n - 1) / (n - k)
    covariance = scale * bread @ meat @ bread
    return {"covariance": covariance, "groups": groups, "columns": list(columns)}


def wald_test(coef: np.ndarray, covariance: np.ndarray, columns, groups: int) -> dict:
    b = coef[columns]
    v = covariance[np.ix_(columns, columns)]
    statistic = float(b @ np.linalg.pinv(v) @ b / len(columns))
    return {"f": statistic, "df1": len(columns), "df2": groups - 1,
            "p_value": float(stats.f.sf(statistic, len(columns), groups - 1))}


def freedman_lane(base: np.ndarray, full: np.ndarray, y: np.ndarray, strata, q: int,
                  observed_f: float, rng: np.random.Generator, permutations: int) -> float:
    coef_base, residual_base = ols(base, y)
    fitted = base @ coef_base
    strata = pd.factorize(pd.Series(np.asarray(strata)).astype(str))[0]
    blocks = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    exceed = 0
    for _ in range(permutations):
        shuffled = residual_base.copy()
        for block in blocks:
            shuffled[block] = residual_base[rng.permutation(block)]
        y_star = fitted + shuffled
        _, r0 = ols(base, y_star)
        _, r1 = ols(full, y_star)
        if nested_f(r0 @ r0, r1 @ r1, q, len(y), full.shape[1]) >= observed_f - 1e-12:
            exceed += 1
    return float((exceed + 1) / (permutations + 1))


def increment_over_population(frame: pd.DataFrame, outcome: dict, controls: str,
                              rng: np.random.Generator, permutations: int) -> dict:
    data = frame.dropna(subset=[outcome["column"], "population"])
    y = data[outcome["column"]].to_numpy(float)
    if outcome["transform"] == "log":
        y = np.log(y)
    blocks = [np.ones((len(data), 1)), dummies(data["region_code"]),
              np.log(data["population"].to_numpy(float))[:, None]]
    if controls == "region_population_mo_type":
        blocks.append(dummies(data["municipal_district_type"]))
    base = np.column_stack(blocks)
    types = dummies(data["cluster"])
    full = np.column_stack([base, types])
    type_columns = list(range(base.shape[1], full.shape[1]))
    _, r0 = ols(base, y)
    coef, r1 = ols(full, y)
    r2_base, r2_full = r_squared(r0, y), r_squared(r1, y)
    q = len(type_columns)
    f = nested_f(r0 @ r0, r1 @ r1, q, len(y), np.linalg.matrix_rank(full))
    robust = cluster_wald(full, r1, data["region_code"], type_columns)
    wald = wald_test(coef, robust["covariance"], type_columns, robust["groups"])
    pop_only = np.column_stack([np.ones((len(data), 1)), np.log(data["population"].to_numpy(float))[:, None]])
    _, rp = ols(pop_only, y)
    reference = sorted(data["cluster"].astype(str).unique())[0]
    names = sorted(data["cluster"].astype(str).unique())[1:]
    se = np.sqrt(np.diag(robust["covariance"]))[type_columns]
    return {
        "outcome": outcome["column"], "label": outcome["label"], "transform": outcome["transform"],
        "controls": controls, "n": int(len(y)), "regions": robust["groups"],
        "r2_population_only": r_squared(rp, y),
        "r2_base": r2_base, "r2_with_type": r2_full, "delta_r2": r2_full - r2_base,
        "partial_r2_type": (r2_full - r2_base) / (1 - r2_base),
        "classical_f": f, "classical_p": float(stats.f.sf(f, q, len(y) - np.linalg.matrix_rank(full))),
        "cluster_robust_wald": wald,
        "freedman_lane_permutation_p": freedman_lane(base, full, y, data["region_code"], q, f, rng, permutations),
        "type_coefficients_vs_type": reference,
        "type_coefficients": {name: {"estimate": float(coef[c]), "cluster_se": float(s)}
                              for name, c, s in zip(names, type_columns, se)},
    }


def peer_errors(root: Path, research_config: str, labels_path: str) -> dict:
    cfg = json.loads((root / research_config).read_text("utf-8"))
    _, _, monthly, ids, scaler, _ = load_development(root, cfg)
    reference = annual_profile(monthly)
    labels_frame = pd.read_csv(root / labels_path, dtype={"entity_id": str})
    if labels_frame["entity_id"].tolist() != list(ids):
        raise ValueError("Label order differs from the development panel")
    labels = labels_frame["cluster"].to_numpy()
    panel = pd.read_csv(root / "data/processed/panel.csv", dtype={"entity_id": str, "period": str})
    future = annual_profile(np.stack([
        transform_frozen(frame.set_index("entity_id").loc[ids].reset_index(), scaler)
        for period, frame in panel.groupby("period", sort=True) if period >= "2024-01-01"]))
    metadata = panel[panel.period == "2023-12-01"].set_index("entity_id").loc[ids]
    regions = metadata.region_code.to_numpy()
    level_2023 = panel[panel.period < "2024-01-01"].groupby("entity_id")[TOTAL].mean().loc[ids].to_numpy()
    level_2024 = panel[panel.period >= "2024-01-01"].groupby("entity_id")[TOTAL].mean().loc[ids].to_numpy()
    growth = 100 * (level_2024 / level_2023 - 1)
    lat = metadata.municipal_district_center_lat.to_numpy(float)
    lon = metadata.municipal_district_center_lon.to_numpy(float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    for _ in range(len(ids)):
        region_counts = Counter(regions[valid])
        cluster_counts = Counter(labels[valid])
        next_valid = valid & np.array([region_counts[r] >= 2 and cluster_counts[c] >= 2
                                       for r, c in zip(regions, labels)])
        if np.array_equal(valid, next_valid):
            break
        valid = next_valid
    x = reference[valid]
    target = growth[valid]
    selected_regions = regions[valid]
    distance = cdist(x, x)
    peers = {
        "temporal_consensus_15": consensus_peers(monthly[:, valid, :], 15, 60)[0],
        "profile_15": nearest_indices(distance, 15),
        "region_all": nearest_indices(np.zeros_like(distance), len(x) - 1,
                                      selected_regions[:, None] == selected_regions[None, :]),
    }
    errors = {name: np.abs(target - np.array([np.median(target[n]) for n in neighbors]))
              for name, neighbors in peers.items()}
    return {"errors": errors, "regions": selected_regions, "n": int(valid.sum())}


def peer_intervals(root: Path, cfg: dict) -> dict:
    result = peer_errors(root, cfg["research_config"], cfg["labels"])
    errors, regions = result["errors"], result["regions"]
    means = {name: float(e.mean()) for name, e in errors.items()}
    for name, value in cfg["published_peer_means"].items():
        if abs(means[name] - value) > 5e-4:
            raise ValueError(f"Peer error for {name} does not match the published value")
    contrasts = []
    for method, baseline in cfg["peer_contrasts"]:
        contrasts.append({"method": method, "baseline": baseline,
                          **regional_bootstrap_difference(errors[method], errors[baseline], regions,
                                                          seed=cfg["seed"], repetitions=cfg["bootstrap_repetitions"])})
    return {"n_territories": result["n"], "mean_absolute_growth_error_pp": means,
            "contrasts": contrasts,
            "reading": "mean_improvement > 0: method is closer to observed 2024 growth than baseline"}


def run(cfg: dict, root: Path = ROOT, with_peers: bool = True) -> dict:
    indicators = pd.read_csv(root / cfg["indicators"], dtype={"entity_id": str, "region_code": str})
    access = pd.read_csv(root / cfg["market_access"])
    indicators = indicators.merge(access, on="territory_id", how="left", validate="one_to_one")
    rng = np.random.default_rng(cfg["seed"])
    models = [increment_over_population(indicators, outcome, controls, rng, cfg["permutations"])
              for outcome in cfg["outcomes"] for controls in cfg["controls"]]
    result = {
        "question": "Do the four types explain external indicators beyond region and population?",
        "inputs": {key: {"path": cfg[key], "sha256": digest(root / cfg[key])}
                   for key in ("indicators", "market_access", "labels")},
        "seed": cfg["seed"], "permutations": cfg["permutations"],
        "type_names": cfg["type_names"],
        "increment_over_population": models,
        "notes": ["labels are fixed; nothing is refitted",
                  "cluster-robust Wald test: CR1 by region, F reference with G-1 denominator df",
                  "Freedman-Lane permutation of reduced-model residuals within regions"],
    }
    if with_peers:
        result["peer_bootstrap"] = peer_intervals(root, cfg)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/type_validation.json")
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
