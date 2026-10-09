"""Frozen-cluster external association and leave-region-out economic probes.

Economic targets are never used to choose or fit clusters. The held-out region
contributes no outcomes to a probe, intercept, regional effect or scaler.
Cluster additions are compared with the original continuous consumer features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from sbercluster.io import CATEGORIES, TOTAL
from scripts.download_external_national import ROOT, sha256

OUTCOMES = ["log_annual_monthly_wage_rubles", "log_covered_organization_jobs_per_resident",
            "manufacturing_headcount_share", "agriculture_headcount_share",
            "public_administration_education_health_headcount_share"]
MODELS = ["historical_kmeans4", "sw_constrained_huber75", "sw_constrained_ordinary"]
SEED = 20261003


def orthogonal_basis(matrix, reference_norms=None):
    """Numerical column space, retaining the scale before residualization.

    Residual additions must be compared with their original column norms. If
    each projected roundoff column were renormalized to unit length, an exactly
    confounded predictor could acquire a false added rank and partial R².
    """
    matrix = np.asarray(matrix, float)
    if not np.isfinite(matrix).all():
        raise ValueError("Finite design required")
    scales = np.linalg.norm(matrix, axis=0) if reference_norms is None else np.asarray(reference_norms, float).copy()
    if scales.shape != (matrix.shape[1],) or not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("Finite nonnegative original column norms required")
    scales[scales == 0] = 1
    u, singular, _ = np.linalg.svd(matrix / scales, full_matrices=False)
    largest = singular[0] if len(singular) else 0
    # With references, a tiny *entire* residual design is unidentifiable relative
    # to its original predictors, even when its own leading singular value is >0.
    threshold = max(1., largest) * 1e-11 if reference_norms is not None else largest * 1e-11
    return u[:, singular > threshold]


def within_region(matrix, regions):
    matrix = np.asarray(matrix, float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    output = matrix.copy()
    for region in np.unique(regions):
        mask = regions == region
        anchored = output[mask] - output[mask][0]
        output[mask] = anchored - anchored.mean(axis=0)
    return output


def residualized_effect(y, controls, additions, regions):
    yy = within_region(y, regions)[:, 0]
    c = within_region(controls, regions)
    z = within_region(additions, regions)
    original_addition_norms = np.linalg.norm(z, axis=0)
    q = orthogonal_basis(c)
    yy = yy - q @ (q.T @ yy)
    z = z - q @ (q.T @ z)
    tss = float(yy @ yy)
    if tss <= max(1e-300, float(y @ y)) * 1e-24:
        return None, yy, z, q
    qz = orthogonal_basis(z, reference_norms=original_addition_norms)
    return float(np.clip(np.sum((qz.T @ yy) ** 2) / tss, 0, 1)), yy, z, q


def permutation_test(y, controls, labels, regions, draws=999, seed=SEED):
    z = np.eye(int(np.max(labels)) + 1)[labels]
    observed, yy, _, q = residualized_effect(y, controls, z, regions)
    if observed is None:
        return {"partial_r_squared": None, "permutation_p": None, "valid_draws": 0}
    tss = float(yy @ yy)
    original_addition_norms = np.linalg.norm(within_region(z, regions), axis=0)
    strata = [np.flatnonzero(regions == r) for r in np.unique(regions)]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        permuted = labels.copy()
        for ix in strata:
            permuted[ix] = rng.permutation(labels[ix])
        zz = within_region(np.eye(z.shape[1])[permuted], regions)
        zz -= q @ (q.T @ zz)
        qz = orthogonal_basis(zz, reference_norms=original_addition_norms)
        value = float(np.sum((qz.T @ yy) ** 2) / tss)
        exceed += value >= observed - 1e-12
    return {"partial_r_squared": observed, "permutation_p": (exceed + 1) / (draws + 1),
            "permutation_scheme": "labels permuted within region; both sides re-residualized on all controls",
            "permutation_interpretation": "Exploratory random-label reference, not a rigorous general conditional-null test: labels may remain associated with continuous controls. Holm adjustment does not fix that null mismatch.",
            "valid_draws": draws, "seed": seed}


def fit_predict(train_x, train_y, test_x):
    train_x = np.asarray(train_x, float)
    train_y = np.asarray(train_y, float)
    test_x = np.asarray(test_x, float)
    if len(train_x) != len(train_y) or not np.isfinite(train_x).all() or not np.isfinite(train_y).all() or not np.isfinite(test_x).all():
        raise ValueError("Aligned finite training/test data required")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    active = scale > 1e-12
    design = np.column_stack([np.ones(len(train_x)), (train_x[:, active] - mean[active]) / scale[active]])
    target = np.column_stack([np.ones(len(test_x)), (test_x[:, active] - mean[active]) / scale[active]])
    beta = np.linalg.lstsq(design, train_y, rcond=1e-11)[0]
    return target @ beta


def blocked_predictions(y, designs, regions):
    regions = np.asarray(regions)
    if len(np.unique(regions)) < 3:
        raise ValueError("At least three regions required")
    predictions = {name: np.full(len(y), np.nan) for name in designs}
    for held_out in np.unique(regions):
        test = regions == held_out
        train = ~test
        for name, x in designs.items():
            predictions[name][test] = fit_predict(x[train], y[train], x[test])
    if not all(np.isfinite(x).all() for x in predictions.values()):
        raise ValueError("Incomplete held-out predictions")
    return predictions


def paired_scores(y, predictions, regions, bootstrap=1999, seed=SEED):
    rows = []
    baseline_errors = (y - predictions["controls"]) ** 2
    baseline_by_region = np.array([baseline_errors[regions == r].mean() for r in np.unique(regions)])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(baseline_by_region), size=(bootstrap, len(baseline_by_region)))
    for name, pred in predictions.items():
        sq = (y - pred) ** 2
        region_mse = np.array([sq[regions == r].mean() for r in np.unique(regions)])
        improvement = baseline_by_region - region_mse
        interval = np.quantile(improvement[samples].mean(axis=1), [.025, .975]).tolist()
        rows.append({"predictor": name, "municipality_rmse": float(np.sqrt(sq.mean())),
                     "municipality_mae": float(np.abs(y - pred).mean()),
                     "equal_region_mse": float(region_mse.mean()),
                     "equal_region_mse_improvement_vs_controls": float(improvement.mean()),
                     "paired_region_bootstrap_improvement_95": interval,
                     "region_bootstrap_draws": bootstrap})
        for comparator in ["controls_plus_historical_kmeans4", "controls_plus_continuous5"]:
            if comparator not in predictions:
                continue
            reference_sq = (y - predictions[comparator]) ** 2
            reference_regions = np.array([reference_sq[regions == r].mean() for r in np.unique(regions)])
            gain = reference_regions - region_mse
            rows[-1][f"equal_region_mse_improvement_vs_{comparator}"] = float(gain.mean())
            rows[-1][f"paired_region_bootstrap_improvement_vs_{comparator}_95"] = np.quantile(gain[samples].mean(axis=1), [.025, .975]).tolist()
    return rows


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    return adjusted.tolist()


def panel_controls(panel_path):
    panel = pd.read_csv(panel_path)
    panel = panel.loc[panel.period.str.startswith("2023-")].copy()
    values = panel[CATEGORIES].to_numpy(float)
    totals = panel[TOTAL].to_numpy(float)
    if (values <= 0).any() or (totals <= 0).any() or not np.isfinite(values).all() or not np.isfinite(totals).all():
        raise ValueError("Positive finite category estimates required")
    log_categories = np.log(values)
    common = (log_categories - np.log(totals)[:, None]).mean(axis=1)
    contrasts = log_categories - log_categories.mean(axis=1)[:, None]
    extra = pd.DataFrame({"entity_id": panel.entity_id, "expense_2023": totals, "shared_ratio_level": common})
    for j in range(5):
        extra[f"contrast_{j}"] = contrasts[:, j]
    # Project before monthly median: cancellation of total is exact per month.
    return extra.groupby("entity_id", sort=False).median()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/external-national-2026-10-03")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=1999)
    args = parser.parse_args()
    if args.permutations < 99 or args.bootstrap < 499:
        raise ValueError("At least99 permutations and499 region bootstrap draws required")
    frozen = json.loads((args.report / "frozen-labels.json").read_text(encoding="utf-8"))
    ids = frozen["ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frozen ID")
    features = np.load(args.report / "consumer-features.npy", allow_pickle=False)
    if features.shape != (len(ids), 5) or not np.isfinite(features).all():
        raise ValueError("Expected aligned five frozen consumer features")
    for model in MODELS:
        labels = np.asarray(frozen["labels"][model])
        if labels.shape != (len(ids),) or not np.isin(labels, [0, 1, 2, 3]).all():
            raise ValueError("Expected aligned fixed K4 labels")
    extra = panel_controls(ROOT / "data/processed/panel.csv")
    all_effects, all_scores, excluded, predictions_output = [], [], [], []
    for year in [2023, 2024]:
        frame = pd.read_csv(args.report / f"cohort-{year}.csv").set_index("entity_id").reindex(ids).join(extra)
        if frame.region_code.isna().any():
            raise ValueError("Missing cohort identity")
        frame["log_population"] = np.log(frame.population_total.where(frame.population_total > 0))
        frame["log_expense"] = np.log(frame.expense_2023)
        type_columns = pd.get_dummies(frame.municipal_district_type, dtype=float).to_numpy()
        common_controls = np.column_stack([type_columns, frame.log_population, frame.log_expense])
        shared = frame.shared_ratio_level.to_numpy(float)[:, None]
        contrasts = frame[[f"contrast_{j}" for j in range(5)]].to_numpy(float)
        regions = frame.region_code.to_numpy(int)
        for outcome in OUTCOMES:
            y_all = frame[outcome].to_numpy(float)
            mask = np.isfinite(y_all) & np.isfinite(common_controls).all(axis=1) & np.isfinite(shared).all(axis=1) & np.isfinite(contrasts).all(axis=1)
            y, controls, rr = y_all[mask], common_controls[mask], regions[mask]
            excluded.append({"year": year, "outcome": outcome, "source_outcome_n": int(np.isfinite(y_all).sum()),
                             "complete_controls_n": len(y), "regions": len(np.unique(rr)),
                             "complete_controls_counts_by_cluster": pd.Series(np.asarray(frozen["labels"][MODELS[0]])[mask]).value_counts().sort_index().to_dict()})
            if len(y) < 50 or len(np.unique(rr)) < 10:
                raise ValueError("Insufficient verified external cohort")
            designs = {"controls": controls,
                       "controls_plus_continuous5": np.column_stack([controls, features[mask]]),
                       "controls_plus_shared_ratio_level": np.column_stack([controls, shared[mask]]),
                       "controls_plus_denominator_free_contrasts": np.column_stack([controls, contrasts[mask]])}
            for model in MODELS:
                labels = np.asarray(frozen["labels"][model], int)[mask]
                additions = np.eye(4)[labels]
                designs[f"controls_plus_{model}"] = np.column_stack([controls, additions])
                result = permutation_test(y, controls, labels, rr, args.permutations)
                shared_effect = residualized_effect(y, np.column_stack([controls, shared[mask]]), additions, rr)[0]
                contrast_effect = residualized_effect(y, np.column_stack([controls, contrasts[mask]]), additions, rr)[0]
                effect = {"year": year, "outcome": outcome, "model": model, "n": len(y), "regions": len(np.unique(rr)),
                          **result, "partial_r2_after_shared_ratio_level_control": shared_effect,
                          "partial_r2_after_denominator_free_contrasts_control": contrast_effect,
                          "sensitivity": {}}
                for name, condition in {"exclude_intracity": ~frame.intracity_moscow_petersburg.to_numpy(bool)[mask],
                                         "exclude_source_boundary_history": ~frame.source_boundary_history_flag.to_numpy(bool)[mask],
                                         "exclude_registry_version_changes": ~frame.registry_changed_2023_to_year.to_numpy(bool)[mask]}.items():
                    n = int(condition.sum())
                    effect["sensitivity"][name] = {"n": n, "partial_r_squared": residualized_effect(y[condition], controls[condition], additions[condition], rr[condition])[0] if n >= 30 else None}
                all_effects.append(effect)
            predictions = blocked_predictions(y, designs, rr)
            all_scores.extend({"year": year, "outcome": outcome, "n": len(y), "regions": len(np.unique(rr)), **row}
                              for row in paired_scores(y, predictions, rr, args.bootstrap))
            pred_frame = pd.DataFrame({"entity_id": np.asarray(ids)[mask], "year": year, "outcome": outcome,
                                       "region_code": rr, "observed": y, **predictions})
            predictions_output.append(pred_frame)
            print(json.dumps({"year": year, "outcome": outcome, "n": len(y), "regions": len(np.unique(rr)),
                              "status": "complete"}), flush=True)
    for year in [2023, 2024]:
        family = [effect for effect in all_effects if effect["year"] == year]
        adjusted = holm([effect["permutation_p"] for effect in family])
        for row, value in zip(family, adjusted):
            row["holm_p_year_5outcomes_3models"] = value
    pd.concat(predictions_output, ignore_index=True).to_csv(args.report / "blocked-predictions.csv", index=False)
    pd.DataFrame(all_scores).to_csv(args.report / "blocked-scores.csv", index=False)
    report = {"status": "completed_exploratory_independent_external_validation", "outcomes": OUTCOMES, "models": MODELS,
              "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [args.report / "analysis-spec.json", args.report / "analysis-spec-addendum.json", args.report / "frozen-labels.json", args.report / "consumer-features.npy", args.report / "cohort-2023.csv", args.report / "cohort-2024.csv", ROOT / "data/processed/panel.csv"]},
              "controls": "Within-region: region FE, municipal type,logresidentpopulation,logmedianannual2023totalexpense. Blocked: same observedcovariates without region FE.",
              "held_out_region_handling": "No held-out region outcomes used for fitting/scaling/intercept; unseenregion FE never estimated. Frozen unsupervised labels/features are transductive to expensepanel.",
              "denominator_audit": "Monthly commonratiolevel=mean_jlog(category_j/total). Monthly contrasts=logcategory_j-mean_jlogcategory; total cancels exactly. Projection occurs BEFORE coordinatewiseannualmedian.",
              "limits": "Annual retrospective associations only; no industrialtyping, causalclaim orfutureforecast. Organizationheadcount excludes smallbusiness; jobs/resident is notresidentemploymentrate. Economicoutcomes didnotchoose clusters. Random-label permutation p-values are exploratory references, not rigorous conditional-null p-values given correlated continuouscontrols; Holm adjustment cannotrepairthatnull. Regionbootstrap errors are descriptive; unmeasuredconfounding/spatialdependence mayremain.",
              "cohorts": excluded, "within_region_associations": all_effects, "blocked_scores": all_scores}
    (args.report / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
