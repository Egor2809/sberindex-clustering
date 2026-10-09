"""Exploratory, fixed quadratic confounder and representation challenge.

The preregistered linear national report is kept separate. All five outcomes,
both years and all three preselected partitions receive the same checks.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from scripts.download_external_national import ROOT, sha256
from scripts.validate_external_national import (MODELS, OUTCOMES, SEED, blocked_predictions,
                                               paired_scores, panel_controls, residualized_effect)


def quadratic(matrix):
    matrix = np.asarray(matrix, float)
    terms = [matrix[:, j] for j in range(matrix.shape[1])]
    terms.extend(matrix[:, i] * matrix[:, j] for i, j in itertools.combinations_with_replacement(range(matrix.shape[1]), 2))
    return np.column_stack(terms)


def pairwise_region_loss(y, predictions, regions):
    """Equivalent mean pairwise error-difference loss, computed AFTER prediction.

    A held-out region's observed outcomes never enter fitting or prediction.
    Centered evaluation errors remove an additive regional prediction offset.
    """
    rows = []
    for name, prediction in predictions.items():
        errors = y - prediction
        variances = []
        for region in np.unique(regions):
            residual = errors[regions == region]
            if len(residual) < 2:
                continue
            variances.append(float(2 * np.var(residual, ddof=1)))
        rows.append({"predictor": name, "mean_within_region_pairwise_squared_error_difference": float(np.mean(variances)),
                     "regions_with_at_least_two_observations": len(variances)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/external-national-2026-10-03")
    args = parser.parse_args()
    labels = json.loads((args.report / "frozen-labels.json").read_text(encoding="utf-8"))
    ids = labels["ids"]
    features = np.load(args.report / "consumer-features.npy", allow_pickle=False)
    extra = panel_controls(ROOT / "data/processed/panel.csv")
    rows, associations, pairwise, population_sensitivity = [], [], [], []
    population_disagreement_ids = {"tid_1845", "tid_1846", "tid_1848"}
    for year in [2023, 2024]:
        frame = pd.read_csv(args.report / f"cohort-{year}.csv").set_index("entity_id").reindex(ids).join(extra)
        types = pd.get_dummies(frame.municipal_district_type, dtype=float).to_numpy()
        # Fixed centering constants improve conditioning; no outcomes determine them.
        numeric = np.column_stack([np.log(frame.population_total.where(frame.population_total > 0)) - 10,
                                   np.log(frame.expense_2023) - 10, frame.shared_ratio_level])
        qcontrols = np.column_stack([types, quadratic(numeric)])
        regions = frame.region_code.to_numpy(int)
        for outcome in OUTCOMES:
            y = frame[outcome].to_numpy(float)
            mask = np.isfinite(y) & np.isfinite(qcontrols).all(axis=1)
            yy, cc, rr = y[mask], qcontrols[mask], regions[mask]
            designs = {"controls": cc, "controls_plus_continuous5": np.column_stack([cc, features[mask]]),
                       "controls_plus_quadratic_continuous5": np.column_stack([cc, quadratic(features[mask])])}
            for model in MODELS:
                addition = np.eye(4)[np.asarray(labels["labels"][model], int)[mask]]
                designs[f"controls_plus_{model}"] = np.column_stack([cc, addition])
                associations.append({"year": year, "outcome": outcome, "model": model, "n": len(yy),
                                     "regions": len(np.unique(rr)),
                                     "partial_r2_after_quadratic_population_expense_shared_factor_controls": residualized_effect(yy, cc, addition, rr)[0]})
            predictions = blocked_predictions(yy, designs, rr)
            rows.extend({"year": year, "outcome": outcome, "n": len(yy), "regions": len(np.unique(rr)), **row}
                        for row in paired_scores(yy, predictions, rr, bootstrap=1999))
            pairwise.extend({"year": year, "outcome": outcome, **row} for row in pairwise_region_loss(yy, predictions, rr))
            if outcome == "log_annual_monthly_wage_rubles":
                included = mask & ~np.isin(ids, list(population_disagreement_ids))
                y_s, r_s = y[included], regions[included]
                linear_controls = np.column_stack([types, numeric[:, :2]])[included]
                sensitivity_designs = {"controls": linear_controls,
                                       "controls_plus_continuous5": np.column_stack([linear_controls, features[included]])}
                effects = []
                for model in MODELS:
                    addition = np.eye(4)[np.asarray(labels["labels"][model], int)[included]]
                    sensitivity_designs[f"controls_plus_{model}"] = np.column_stack([linear_controls, addition])
                    effects.append({"model": model, "partial_r_squared": residualized_effect(y_s, linear_controls, addition, r_s)[0]})
                sensitivity_predictions = blocked_predictions(y_s, sensitivity_designs, r_s)
                population_sensitivity.append({"year": year, "outcome": outcome, "n": len(y_s),
                                               "excluded_ids": sorted(population_disagreement_ids),
                                               "controls": "Original linear type/logpopulation/logexpense controls; within-association also regionFE",
                                               "within_region_associations": effects,
                                               "blocked_scores": paired_scores(y_s, sensitivity_predictions, r_s, bootstrap=1999)})
            print(json.dumps({"year": year, "outcome": outcome, "status": "quadratic_challenge_complete"}), flush=True)
    report = {"status": "exploratory_robustness_completed", "selection_rule": "Fixed complete second-order polynomial(population,expense,sharedratiofactor); all main outcomes/years/models retained. Economic targets did not choose cluster models or terms.",
              "timing": "Requested after the initial linear outcomes were inspected; this challenge is explicitly post hoc, separate from the preregistered linear analysis.",
              "controls": "Municipal type and degree2 monomials of logpopulation,logexpense,sharedratiofactor including interactions. Within-association also regionFE; blockedprobes use otherregionoutcomes only.",
              "strong_representation_comparator": "Frozen continuous5 consumerfeatures; also their complete quadratic basis. No outcome-selected regularization/hyperparameters.",
              "pairwise_loss": "After all OOF predictions are fixed, evaluate 2*samplevariance of errors within each region (equivalent mean pairwise squared error difference). This uses heldoutoutcomes only for evaluation, never prediction or fitting.",
              "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [args.report / "results.json", args.report / "frozen-labels.json", args.report / "consumer-features.npy", args.report / "cohort-2023.csv", args.report / "cohort-2024.csv"]},
              "within_region_associations": associations, "blocked_scores": rows, "within_region_pairwise_loss": pairwise,
              "population_crossindicator_disagreement_exclusion_sensitivity": population_sensitivity}
    (args.report / "nonlinear-robustness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.report / "nonlinear-blocked-scores.csv", index=False)


if __name__ == "__main__":
    main()
