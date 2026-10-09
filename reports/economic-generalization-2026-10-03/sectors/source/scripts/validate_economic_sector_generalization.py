"""Predeclared three-sector transfer with full eligible-cohort missingness bounds."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.validate_economic_generalization import ROOT, SOURCE, load_data, digest, save_json, set_runtime_priority
from sbercluster.economic_generalization import fit_region_representation, fit_predict_probe, type_design, stable_hash, HISTOGRAM_SPEC
from sbercluster.economic_sector_bounds import squared_loss_difference_bounds

OUTCOMES = ["manufacturing_headcount_share", "agriculture_headcount_share",
            "public_administration_education_health_headcount_share"]
REPRESENTATIONS = ["controls", "continuous5", "kmeans4", "huber75"]
CODE = ["scripts/validate_economic_sector_generalization.py", "sbercluster/economic_sector_bounds.py",
        "scripts/validate_economic_generalization.py", "sbercluster/economic_generalization.py", "sbercluster/selection_frontier.py"]


def run(args):
    output, expense_folds = args.output.resolve(), args.expense_folds.resolve()
    if output.exists():
        raise FileExistsError("Use a new sector output directory")
    output.mkdir(parents=True)
    original_protocol = json.loads((expense_folds / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in original_protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Original validated expense/cohort input snapshot changed")
    data = load_data()
    # Predetermine the denominator using only X/control availability, never any sector Y.
    shared = np.median(data["monthly"].mean(axis=2), axis=0)
    eligible = data["controls_ok"] & np.isfinite(shared)
    eligible_regions = np.unique(data["regions"][eligible])
    spec = {"status": "fixed_before_sector_prediction_fits", "outcomes": OUTCOMES,
        "outcome_year": 2023, "learner": "histogram", "learner_spec": HISTOGRAM_SPEC,
        "representation_variants": REPRESENTATIONS, "selection": "Predeclared nonlinear stress-test family from wage protocol, all four variants and three preexisting sector endpoints; parameters reused without sector tuning or architecture selection",
        "controls": "training-only type indicators,logpopulation2023,logmedianmonthlyTOTAL2023,sharedaxis=median_month(mean_jlog(category_j/total))",
        "eligibility": "frozen2016expenseentities with finite2023controls and sharedaxis; independent of outcome observation",
        "eligible_entity_ids": data["ids"][eligible].tolist(), "eligible_regions": eligible_regions.astype(int).tolist(),
        "prediction_range": "Every model prediction clipped[0,1]; observedtargets never clipped; invalidobservedshare raiseserror",
        "training": "fit only finite legal2023sector targets in other regions; reuse validated regionexcluded2023expense calibration/prototypes, retainingoutcomemissingX",
        "missingness_target": "Separate full fixedeligiblecohort for eachtarget; knownlosses exact, unknownshares independentlyany[0,1]; sharpendpointbounds, equalregion primary/municipalitysecondary; noMAR assumption or samplingCI",
        "representation_protocol_sha256": stable_hash(original_protocol),
        "representation_original_source_sha256": original_protocol["source_sha256"],
        "input_sha256": original_protocol["input_sha256"],
        "source_sha256": {name: digest(ROOT / name) for name in CODE},
        "limitations": ["Sectorheadcount excludes smallbusiness and is workplace, not productionoutput or residentemployment",
             "Both2023/2024data previously inspected; fixedarchitectures historically selected on nationalX",
             "Eligibility conditions on2023control availability and retrospective complete2023-24expensecohort",
             "Separate coordinatewise bounds ignore jointsectorconstraints; no sectorlabels assigned from test outcomes"]}
    save_json(output / "protocol.json", spec)
    protocol_sha = stable_hash(spec)
    for name in CODE:
        dst = output / "source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())
    # Sector Y is accessed only after the target cohort and complete protocol were saved.
    cohort = pd.read_csv(SOURCE / "cohort-2023.csv").set_index("entity_id").reindex(data["ids"])
    targets = {outcome: cohort[outcome].to_numpy(float) for outcome in OUTCOMES}
    for outcome, y in targets.items():
        if np.isinf(y).any() or not ((y[np.isfinite(y)] >= 0) & (y[np.isfinite(y)] <= 1)).all():
            raise ValueError(f"Invalid observed share {outcome}; observed data are not clipped")
    numeric = np.column_stack([data["numeric"], shared])
    output_frames, fold_artifacts = [], []
    extra_folds = []
    started = time.perf_counter()
    for region in eligible_regions:
        folder = expense_folds / "folds" / str(int(region))
        if not (folder / "representation.json").exists():
            fitted = fit_region_representation(data["monthly"], data["regions"], region, data["ids"], True)
            extra_folds.append(int(region))
            dst = output / "extra-expense-folds" / str(int(region))
            dst.mkdir(parents=True)
            save_json(dst / "representation.json", fitted["artifact"])
            np.savez_compressed(dst / "profiles-and-labels.npz", profiles=fitted["profiles"], **fitted["labels"])
            artifact = fitted["artifact"]
        else:
            artifact = json.loads((folder / "representation.json").read_text(encoding="utf-8"))
            fitted_original = {k: v for k, v in artifact.items() if k not in ["artifact_sha256", "protocol_sha256", "elapsed_seconds", "cache_sha256"]}
            if stable_hash(fitted_original) != artifact["artifact_sha256"] or artifact["protocol_sha256"] != spec["representation_protocol_sha256"]:
                raise ValueError("Expense fold source identity mismatch")
            if digest(folder / "profiles-and-labels.npz") != artifact["cache_sha256"]:
                raise ValueError("Expense fold cache bytes mismatch")
            with np.load(folder / "profiles-and-labels.npz", allow_pickle=False) as cached:
                fitted = {"profiles": cached["profiles"], "labels": {name: cached[name] for name in ["kmeans4", "huber75"]}}
        if set(artifact["training_entity_ids"]) != set(data["ids"][data["regions"] != region]):
            raise ValueError("Expense fold excludes wrong entity set")
        test = eligible & (data["regions"] == region)
        for outcome in OUTCOMES:
            y = targets[outcome]
            train = eligible & (data["regions"] != region) & np.isfinite(y)
            if train.sum() < 50:
                raise ValueError("Too few observed training sector targets")
            train_type, test_type, vocabulary = type_design(data["types"][train], data["types"][test])
            train_base, test_base = np.column_stack([train_type, numeric[train]]), np.column_stack([test_type, numeric[test]])
            variants = {"controls": (train_base, test_base),
                        "continuous5": (np.column_stack([train_base, fitted["profiles"][train]]), np.column_stack([test_base, fitted["profiles"][test]]))}
            for name in ["kmeans4", "huber75"]:
                labels = fitted["labels"][name]
                variants[name] = (np.column_stack([train_base, np.eye(4)[labels[train]]]), np.column_stack([test_base, np.eye(4)[labels[test]]]))
            frame = pd.DataFrame({"entity_id": data["ids"][test], "region_code": region, "outcome": outcome,
                                  "observed": y[test], "target_observed": np.isfinite(y[test])})
            detail = {"region_code": int(region), "outcome": outcome, "training_ids": data["ids"][train].tolist(),
                      "test_eligible_ids": data["ids"][test].tolist(), "type_vocabulary": vocabulary,
                      "expense_training_ids_sha256": stable_hash(artifact["training_entity_ids"]), "probes": {}, "training_cluster_profiles": {}}
            for name, (xx, target) in variants.items():
                pred, meta = fit_predict_probe(xx, y[train], target, "histogram")
                frame[name] = np.clip(pred, 0, 1)
                detail["probes"][name] = meta
            for name in ["kmeans4", "huber75"]:
                labels = fitted["labels"][name]
                detail["training_cluster_profiles"][name] = [{"label": c, "observed_training_n": int((train & (labels == c)).sum()),
                        "observed_training_regions": int(len(np.unique(data["regions"][train & (labels == c)]))),
                        "mean_share": float(np.mean(y[train & (labels == c)])) if (train & (labels == c)).any() else None} for c in range(4)]
            fold_artifacts.append(detail)
            output_frames.append(frame)
        print(json.dumps({"event": "sector_fold_complete", "region": int(region), "targets": 3, "eligible_test_n": int(test.sum())}), flush=True)
    frames = pd.concat(output_frames, ignore_index=True)
    frames.to_csv(output / "all-eligible-oof-predictions.csv", index=False)
    with (output / "fold-probe-metadata.jsonl").open("w", encoding="utf-8") as stream:
        for item in fold_artifacts:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    results = {"status": "completed_exploratory_sector_transfer_exact_missingness_bounds", "protocol_sha256": protocol_sha,
               "eligible_n_per_target": int(eligible.sum()), "eligible_regions": eligible_regions.astype(int).tolist(),
               "extra_expense_folds": extra_folds, "observed_scores": [], "full_cohort_comparisons": [],
               "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]}, "limits": spec["limitations"]}
    for outcome in OUTCOMES:
        part = frames.loc[frames.outcome == outcome]
        if set(part.entity_id) != set(spec["eligible_entity_ids"]) or part.entity_id.duplicated().any():
            raise ValueError("Sector target cohort does not match fixed eligibility denominator")
        y, regions = part.observed.to_numpy(float), part.region_code.to_numpy(int)
        known = np.isfinite(y)
        for name in REPRESENTATIONS:
            pred = part[name].to_numpy(float)
            equal_mse = float(np.mean([np.mean((y[known & (regions == r)] - pred[known & (regions == r)]) ** 2) for r in np.unique(regions[known])]))
            results["observed_scores"].append({"outcome": outcome, "predictor": name, "observed_n": int(known.sum()),
                "observed_regions": int(len(np.unique(regions[known]))), "observed_municipality_rmse": float(np.sqrt(np.mean((y[known] - pred[known]) ** 2))),
                "observed_equal_region_mse": equal_mse, "eligible_n": int(eligible.sum()), "eligible_regions": len(eligible_regions)})
            for reference in REPRESENTATIONS:
                bounds = squared_loss_difference_bounds(y, pred, part[reference].to_numpy(float), regions)
                results["full_cohort_comparisons"].append({"outcome": outcome, "candidate": name, "reference": reference, **bounds})
    save_json(output / "results.json", results)
    pd.DataFrame(results["observed_scores"]).to_csv(output / "observed-scores.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k not in ["per_region", "regions_with_no_observed_target"]} for x in results["full_cohort_comparisons"]]).to_csv(output / "full-cohort-bounds.csv", index=False)
    save_json(output / "status.json", {"status": "completed", "protocol_sha256": protocol_sha,
        "region_folds": len(eligible_regions), "targets": len(OUTCOMES), "models": len(REPRESENTATIONS),
        "extra_expense_folds": extra_folds, "elapsed_seconds": time.perf_counter() - started,
        "results_sha256": digest(output / "results.json"), "predictions_sha256": digest(output / "all-eligible-oof-predictions.csv")})
    print(json.dumps({"event": "completed", "eligible_n": int(eligible.sum()), "folds": len(eligible_regions), "extra_folds": extra_folds,
                      "elapsed_seconds": time.perf_counter() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expense-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_runtime_priority()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
