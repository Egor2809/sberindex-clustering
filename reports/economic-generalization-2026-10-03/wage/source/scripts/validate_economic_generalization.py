"""Fixed architecture region-inductive wage validation and exploratory2024 forecast.

Representation folds are resumable only when protocol/input/source hashes agree.
Existing national reports and raw data are never modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
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
from sbercluster.economic_generalization import (BASELINE_SPEC, H75_SPEC, HISTOGRAM_SPEC,
        CLUSTER_SEED, PROBE_SEED, fit_region_representation, type_design,
        fit_predict_probe, score_predictions, stable_hash)
from sbercluster.io import CATEGORIES, TOTAL

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/external-national-2026-10-03"
OUTCOME = "log_annual_monthly_wage_rubles"
CODE = ["sbercluster/economic_generalization.py", "scripts/validate_economic_generalization.py",
        "sbercluster/selection_frontier.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_data():
    frozen = json.loads((SOURCE / "frozen-labels.json").read_text(encoding="utf-8"))
    ids = frozen["ids"]
    cohort = pd.read_csv(SOURCE / "cohort-2023.csv").set_index("entity_id").reindex(ids)
    later = pd.read_csv(SOURCE / "cohort-2024.csv").set_index("entity_id").reindex(ids)
    panel = pd.read_csv(ROOT / "data/processed/panel.csv")
    panel = panel.loc[panel.period.str.startswith("2023-") & panel.entity_id.isin(ids)].copy()
    if panel.duplicated(["entity_id", "period"]).any() or len(panel) != 12 * len(ids):
        raise ValueError("Exactly twelve distinct2023 records per frozen expense entity required")
    months = sorted(panel.period.unique())
    if len(months) != 12:
        raise ValueError("Twelve2023 months required")
    tensor = []
    totals = []
    for month in months:
        block = panel.loc[panel.period == month].set_index("entity_id").reindex(ids)
        values, total = block[CATEGORIES].to_numpy(float), block[TOTAL].to_numpy(float)
        if not np.isfinite(values).all() or not np.isfinite(total).all() or (values <= 0).any() or (total <= 0).any():
            raise ValueError("Positive finite raw expense observations required")
        if not np.array_equal(block.region_code.to_numpy(int), cohort.region_code.to_numpy(int)):
            raise ValueError("Panel/cohort region identity mismatch")
        tensor.append(np.log(values / total[:, None]))
        totals.append(total)
    monthly = np.asarray(tensor)
    numeric = np.column_stack([np.log(cohort.population_total.where(cohort.population_total > 0)),
                               np.log(np.median(np.asarray(totals), axis=0))])
    types = cohort.municipal_district_type.fillna("__missing__").to_numpy(str)
    controls_ok = np.isfinite(numeric).all(axis=1) & (types != "__missing__")
    return {"ids": np.asarray(ids), "monthly": monthly, "regions": cohort.region_code.to_numpy(int),
            "numeric": numeric, "types": types, "controls_ok": controls_ok,
            "y2023": cohort[OUTCOME].to_numpy(float), "y2024": later[OUTCOME].to_numpy(float)}


def protocol(include_h75):
    inputs = [ROOT / "data/processed/panel.csv", SOURCE / "frozen-labels.json",
              SOURCE / "cohort-2023.csv", SOURCE / "cohort-2024.csv"]
    return {"schema": 1, "fixed_before_new_inductive_fits": True, "primary_outcome": OUTCOME,
            "clustering": {"baseline": BASELINE_SPEC, "huber75": H75_SPEC if include_h75 else "not included", "seed": CLUSTER_SEED},
            "learner_kinds": ["linear", "quadratic", "histogram"], "histogram": HISTOGRAM_SPEC,
            "controls": "municipal type(training-only vocabulary),log2023population,logmedian2023totalexpenses",
            "quadratic": "all degree2 main/interaction terms for continuous numeric predictors; cluster/type dummies additive",
            "feature_calibration": "median/IQR over2023monthlyX of all OTHER-region expense entities, including missing outcomes",
            "clustering_initialization": "KMeans20starts seed1729 on other regions only; H75 init just-fitted other-region KMeans",
            "selection": "Fixed previouslyselected architectures; no economic outcome selection/tuning. No national fitted labels used.",
            "historical_architecture_caveat": "The architecture itself was chosen after inspecting national2023X; heldoutregion influences that historical architecture choice. New fits are inductive conditional on fixed architecture, not full selection-process independence.",
            "cohort": "Frozen2016complete2023-24expense-panel entities, retrospectively selected; nonrandom external outcome missingness",
            "forecast2024": "Every economic learner fits2023other-region wages; apply SAME2023covariates/features/labels to predict2024heldoutregion wage. No2024X or2024y calibration/refitting. 2024was inspected previously; exploratory forecast backtest.",
            "metrics": "same paired cohort eachlearner; municipalityRMSE, equalregionMSE, equalregion withinregion centerederror samplevariance (evaluation only)",
            "uncertainty": "1999 paired region bootstrap conditional on fixed OOF predictions; dependent overlapping fits/spatial regions mean no generic inferential95coverage guarantee; no conditionalpvalue",
            "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "source_sha256": {str(p): digest(ROOT / p) for p in CODE}}


def evaluate_fold(data, fitted, held_out, include_h75):
    train = (data["regions"] != held_out) & data["controls_ok"] & np.isfinite(data["y2023"])
    test = (data["regions"] == held_out) & data["controls_ok"]
    if train.sum() < 50 or not test.any():
        raise ValueError("Insufficient fold wage fitting/prediction controls")
    train_types, test_types, vocabulary = type_design(data["types"][train], data["types"][test])
    base_train = np.column_stack([train_types, data["numeric"][train]])
    base_test = np.column_stack([test_types, data["numeric"][test]])
    numeric = np.arange(train_types.shape[1], base_train.shape[1])
    variants = {"controls": (base_train, base_test, numeric),
                "continuous5": (np.column_stack([base_train, fitted["profiles"][train]]),
                                 np.column_stack([base_test, fitted["profiles"][test]]),
                                 np.arange(train_types.shape[1], base_train.shape[1] + 5))}
    for name in ["kmeans4", "huber75"] if include_h75 else ["kmeans4"]:
        labels = fitted["labels"][name]
        variants[name] = (np.column_stack([base_train, np.eye(4)[labels[train]]]),
                          np.column_stack([base_test, np.eye(4)[labels[test]]]), numeric)
    frame = pd.DataFrame({"entity_id": data["ids"][test], "held_out_region": held_out,
                          "observed2023": data["y2023"][test], "observed2024": data["y2024"][test]})
    artifacts = {"held_out_region": int(held_out), "training_outcome_year": 2023,
                 "training_outcome_entities": data["ids"][train].tolist(), "type_vocabulary": vocabulary,
                 "representation_sha256": fitted["artifact_sha256"], "probes": {}}
    for kind in ["linear", "quadratic", "histogram"]:
        for name, (xx, target, numeric_columns) in variants.items():
            pred, meta = fit_predict_probe(xx, data["y2023"][train], target, kind, numeric_columns)
            frame[f"{kind}_{name}"] = pred
            artifacts["probes"][f"{kind}_{name}"] = meta
    return frame, artifacts


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = protocol(args.include_h75)
    spec_hash = stable_hash(spec)
    spec_path = output / "protocol.json"
    if spec_path.exists():
        if stable_hash(json.loads(spec_path.read_text(encoding="utf-8"))) != spec_hash:
            raise ValueError("Resume rejected: protocol/input/source snapshot changed; use new output")
    else:
        save_json(spec_path, spec)
        for name in CODE:
            dst = output / "source" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((ROOT / name).read_bytes())
    started = time.perf_counter()
    data = load_data()
    regions = np.unique(data["regions"][data["controls_ok"] &
                        (np.isfinite(data["y2023"]) | np.isfinite(data["y2024"]))])
    newly_fitted = 0
    output_frames = []
    for region in regions:
        folder = output / "folds" / str(int(region))
        folder.mkdir(parents=True, exist_ok=True)
        representation_path = folder / "representation.json"
        if representation_path.exists():
            artifact = json.loads(representation_path.read_text(encoding="utf-8"))
            if artifact["protocol_sha256"] != spec_hash:
                raise ValueError("Cached fold protocol mismatch")
            original = {k: v for k, v in artifact.items() if k not in
                        ["artifact_sha256", "protocol_sha256", "elapsed_seconds", "cache_sha256"]}
            if stable_hash(original) != artifact["artifact_sha256"]:
                raise ValueError("Cached fitted artifact digest mismatch")
            with np.load(folder / "profiles-and-labels.npz", allow_pickle=False) as cache:
                fitted = {"profiles": cache["profiles"], "labels": {name: cache[name] for name in artifact["models"]},
                          "artifact_sha256": artifact["artifact_sha256"]}
        else:
            if args.max_new_folds and newly_fitted >= args.max_new_folds:
                break
            fold_start = time.perf_counter()
            fitted = fit_region_representation(data["monthly"], data["regions"], region,
                                                data["ids"], args.include_h75)
            np.savez_compressed(folder / "profiles-and-labels.npz", profiles=fitted["profiles"], **fitted["labels"])
            artifact = {**fitted["artifact"], "artifact_sha256": fitted["artifact_sha256"],
                        "protocol_sha256": spec_hash, "elapsed_seconds": time.perf_counter() - fold_start,
                        "cache_sha256": digest(folder / "profiles-and-labels.npz")}
            save_json(representation_path, artifact)
            newly_fitted += 1
            print(json.dumps({"event": "new_representation_fold", "region": int(region),
                              "seconds": artifact["elapsed_seconds"], "train_n": len(fitted["train_indices"])}), flush=True)
        if digest(folder / "profiles-and-labels.npz") != artifact["cache_sha256"]:
            raise ValueError("Cached arrays digest mismatch")
        # Source/inputidentity guards the cache; stored IDs independently guard fold membership.
        if set(artifact["training_entity_ids"]) != set(data["ids"][data["regions"] != region]):
            raise ValueError("Cached representation fit entity membership mismatch")
        if args.stage == "representations":
            continue
        frame_path = folder / "predictions.csv"
        if frame_path.exists():
            frame = pd.read_csv(frame_path)
            probe = json.loads((folder / "probes.json").read_text(encoding="utf-8"))
            if probe["representation_sha256"] != fitted["artifact_sha256"] or probe["protocol_sha256"] != spec_hash:
                raise ValueError("Cached economic probe identity mismatch")
            if digest(frame_path) != probe["predictions_sha256"]:
                raise ValueError("Cached economic predictions digest mismatch")
        else:
            frame, probes = evaluate_fold(data, fitted, region, args.include_h75)
            frame.to_csv(frame_path, index=False)
            save_json(folder / "probes.json", {**probes, "protocol_sha256": spec_hash, "predictions_sha256": digest(frame_path)})
        output_frames.append(frame)
    existing = [int(r) for r in regions if (output / "folds" / str(int(r)) / "representation.json").exists()]
    complete = len(existing) == len(regions)
    save_json(output / "status.json", {"status": "representations_complete" if complete else "partial_representations",
              "stage": args.stage, "protocol_sha256": spec_hash, "fitted_region_count": len(existing),
              "expected_region_count": len(regions), "newly_fitted_this_run": newly_fitted,
              "elapsed_seconds": time.perf_counter() - started, "completed_at": datetime.now(timezone.utc).isoformat()})
    if args.stage == "representations" or not complete:
        return
    frame = pd.concat(output_frames, ignore_index=True)
    if frame.entity_id.duplicated().any():
        raise ValueError("Duplicated OOF prediction entity")
    prediction_columns = [name for name in frame if name not in ["entity_id", "held_out_region", "observed2023", "observed2024"]]
    results = {"protocol_sha256": spec_hash, "status": "completed_exploratory_strict_region_inductive_fixed_architecture",
               "cohorts": [], "scores": [], "limits": [spec[name] for name in ["historical_architecture_caveat", "cohort", "forecast2024", "uncertainty"]],
               "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]}}
    for year in [2023, 2024]:
        mask = np.isfinite(frame[f"observed{year}"].to_numpy(float))
        part = frame.loc[mask]
        y = part[f"observed{year}"].to_numpy(float)
        rr = part.held_out_region.to_numpy(int)
        predictions = {name: part[name].to_numpy(float) for name in prediction_columns}
        scores = score_predictions(y, predictions, rr)
        results["cohorts"].append({"outcome_year": year, "training_outcome_year": 2023, "n": len(y), "regions": len(np.unique(rr)),
                                  "prediction_inputs_year": 2023, "fit_excludes_scored_region": True})
        results["scores"].extend({"outcome_year": year, "training_outcome_year": 2023, **row} for row in scores)
    frame.to_csv(output / "all-oof-predictions.csv", index=False)
    save_json(output / "results.json", results)
    pd.DataFrame([{k: v for k, v in row.items() if k != "comparisons"} for row in results["scores"]]).to_csv(output / "scores.csv", index=False)
    save_json(output / "status.json", {"status": "completed", "protocol_sha256": spec_hash,
              "fitted_region_count": len(regions), "prediction_entities": len(frame), "elapsed_seconds": time.perf_counter() - started,
              "results_sha256": digest(output / "results.json"), "predictions_sha256": digest(output / "all-oof-predictions.csv")})
    print(json.dumps({"event": "completed", "cohorts": results["cohorts"], "seconds": time.perf_counter() - started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["representations", "all"], default="all")
    parser.add_argument("--include-h75", action="store_true")
    parser.add_argument("--max-new-folds", type=int, default=0, help="0=all finite folds, 1=affordability benchmark with resume")
    args = parser.parse_args()
    if args.max_new_folds < 0:
        raise ValueError("Nonnegative fold limit required")
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except ImportError:
        raise RuntimeError("psutil required to enforce BelowNormal process priority")
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
