"""Strict region-held-out, fixed-architecture economic representation probes.

Exploratory: 2023 and 2024 were inspected before this protocol. There is no
claim of a conditional permutation null, causal effect or production type.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sbercluster.selection_frontier import fit_frontier, predict_frontier

CLUSTER_SEED = 1729
PROBE_SEED = 20261003
BASELINE_SPEC = {"id": "historical_architecture_kmeans4", "k": 4, "min_fraction": .03}
H75_SPEC = {"id": "fixed_architecture_huber75", "k": 4, "min_fraction": .03,
            "huber_quantile": .75, "tune_silhouette": True,
            "max_passes": 8, "min_ch_ratio": .95}
HISTOGRAM_SPEC = {"max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                  "l2_regularization": 1., "learning_rate": .05,
                  "early_stopping": False, "random_state": PROBE_SEED}


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fit_region_representation(monthly_log_ratios, regions, held_out, ids=None,
                              include_h75=False):
    """Fit every representation component using OTHER regions' expense X.

    Expense entities without economic outcomes remain eligible for fitting.
    The returned artifact contains only training-derived fitted quantities;
    held-out values occur solely in transformed profiles and predictions.
    """
    monthly = np.asarray(monthly_log_ratios, float)
    regions = np.asarray(regions)
    if monthly.ndim != 3 or monthly.shape[0] != 12 or monthly.shape[2] != 5:
        raise ValueError("Expected twelve-month, entity, five-log-ratio tensor")
    if regions.shape != (monthly.shape[1],) or regions.dtype.kind not in "iu" or not np.isfinite(monthly).all():
        raise ValueError("Aligned finite expenses and region IDs required")
    if ids is None:
        ids = np.arange(len(regions)).astype(str)
    ids = np.asarray(ids)
    if ids.shape != regions.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError("Aligned unique entity IDs required")
    train = np.flatnonzero(regions != held_out)
    test = np.flatnonzero(regions == held_out)
    if len(train) < 5 or not len(test):
        raise ValueError("Nonempty held-out region and at least five training entities required")
    ref = monthly[:, train, :].reshape(-1, 5)
    center = np.median(ref, axis=0)
    iqr = np.quantile(ref, .75, axis=0) - np.quantile(ref, .25, axis=0)
    if not np.isfinite(iqr).all() or (iqr <= 0).any():
        raise ValueError("Positive finite training IQR required")
    profiles = np.median((monthly - center) / iqr / np.sqrt(5), axis=0)
    if not np.isfinite(profiles).all():
        raise ValueError("Transformed expense profiles must remain finite")
    labels, baseline, info = fit_frontier(profiles[train], BASELINE_SPEC, CLUSTER_SEED)
    models, infos = {"kmeans4": baseline}, {"kmeans4": info}
    if include_h75:
        _, model, detail = fit_frontier(profiles[train], H75_SPEC, CLUSTER_SEED,
                                        initial_labels=labels)
        models["huber75"], infos["huber75"] = model, detail
    all_labels = {name: predict_frontier(profiles, model) for name, model in models.items()}
    artifact = {"held_out_region": int(held_out), "training_regions": np.unique(regions[train]).astype(int).tolist(),
                "training_entity_ids": ids[train].tolist(), "training_entities": len(train),
                "calibration_observations": len(train) * 12,
                "scaler": {"ratio_center": center.tolist(), "ratio_iqr": iqr.tolist(),
                           "mode": "log_ratios_to_total", "level_weight": 0.,
                           "calibration_year": 2023}, "models": models, "model_info": infos}
    return {"train_indices": train, "test_indices": test, "profiles": profiles,
            "labels": all_labels, "artifact": artifact, "artifact_sha256": stable_hash(artifact)}


def type_design(training_types, target_types):
    """Training-only type vocabulary; unknown test categories get all zeros."""
    training_types, target_types = np.asarray(training_types), np.asarray(target_types)
    if training_types.ndim != 1 or target_types.ndim != 1:
        raise ValueError("One-dimensional municipal types required")
    vocabulary = sorted(set(training_types.tolist()))
    # Keep all columns; SVD handles the intercept-dependent dummy column.
    training = np.column_stack([training_types == name for name in vocabulary]).astype(float)
    target = np.column_stack([target_types == name for name in vocabulary]).astype(float)
    return training, target, vocabulary


def polynomial_design(train, test, numeric_columns):
    """Fixed degree-two expansion scaled using training inputs before products."""
    train, test = np.asarray(train, float), np.asarray(test, float)
    numeric_columns = np.asarray(numeric_columns, int)
    mean = train[:, numeric_columns].mean(axis=0)
    scale = train[:, numeric_columns].std(axis=0)
    scale[scale <= 1e-12] = 1.
    a, b = (train[:, numeric_columns] - mean) / scale, (test[:, numeric_columns] - mean) / scale
    keep = np.setdiff1d(np.arange(train.shape[1]), numeric_columns)
    def expand(x, original):
        terms = [x[:, j] for j in range(x.shape[1])]
        terms += [x[:, i] * x[:, j] for i, j in itertools.combinations_with_replacement(range(x.shape[1]), 2)]
        return np.column_stack([original[:, keep], *terms])
    return expand(a, train), expand(b, test), {"mean": mean.tolist(), "scale": scale.tolist()}


def fit_predict_probe(train_x, train_y, test_x, kind="linear", numeric_columns=None):
    """Fixed learner; no adaptive hyperparameter search or internal outcome split."""
    train_x, train_y, test_x = np.asarray(train_x, float), np.asarray(train_y, float), np.asarray(test_x, float)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Aligned two-dimensional training and target predictors required")
    if train_y.shape != (len(train_x),) or len(train_x) < 3:
        raise ValueError("Aligned sufficient training outcome required")
    if not all(np.isfinite(x).all() for x in [train_x, train_y, test_x]):
        raise ValueError("Finite probe inputs required")
    if kind == "histogram":
        estimator = HistGradientBoostingRegressor(**HISTOGRAM_SPEC).fit(train_x, train_y)
        prediction = estimator.predict(test_x)
        fitted = {"kind": kind, "spec": HISTOGRAM_SPEC, "iterations": int(estimator.n_iter_),
                  "training_prediction_sha256": hashlib.sha256(estimator.predict(train_x).tobytes()).hexdigest()}
    elif kind in ["linear", "quadratic"]:
        poly = None
        if kind == "quadratic":
            if numeric_columns is None:
                raise ValueError("Quadratic numeric column indices required")
            train_x, test_x, poly = polynomial_design(train_x, test_x, numeric_columns)
        mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
        active = scale > 1e-12
        design = np.column_stack([np.ones(len(train_x)), (train_x[:, active] - mean[active]) / scale[active]])
        target = np.column_stack([np.ones(len(test_x)), (test_x[:, active] - mean[active]) / scale[active]])
        beta, _, rank, singular = np.linalg.lstsq(design, train_y, rcond=1e-11)
        prediction = target @ beta
        fitted = {"kind": kind, "mean": mean.tolist(), "scale": scale.tolist(),
                  "active_columns": np.flatnonzero(active).tolist(), "beta": beta.tolist(),
                  "rank": int(rank), "columns_with_intercept": design.shape[1],
                  "smallest_singular": float(singular[-1]), "polynomial": poly,
                  "training_prediction_sha256": hashlib.sha256((design @ beta).tobytes()).hexdigest()}
    else:
        raise ValueError("Unknown fixed probe kind")
    if not np.isfinite(prediction).all():
        raise ValueError("Probe prediction must remain finite")
    fitted["fitted_sha256"] = stable_hash(fitted)
    return prediction, fitted


def score_predictions(y, predictions, regions, bootstrap=1999, seed=PROBE_SEED):
    """Pair on identical cases; bootstrap is DESCRIPTIVE conditional on OOF."""
    y, regions = np.asarray(y, float), np.asarray(regions)
    if y.ndim != 1 or regions.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("Finite aligned outcomes and regions required")
    groups = [np.flatnonzero(regions == r) for r in np.unique(regions)]
    if not groups:
        raise ValueError("Nonempty scoring cohort required")
    if not isinstance(bootstrap, int) or bootstrap < 99:
        raise ValueError("At least99 bootstrap repetitions required")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(bootstrap, len(groups)))
    pair_count = sum(len(ix) >= 2 for ix in groups)
    pair_draws = rng.integers(pair_count, size=(bootstrap, pair_count)) if pair_count else None
    losses, centered, rows = {}, {}, []
    for name, prediction in predictions.items():
        prediction = np.asarray(prediction, float)
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise ValueError("All methods must predict the identical finite scoring cohort")
        error = y - prediction
        losses[name] = np.array([np.mean(error[ix] ** 2) for ix in groups])
        centered[name] = np.array([np.var(error[ix], ddof=1) if len(ix) >= 2 else np.nan for ix in groups])
        rows.append({"predictor": name, "n": len(y), "regions": len(groups),
                     "municipality_rmse": float(np.sqrt(np.mean(error ** 2))),
                     "municipality_mae": float(np.mean(np.abs(error))),
                     "equal_region_mse": float(losses[name].mean()),
                     "equal_region_centered_error_sample_variance": float(np.nanmean(centered[name])),
                     "equal_region_mean_error": float(np.mean([error[ix].mean() for ix in groups])),
                     "regions_with_pairs": sum(len(ix) >= 2 for ix in groups), "comparisons": {}})
    for row in rows:
        name = row["predictor"]
        for reference in predictions:
            gain = losses[reference] - losses[name]
            valid = np.isfinite(centered[reference]) & np.isfinite(centered[name])
            centered_gain = centered[reference][valid] - centered[name][valid]
            row["comparisons"][reference] = {
                "equal_region_mse_improvement": float(gain.mean()),
                "conditional_region_bootstrap_95": np.quantile(gain[draws].mean(axis=1), [.025, .975]).tolist(),
                "equal_region_centered_error_variance_improvement": float(centered_gain.mean()) if len(centered_gain) else None,
                "conditional_centered_region_bootstrap_95": np.quantile(centered_gain[pair_draws].mean(axis=1), [.025, .975]).tolist() if len(centered_gain) else None,
                "region_fraction_improved_mse": float(np.mean(gain > 0))}
    return rows
