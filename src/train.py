"""Train, compare, calibrate and save the Version 2 pricing model.

Run from the project root:
    python -m src.train

Order of work (each step only sees what it is allowed to see):
    1. clean the full file (rule-based only) and split 80/20; test is locked away
    2. compare linear models and two gradient-boosting libraries with 5-fold CV
       on TRAIN, tuning each; pick the winner by CV error (rule in select_model)
    3. out-of-fold (OOF) predictions on TRAIN show errors per price band,
       linear vs boosting ("before / after")
    4. price ranges: fit 10% / 90% quantile models, then conformal calibration
       per price band on their OOF predictions (still TRAIN only)
    5. refit everything on all of TRAIN, score ONCE on TEST, save model + metadata
    6. (Version 3) save a drift reference profile and training-row fingerprints,
       and log the whole run to MLflow

`run_training()` is reused by the retraining pipeline (src/retrain.py).
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.cleaning import clean, cleaning_report, load_raw, row_hashes
from src.config import (
    CATEGORICAL_COLUMNS,
    CV_FOLDS,
    FEATURE_COLUMNS,
    DATA_PATH,
    HASHES_FILE,
    LIST_COLUMNS,
    METADATA_FILE,
    MIN_CATEGORY_COUNT,
    MODEL_DIR,
    MODEL_FILE,
    NUMERIC_COLUMNS,
    PROFILE_FILE,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from src.drift import build_reference_profile
from src.evaluate import regression_metrics
from src.features import FeatureListBinarizer, build_linear_pipeline, build_tree_pipeline
from src.model import (
    QUANTILES,
    TARGET_COVERAGE,
    PricingModel,
    assign_band,
    conformal_correction,
    conformity_scores,
)

CV = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"mape": "neg_mean_absolute_percentage_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
N_SEARCH = 15  # random hyper-parameter combinations tried per boosting library
P = "regressor__model__estimator__"  # path to the boosting model's settings inside the pipeline


def split_data(df: pd.DataFrame):
    return train_test_split(
        df[FEATURE_COLUMNS], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def lgbm(**kw) -> LGBMRegressor:
    return LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **kw)


def candidate_searches() -> dict:
    """Name -> a CV search object. Linear models use small grids, boosting random search."""
    a = "regressor__model__alpha"
    return {
        "ridge": GridSearchCV(build_linear_pipeline(Ridge()), {a: [0.3, 1, 3, 10]}, cv=CV,
                              scoring=SCORING, refit="mape", n_jobs=2),
        "lasso": GridSearchCV(build_linear_pipeline(Lasso(max_iter=20_000)), {a: [3e-5, 1e-4, 3e-4]},
                              cv=CV, scoring=SCORING, refit="mape", n_jobs=2),
        "hist_gradient_boosting": RandomizedSearchCV(
            build_tree_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                P + "max_iter": [300, 600, 1000],
                P + "learning_rate": [0.03, 0.05, 0.1],
                P + "max_leaf_nodes": [15, 31, 63],
                P + "min_samples_leaf": [10, 20, 40],
                P + "l2_regularization": [0.0, 1.0],
                P + "max_features": [0.7, 1.0],
            },
            n_iter=N_SEARCH, cv=CV, scoring=SCORING, refit="mape", random_state=RANDOM_STATE, n_jobs=2,
        ),
        "lightgbm": RandomizedSearchCV(
            build_tree_pipeline(lgbm()),
            {
                P + "n_estimators": [300, 600, 1000],
                P + "learning_rate": [0.03, 0.05, 0.1],
                P + "num_leaves": [15, 31, 63],
                P + "min_child_samples": [10, 20, 40],
                P + "reg_lambda": [0.0, 1.0],
                P + "colsample_bytree": [0.7, 1.0],
            },
            n_iter=N_SEARCH, cv=CV, scoring=SCORING, refit="mape", random_state=RANDOM_STATE, n_jobs=2,
        ),
    }


def compare_models(X_train, y_train):
    """Tune every candidate with 5-fold CV on TRAIN. Returns (table, fitted searches)."""
    rows, searches = [], {}
    for name, search in candidate_searches().items():
        t0 = time.time()
        search.fit(X_train, y_train)
        i, res = search.best_index_, search.cv_results_
        rows.append({
            "model": name,
            "family": "linear" if name in ("ridge", "lasso") else "boosting",
            "cv_mape_pct": -res["mean_test_mape"][i] * 100,
            "cv_mape_std": res["std_test_mape"][i] * 100,
            "cv_mae_eur": -res["mean_test_mae"][i],
            "cv_r2": res["mean_test_r2"][i],
            "search_minutes": (time.time() - t0) / 60,
            "best_params": json.dumps({k.split("__")[-1]: v for k, v in search.best_params_.items()}),
        })
        searches[name] = search
    return pd.DataFrame(rows).set_index("model").sort_values("cv_mape_pct"), searches


def select_model(table: pd.DataFrame) -> str:
    """Selection rule, fixed BEFORE tuning results were seen:

    1. lowest cross-validated MAPE wins;
    2. but if the best boosting model is within one fold-std of LightGBM, take
       LightGBM: it is faster and has exact per-car explanations built in
       (pred_contrib = TreeSHAP), so the app needs no extra explanation library.
    """
    best = table.index[0]
    if best == "hist_gradient_boosting":
        gap = table.loc["lightgbm", "cv_mape_pct"] - table.loc[best, "cv_mape_pct"]
        if gap <= table.loc[best, "cv_mape_std"]:
            return "lightgbm"
    return best


def band_edges_from(y_train) -> list:
    """Price bands = training-price quintiles (5 bands of equal size)."""
    return [float(v) for v in np.percentile(y_train, [20, 40, 60, 80])]


def band_labels(edges) -> list:
    e = [f"€{v:,.0f}" for v in edges]
    return [f"under {e[0]}"] + [f"{e[i]}–{e[i + 1]}" for i in range(len(e) - 1)] + [f"over {e[-1]}"]


def errors_by_band(y, pred, edges) -> pd.DataFrame:
    """MAPE, MAE and median bias per band of ACTUAL price."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    band = assign_band(y, edges)
    rows = []
    for b, label in enumerate(band_labels(edges)):
        m = band == b
        rel = (pred[m] - y[m]) / y[m] * 100
        rows.append({"band": label, "n": int(m.sum()), "mape_pct": float(np.abs(rel).mean()),
                     "mae_eur": float(np.abs(pred[m] - y[m]).mean()), "median_bias_pct": float(np.median(rel))})
    return pd.DataFrame(rows).set_index("band")


def quantile_model(best_tree_params: dict, alpha: float):
    """Same tuned LightGBM settings, but trained to predict a quantile of price.

    LightGBM does not allow monotonic constraints with the quantile objective, so
    the two range models are unconstrained. The point estimate keeps its constraint,
    and PricingModel.predict_range makes sure low <= estimate <= high.
    """
    return build_tree_pipeline(lgbm(objective="quantile", alpha=alpha, **best_tree_params), monotone=False)


def coverage_table(y, low, high, est, edges) -> pd.DataFrame:
    """Share of real prices inside the range, overall and per band.

    'by_estimate' groups cars by their estimated price (what the app knows);
    'by_actual' groups them by real price (harder, shown for honesty).
    """
    y, low, high, est = (np.asarray(a, float) for a in (y, low, high, est))
    inside = (y >= low) & (y <= high)
    width = (high - low) / est * 100
    rows = [{"band": "all", "by_estimate": inside.mean(), "by_actual": inside.mean(),
             "mean_width_pct": width.mean(), "n": len(y)}]
    be, ba = assign_band(est, edges), assign_band(y, edges)
    for b, label in enumerate(band_labels(edges)):
        rows.append({"band": label, "by_estimate": inside[be == b].mean(), "by_actual": inside[ba == b].mean(),
                     "mean_width_pct": width[be == b].mean(), "n": int((be == b).sum())})
    return pd.DataFrame(rows).set_index("band")


def app_options(X_train: pd.DataFrame) -> dict:
    """Dropdowns, input ranges, per-model defaults and depreciation-chart limits for the app.

    Only aggregated values (modes, medians, ranges, percentiles) are stored, no raw rows.
    """
    categories = {}
    for col in CATEGORICAL_COLUMNS:
        counts = X_train[col].value_counts()
        categories[col] = sorted(counts[counts >= MIN_CATEGORY_COUNT].index.tolist())
    ranges = {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERIC_COLUMNS + ["km"]}
    equip = FeatureListBinarizer().fit(X_train[LIST_COLUMNS]).vocabulary_

    # Depreciation chart: only draw km values that real cars of each age have
    # (5th to 95th percentile of real readings), so the chart never extrapolates.
    real_km = X_train["km"].where(X_train["km"] % 1 == 0)
    km_by_age = {
        str(int(a)): [float(real_km[X_train["age"] == a].quantile(q)) for q in (0.05, 0.95)]
        for a in sorted(X_train["age"].unique())
    }

    defaults = {}
    for model_name, grp in X_train.groupby("make_model"):
        if len(grp) < MIN_CATEGORY_COUNT:
            continue
        d = {c: grp[c].mode().iat[0] for c in CATEGORICAL_COLUMNS}
        d.update({c: float(grp[c].median()) for c in NUMERIC_COLUMNS})
        d["km"] = float(round(grp["km"].median()))  # whole km: fractional km means "filled in"
        d["equipment"] = {}
        for col in LIST_COLUMNS:
            items = grp[col].map(FeatureListBinarizer._split)
            d["equipment"][col] = [t for t in equip[col] if items.map(lambda s, t=t: t in s).mean() >= 0.5]
        defaults[model_name] = d
    return {"categories": categories, "ranges": ranges, "equipment": equip,
            "km_by_age": km_by_age, "defaults": defaults}


def data_facts(df: pd.DataFrame) -> dict:
    """Facts about the cleaned data that the README quotes (computed, never typed)."""
    frac = df["km"] % 1 != 0
    return {
        "price_skew": float(df[TARGET].skew()),
        "log_price_skew": float(np.log1p(df[TARGET]).skew()),
        "fractional_km_rows": int(frac.sum()),
        "fractional_km_top_values": {f"{k:,.2f}": int(v) for k, v in df.loc[frac, "km"].value_counts().head(3).items()},
        "age_range": [int(df["age"].min()), int(df["age"].max())],
    }


def records(df: pd.DataFrame) -> list:
    return df.reset_index().to_dict(orient="records")


def fixed_settings_table(X_train, y_train, tree_params: dict, lasso_alpha: float):
    """Retraining without a new search: cross-validate the champion's settings on the new data.

    Returns the same kind of table as compare_models, plus the unfitted pipelines.
    """
    candidates = {
        "lightgbm": (build_tree_pipeline(lgbm(**tree_params)), tree_params),
        "lasso": (build_linear_pipeline(Lasso(alpha=lasso_alpha, max_iter=20_000)), {"alpha": lasso_alpha}),
    }
    rows, pipes = [], {}
    for name, (pipe, params) in candidates.items():
        t0 = time.time()
        res = cross_validate(pipe, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=2)
        rows.append({
            "model": name,
            "family": "linear" if name == "lasso" else "boosting",
            "cv_mape_pct": -res["test_mape"].mean() * 100,
            "cv_mape_std": res["test_mape"].std() * 100,
            "cv_mae_eur": -res["test_mae"].mean(),
            "cv_r2": res["test_r2"].mean(),
            "search_minutes": (time.time() - t0) / 60,
            "best_params": json.dumps(params),
        })
        pipes[name] = pipe
    return pd.DataFrame(rows).set_index("model").sort_values("cv_mape_pct"), pipes


def file_sha256(path) -> str:
    """Fingerprint of the data file, so every saved model and MLflow run says which data trained it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_training(data_path=DATA_PATH, model_dir=MODEL_DIR, tune=True, fixed_params=None,
                 track=True, run_name="train", tags=None) -> dict:
    """Train, calibrate, test once and save a complete model folder. Returns the metadata.

    tune=True        full model comparison and hyper-parameter search (about 5 minutes)
    tune=False       reuse `fixed_params` = {"lightgbm": {...}, "lasso_alpha": a} (about 1-2 minutes);
                     used by the retraining pipeline to refit the champion's recipe on new data
    track=True       log the run (parameters, CV and test results, model files) to MLflow
    """
    model_dir = Path(model_dir)
    t_start = time.time()
    raw = load_raw(data_path)
    df = clean(raw)
    X_train, X_test, y_train, y_test = split_data(df)
    edges = band_edges_from(y_train)
    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}  Price bands: {band_labels(edges)}")

    # ---- 2. compare and select (TRAIN only)
    if tune:
        table, searches = compare_models(X_train, y_train)
        selected = select_model(table)
        best_linear = table[table.family == "linear"].index[0]
        point_unfitted = searches[selected].best_estimator_
        linear_unfitted = searches[best_linear].best_estimator_
        tree_params = {k.split("__")[-1]: v for k, v in searches[selected].best_params_.items()}
    else:
        tree_params = dict(fixed_params["lightgbm"])
        table, pipes = fixed_settings_table(X_train, y_train, tree_params, fixed_params["lasso_alpha"])
        selected, best_linear = "lightgbm", "lasso"
        point_unfitted, linear_unfitted = pipes["lightgbm"], pipes["lasso"]
    print("\nCross-validation (train only):\n", table.drop(columns="best_params").round(3).to_string())
    print(f"\nSelected: {selected}   (best linear: {best_linear})")
    if selected != "lightgbm":
        raise SystemExit("Explanations and ranges are built on LightGBM; revisit if another model wins.")

    # ---- 3. out-of-fold predictions: errors per band before (linear) / after (boosting)
    oof_point = cross_val_predict(point_unfitted, X_train, y_train, cv=CV, n_jobs=2)
    oof_linear = cross_val_predict(linear_unfitted, X_train, y_train, cv=CV, n_jobs=2)
    cv_bands = {"linear": errors_by_band(y_train, oof_linear, edges),
                "boosting": errors_by_band(y_train, oof_point, edges)}
    print("\nCV errors by actual-price band, boosting:\n", cv_bands["boosting"].round(2).to_string())

    # ---- 4. price ranges: quantile models + per-band conformal calibration on OOF predictions
    lower_unfitted = quantile_model(tree_params, QUANTILES[0])
    upper_unfitted = quantile_model(tree_params, QUANTILES[1])
    oof_lo = np.minimum(cross_val_predict(lower_unfitted, X_train, y_train, cv=CV, n_jobs=2), oof_point)
    oof_hi = np.maximum(cross_val_predict(upper_unfitted, X_train, y_train, cv=CV, n_jobs=2), oof_point)
    raw_cov = coverage_table(y_train, oof_lo, oof_hi, oof_point, edges)
    scores = conformity_scores(y_train.to_numpy(float), oof_lo, oof_hi)
    oof_band = assign_band(oof_point, edges)
    corrections = [conformal_correction(scores[oof_band == b]) for b in range(len(edges) + 1)]
    print("Conformal corrections per band (log scale):", np.round(corrections, 4))

    # ---- 5. final fit on all TRAIN, then the single test evaluation
    model = PricingModel(
        point=clone(point_unfitted).fit(X_train, y_train),
        lower=clone(lower_unfitted).fit(X_train, y_train),
        upper=clone(upper_unfitted).fit(X_train, y_train),
        band_edges=edges,
        corrections=corrections,
    )
    linear_final = clone(linear_unfitted).fit(X_train, y_train)  # only for the before/after report

    est, low, high = model.predict_range(X_test)
    lin_test = linear_final.predict(X_test)
    test_metrics = regression_metrics(y_test, est)
    test_metrics_linear = regression_metrics(y_test, lin_test)
    test_bands = {"linear": errors_by_band(y_test, lin_test, edges),
                  "boosting": errors_by_band(y_test, est, edges)}
    test_cov = coverage_table(y_test, low, high, est, edges)
    print("\nTest, boosting:", {k: round(v, 3) for k, v in test_metrics.items()})
    print("Test, linear:  ", {k: round(v, 3) for k, v in test_metrics_linear.items()})
    print("Test range coverage:\n", test_cov.round(3).to_string())

    # Version 1's weakest spot: the most expensive cars (test set, both models)
    top = y_test.to_numpy(float) > 40_000
    top_end = {"threshold_eur": 40_000, "n": int(top.sum())}
    for name, p in [("linear", lin_test), ("boosting", est)]:
        rel = (p[top] - y_test.to_numpy(float)[top]) / y_test.to_numpy(float)[top] * 100
        top_end[name] = {"share_under_estimated": float(np.mean(rel < 0)), "median_bias_pct": float(np.median(rel))}

    # High-price warning: keep it only if the data still shows less certainty above the threshold.
    threshold = float(np.percentile(y_train, 75))
    oof_width = (np.expm1(np.log1p(oof_hi) + np.array(corrections)[oof_band])
                 - np.expm1(np.log1p(oof_lo) - np.array(corrections)[oof_band])) / oof_point * 100
    above = oof_point > threshold
    rules = {
        "high_price_threshold": threshold,
        "oof_mean_width_pct": {"normal": float(oof_width[~above].mean()), "high": float(oof_width[above].mean())},
        "range_coverage_target": TARGET_COVERAGE,
    }
    rules["show_high_price_warning"] = rules["oof_mean_width_pct"]["high"] > rules["oof_mean_width_pct"]["normal"]

    # ---- save the model folder
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILE
    joblib.dump(model, model_path, compress=3)
    # Fingerprints of the training rows: lets the promotion gate prove a test row was never trained on.
    np.save(model_dir / HASHES_FILE, row_hashes(pd.concat([X_train, y_train], axis=1)))
    # Summary of the training data for the drift check (aggregates only, no rows).
    (model_dir / PROFILE_FILE).write_text(json.dumps(build_reference_profile(X_train, y_train), indent=1))
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "data_file": {"name": Path(data_path).name, "sha256": file_sha256(data_path)},
        "data": {**cleaning_report(raw, df), **data_facts(df)},
        "split": {"train_rows": len(X_train), "test_rows": len(X_test), "random_state": RANDOM_STATE},
        "training_mode": "full search" if tune else "champion settings refit",
        "band_edges": edges,
        "band_labels": band_labels(edges),
        "selected_model": selected,
        "best_linear_model": best_linear,
        "selection_rule": select_model.__doc__.strip(),
        "cv_comparison": records(table),
        "oof_metrics": {"boosting": regression_metrics(y_train, oof_point),
                        "linear": regression_metrics(y_train, oof_linear)},
        "cv_errors_by_band": {k: records(v) for k, v in cv_bands.items()},
        "range": {"method": "conformalized quantile regression, per price band",
                  "quantiles": list(QUANTILES), "corrections_log": corrections,
                  "oof_raw_coverage": records(raw_cov)},
        "test_metrics": test_metrics,
        "test_metrics_linear": test_metrics_linear,
        "test_errors_by_band": {k: records(v) for k, v in test_bands.items()},
        "test_range_coverage": records(test_cov),
        "test_top_end": top_end,
        "rules": rules,
        "model_file_kb": round(model_path.stat().st_size / 1024, 1),
        "training_minutes": round((time.time() - t_start) / 60, 2),
        "app": app_options(X_train),
    }
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nSaved {MODEL_FILE} ({metadata['model_file_kb']} KB), {METADATA_FILE}, {PROFILE_FILE}, "
          f"{HASHES_FILE} to {model_dir}")

    if track:
        from src.tracking import log_training_run  # imported here: the API and app never need MLflow

        metadata["mlflow_run_id"] = log_training_run(metadata, model_dir, run_name=run_name, tags=tags)
        (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
        print(f"Logged to MLflow, run id {metadata['mlflow_run_id']}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train the pricing model (full search) and save it.")
    parser.add_argument("--data", default=str(DATA_PATH), help="CSV to train on (default: data/Car_Price_data.csv)")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="folder to save the model in (default: models/)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    args = parser.parse_args()
    run_training(args.data, args.model_dir, tune=True, track=not args.no_mlflow)


if __name__ == "__main__":
    main()
