"""Experiment tracking with MLflow (local and free: a SQLite file plus an mlruns/ folder).

Every training run logs:
  - parameters : chosen model and its settings, data split, library versions
  - metrics    : cross-validation results for every compared model, test results,
                 price-range coverage overall and per band
  - tags       : which data file trained it (name + SHA-256 fingerprint), training mode
  - artifacts  : the saved model file, metadata.json and the drift reference profile

Open the UI from the project folder (Windows Command Prompt):
    set MLFLOW_DISABLE_TELEMETRY=true
    mlflow ui --backend-store-uri sqlite:///mlflow.db
then browse to http://127.0.0.1:5000
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep MLflow's console output quiet
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # MLflow 3 sends usage telemetry by default: off

import mlflow  # noqa: E402

from src.config import (  # noqa: E402
    METADATA_FILE,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_FILE,
    PROFILE_FILE,
)


def _uri(tracking_uri=None) -> str:
    """Explicit argument > MLFLOW_TRACKING_URI environment variable > the project's mlflow.db."""
    return tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI


def _artifact_root(artifact_root=None) -> Path:
    return Path(artifact_root or os.environ.get("UCPE_MLFLOW_ARTIFACTS") or MLFLOW_ARTIFACT_ROOT)


def _experiment(tracking_uri: str, artifact_root: Path) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        # Artifacts go to an explicit folder, so it does not matter where the command is run from.
        return mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=artifact_root.resolve().as_uri())
    return exp.experiment_id


def log_training_run(metadata: dict, model_dir, run_name="train", tags=None,
                     tracking_uri=None, artifact_root=None) -> str:
    """Log one finished training run. Returns the MLflow run id."""
    model_dir = Path(model_dir)
    exp_id = _experiment(_uri(tracking_uri), _artifact_root(artifact_root))
    cv = {row["model"]: row for row in metadata["cv_comparison"]}
    selected = metadata["selected_model"]

    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.set_tags({
            "data_file": metadata["data_file"]["name"],
            "data_sha256": metadata["data_file"]["sha256"],
            "training_mode": metadata["training_mode"],
            "selected_model": selected,
            **(tags or {}),
        })
        params = {
            "selected_model": selected,
            "best_linear_model": metadata["best_linear_model"],
            "train_rows": metadata["split"]["train_rows"],
            "test_rows": metadata["split"]["test_rows"],
            "random_state": metadata["split"]["random_state"],
            "range_quantiles": metadata["range"]["quantiles"],
            "sklearn_version": metadata["versions"]["sklearn"],
            "lightgbm_version": metadata["versions"]["lightgbm"],
        }
        params.update({f"{selected}__{k}": v for k, v in json.loads(cv[selected]["best_params"]).items()})
        params.update({f"{metadata['best_linear_model']}__{k}": v
                       for k, v in json.loads(cv[metadata["best_linear_model"]]["best_params"]).items()})
        mlflow.log_params(params)

        metrics = {}
        for name, row in cv.items():
            for k in ("cv_mape_pct", "cv_mape_std", "cv_mae_eur", "cv_r2"):
                metrics[f"{k}__{name}"] = row[k]
        for k in ("mape_pct", "median_ape_pct", "mae_eur", "r2"):
            metrics[f"test_{k}"] = metadata["test_metrics"][k]
            metrics[f"test_{k}__linear"] = metadata["test_metrics_linear"][k]
        cov = metadata["test_range_coverage"]
        metrics["test_range_coverage"] = cov[0]["by_estimate"]
        metrics["test_range_width_pct"] = cov[0]["mean_width_pct"]
        for i, row in enumerate(cov[1:]):
            metrics[f"test_range_coverage__band{i}"] = row["by_estimate"]
        metrics["model_file_kb"] = metadata["model_file_kb"]
        if "training_minutes" in metadata:  # MLflow's own "Duration" only covers this logging step
            metrics["training_minutes"] = metadata["training_minutes"]
        mlflow.log_metrics(metrics)

        for f in (MODEL_FILE, METADATA_FILE, PROFILE_FILE):
            mlflow.log_artifact(str(model_dir / f), artifact_path="model")
        return run.info.run_id


def tag_run(run_id: str, tags: dict, metrics: dict | None = None, tracking_uri=None):
    """Add tags (and optional metrics) to an existing run, e.g. the promotion decision."""
    mlflow.set_tracking_uri(_uri(tracking_uri))
    client = mlflow.MlflowClient()
    for k, v in tags.items():
        client.set_tag(run_id, k, str(v))
    for k, v in (metrics or {}).items():
        client.log_metric(run_id, k, float(v))
