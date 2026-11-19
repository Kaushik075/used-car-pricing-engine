"""MLflow logging, using the committed model: runs everywhere, including CI.

The temporary folder name contains a space on purpose, like "C:\\My Projects\\..." on Windows.
"""
import json

import mlflow
import pytest

from src.config import METADATA_PATH, MODEL_DIR
from src.tracking import log_training_run, tag_run


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "My Projects"
    root.mkdir()
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}", root / "mlruns"


def test_a_training_run_is_logged_with_params_metrics_and_model(store):
    uri, artifacts = store
    meta = json.loads(METADATA_PATH.read_text())
    run_id = log_training_run(meta, MODEL_DIR, run_name="test", tracking_uri=uri, artifact_root=artifacts)

    mlflow.set_tracking_uri(uri)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["test_mape_pct"] == pytest.approx(meta["test_metrics"]["mape_pct"])
    assert run.data.metrics["test_range_coverage"] == pytest.approx(meta["test_range_coverage"][0]["by_estimate"])
    assert run.data.params["selected_model"] == "lightgbm"
    assert run.data.tags["data_sha256"] == meta["data_file"]["sha256"]
    logged = {f.path for f in mlflow.MlflowClient().list_artifacts(run_id, "model")}
    assert logged == {"model/pricing_model.joblib", "model/metadata.json", "model/reference_profile.json"}
    assert (artifacts / run_id / "artifacts" / "model" / "pricing_model.joblib").exists()

    tag_run(run_id, {"decision": "champion kept"}, {"gate_eval_rows": 10}, tracking_uri=uri)
    run = mlflow.get_run(run_id)
    assert run.data.tags["decision"] == "champion kept" and run.data.metrics["gate_eval_rows"] == 10
