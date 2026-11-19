"""Promotion gate: the decision rules (run everywhere) and full retraining runs (need the CSV).

The end-to-end tests work on copies in a temporary folder: the real models/ folder
and the project's MLflow store are never touched.
"""
import json
import shutil

import joblib
import pandas as pd
import pytest

from src.config import DATA_PATH, HASHES_FILE, METADATA_FILE, MODEL_DIR, MODEL_FILE, PROFILE_FILE, TARGET
from src.retrain import champion_recipe, decide, retrain

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
GOOD = {"mape_pct": 5.0, "coverage": 0.80}


def test_promotes_when_clearly_better_and_ranges_hold():
    ok, reasons = decide({"mape_pct": 5.3, "coverage": 0.8}, GOOD, n_eval=2000)
    assert ok and all("PASS" in r for r in reasons)


def test_keeps_champion_when_gain_is_too_small():
    ok, reasons = decide({"mape_pct": 5.03, "coverage": 0.8}, GOOD, n_eval=2000)
    assert not ok and "FAIL" in reasons[0]


@pytest.mark.parametrize("coverage", [0.70, 0.90])
def test_keeps_champion_when_ranges_miss_80_percent(coverage):
    ok, reasons = decide({"mape_pct": 6.0, "coverage": 0.8}, {"mape_pct": 5.0, "coverage": coverage}, n_eval=2000)
    assert not ok and "FAIL" in reasons[1]


def test_keeps_champion_without_enough_unseen_rows():
    ok, reasons = decide({"mape_pct": 9.0, "coverage": 0.8}, GOOD, n_eval=100)
    assert not ok and "fair comparison is not possible" in reasons[0]


def test_champion_recipe_reads_saved_settings():
    recipe = champion_recipe(json.loads((MODEL_DIR / METADATA_FILE).read_text()))
    assert recipe["lightgbm"]["num_leaves"] > 0 and recipe["lasso_alpha"] > 0


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of the live model folder plus a private MLflow store, inside a temp folder."""
    models = tmp_path / "models"
    models.mkdir()
    for f in (MODEL_FILE, METADATA_FILE, PROFILE_FILE, HASHES_FILE):
        shutil.copy2(MODEL_DIR / f, models / f)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("UCPE_MLFLOW_ARTIFACTS", str(tmp_path / "mlruns"))
    return tmp_path, models


@needs_data
def test_simulated_price_shift_gets_promoted(sandbox):
    """Simulation: prices 15% higher. The old champion is now systematically low; a
    challenger trained on the new prices should win and replace it."""
    tmp, models = sandbox
    new = pd.read_csv(DATA_PATH).sample(5000, random_state=1)
    new[TARGET] = (new[TARGET] * 1.15).round()
    new.to_csv(tmp / "shifted.csv", index=False)
    before = joblib.load(models / MODEL_FILE)

    entry = retrain(tmp / "shifted.csv", models)

    assert entry["decision"] == "promoted", entry["reasons"]
    assert entry["eval_rows_removed_seen_by_champion"] == 0   # shifted prices -> rows the champion never saw
    assert entry["challenger"]["mape_pct"] < entry["champion"]["mape_pct"]
    assert list((models / "archive").iterdir())                 # old champion kept
    assert joblib.load(models / MODEL_FILE).band_edges != before.band_edges
    log = [json.loads(line) for line in (models / "promotion_log.jsonl").read_text().splitlines()]
    assert log[-1]["decision"] == "promoted"


@needs_data
def test_old_rows_are_never_used_to_judge_the_champion(sandbox):
    """A sample of the ORIGINAL data: most of its test rows were in the champion's training
    set, so they are removed, too few remain, and the champion is kept."""
    tmp, models = sandbox
    pd.read_csv(DATA_PATH).sample(3000, random_state=2).to_csv(tmp / "old_sample.csv", index=False)
    model_bytes = (models / MODEL_FILE).read_bytes()

    entry = retrain(tmp / "old_sample.csv", models)

    assert entry["eval_rows_removed_seen_by_champion"] > 0
    assert entry["decision"] == "champion kept"
    assert "fair comparison is not possible" in entry["reasons"][0]
    assert (models / MODEL_FILE).read_bytes() == model_bytes    # live model untouched
