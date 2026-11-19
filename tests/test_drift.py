"""Drift check: synthetic data (runs everywhere) and the real CSV (skipped without it)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_COLUMNS, DATA_PATH, FEATURE_COLUMNS, LIST_COLUMNS, MODEL_DIR, PROFILE_FILE, TARGET
from src.drift import build_reference_profile, check_drift, psi, to_markdown

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")


def synthetic(n=3000, seed=0) -> pd.DataFrame:
    """Made-up listings with the right columns, for testing the drift logic without real data."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["A", "B", "C"], n, p=[0.6, 0.3, 0.1]) for c in CATEGORICAL_COLUMNS})
    df["km"] = rng.gamma(2, 15000, n).round()
    df["age"] = rng.integers(0, 4, n)
    df["hp_kW"] = rng.normal(90, 20, n).round()
    df["Gears"] = rng.choice([5, 6, 7], n)
    df["Previous_Owners"] = rng.choice([0, 1, 2], n)
    df["Inspection_new"] = rng.integers(0, 2, n)
    df["Displacement_cc"] = rng.normal(1400, 200, n).round()
    df["Weight_kg"] = rng.normal(1300, 150, n).round()
    df["cons_comb"] = rng.normal(5, 0.8, n).round(1)
    for c in LIST_COLUMNS:
        df[c] = rng.choice(["ABS,USB", "ABS", "ABS,USB,Navigation system"], n)
    df[TARGET] = (20000 * np.exp(rng.normal(0, 0.3, n))).round()
    return df[FEATURE_COLUMNS + [TARGET]]


@pytest.fixture(scope="module")
def ref():
    df = synthetic(seed=0)
    return build_reference_profile(df[FEATURE_COLUMNS], df[TARGET])


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_similar_data_is_ok(ref):
    report = check_drift(synthetic(seed=1), ref)  # same process, different random draw
    assert report["verdict"] == "OK", report["reasons"] + report["watch"]


def test_price_rise_triggers_retraining(ref):
    new = synthetic(seed=1)
    new[TARGET] *= 1.15
    report = check_drift(new, ref)
    assert report["verdict"] == "RETRAIN RECOMMENDED"
    # two random samples never have exactly the same median, so allow a window around +15%
    assert 10 < report["price"]["median_shift_pct"] < 20
    assert any(r.startswith("median price moved +") for r in report["reasons"])


def test_unseen_cars_are_flagged(ref):
    new = synthetic(seed=1)
    new.loc[:300, "make_model"] = "Tesla Model 3"
    new.loc[:300, "age"] = 6
    report = check_drift(new, ref)
    assert report["out_of_range_share"] > 0.05
    assert report["verdict"] == "RETRAIN RECOMMENDED"


def test_missing_column_is_a_clear_error(ref):
    with pytest.raises(ValueError, match="missing columns"):
        check_drift(synthetic().drop(columns=["km"]), ref)


def test_markdown_report(ref):
    md = to_markdown(check_drift(synthetic(seed=1), ref), "new.csv")
    assert md.startswith("# Drift report: new.csv") and "| Feature | PSI |" in md


@needs_data
def test_real_training_data_against_its_own_profile_is_ok():
    profile = json.loads((MODEL_DIR / PROFILE_FILE).read_text())
    assert check_drift(pd.read_csv(DATA_PATH), profile)["verdict"] == "OK"


@needs_data
def test_real_data_with_simulated_inflation_and_mileage_needs_retraining():
    profile = json.loads((MODEL_DIR / PROFILE_FILE).read_text())
    new = pd.read_csv(DATA_PATH)
    new[TARGET] = (new[TARGET] * 1.15).round()   # simulated: prices up 15%
    new["km"] = (new["km"] * 1.3).round()        # simulated: cars driven 30% more
    report = check_drift(new, profile)
    assert report["verdict"] == "RETRAIN RECOMMENDED"
