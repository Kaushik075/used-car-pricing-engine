"""Cleaning, training and app-logic checks.

Tests that need the dataset or the saved model are skipped when those files are
absent (for example on a fresh clone, because the CSV is not in the repo).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.cleaning import clean
from src.config import DATA_PATH, FEATURE_COLUMNS, PIPELINE_PATH, TARGET
from src.features import build_linear_pipeline, build_tree_pipeline
from src.predict import deal_check, is_high_price, range_pct

needs_data = pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
needs_model = pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="model not trained")

RULES = {"high_price_threshold": 20000, "show_high_price_warning": True}


def test_clean_drops_exact_duplicates_and_strips_text():
    raw = pd.DataFrame({"make_model": ["Audi A1 ", "Audi A1", "Opel Astra"], TARGET: [1, 1, 2]})
    out = clean(raw)
    assert len(out) == 2
    assert out["make_model"].tolist() == ["Audi A1", "Opel Astra"]


@pytest.mark.parametrize(
    "asking, verdict",
    [(8500, "below range"), (9000, "fair"), (10500, "fair"), (11001, "overpriced")],
)
def test_deal_check_uses_the_cars_own_range(asking, verdict):
    assert deal_check(asking, est=10000, low=9000, high=11000)["verdict"] == verdict


def test_range_pct():
    assert range_pct(10000, 9000, 12000) == pytest.approx((-10.0, 20.0))


def test_high_price_warning_can_be_switched_off_by_the_data():
    assert is_high_price(25000, RULES)
    assert not is_high_price(15000, RULES)
    assert not is_high_price(25000, {**RULES, "show_high_price_warning": False})


@needs_data
@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_pipelines_train_and_predict_on_real_sample(kind):
    from lightgbm import LGBMRegressor

    df = clean(pd.read_csv(DATA_PATH)).sample(2000, random_state=0)
    model = (
        build_linear_pipeline(Ridge())
        if kind == "linear"
        else build_tree_pipeline(LGBMRegressor(n_estimators=100, verbose=-1))
    ).fit(df[FEATURE_COLUMNS], df[TARGET])
    preds = model.predict(df[FEATURE_COLUMNS].head(50))
    assert np.all(preds > 0)
    assert np.mean(np.abs(preds - df[TARGET].head(50)) / df[TARGET].head(50)) < 0.25


@needs_data
@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_unseen_category_does_not_crash(kind):
    from lightgbm import LGBMRegressor

    df = clean(pd.read_csv(DATA_PATH)).sample(1000, random_state=1)
    model = (
        build_linear_pipeline(Ridge())
        if kind == "linear"
        else build_tree_pipeline(LGBMRegressor(n_estimators=50, verbose=-1))
    ).fit(df[FEATURE_COLUMNS], df[TARGET])
    row = df[FEATURE_COLUMNS].head(1).copy()
    row["make_model"] = "Tesla Model 3"
    assert model.predict(row)[0] > 0


@needs_model
def test_saved_model_gives_sane_estimates_and_ranges_for_app_defaults():
    from src.predict import estimate, load_metadata

    for name, defaults in load_metadata()["app"]["defaults"].items():
        car = {k: v for k, v in defaults.items() if k != "equipment"}
        car.update(defaults["equipment"])
        e = estimate(car)
        assert 3000 < e["estimate"] < 100000, name
        assert e["low"] <= e["estimate"] <= e["high"], name
