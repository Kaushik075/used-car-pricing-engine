"""Version 2 model pieces: conformal ranges, explanations, depreciation behaviour."""
import math

import numpy as np
import pytest

from src.config import PIPELINE_PATH
from src.model import assign_band, conformal_correction, conformity_scores

needs_model = pytest.mark.skipif(not PIPELINE_PATH.exists(), reason="model not trained")


def test_conformity_scores_sign():
    # inside the range -> negative, outside -> positive (log scale)
    s = conformity_scores(y=np.array([100.0, 130.0]), lo=np.array([90.0, 90.0]), hi=np.array([110.0, 110.0]))
    assert s[0] < 0 < s[1]


def test_conformal_correction_reaches_target_coverage():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=1000)
    c = conformal_correction(scores, coverage=0.8)
    assert (scores <= c).mean() >= 0.8
    assert (scores <= c).mean() < 0.82  # not much wider than needed


def test_assign_band():
    assert assign_band([5, 15, 25], edges=[10, 20]).tolist() == [0, 1, 2]


def _default_car(name="Audi A3"):
    from src.predict import load_metadata

    meta = load_metadata()
    car = dict(meta["app"]["defaults"][name])
    car.update(car.pop("equipment"))
    return car, meta


@needs_model
def test_explanation_multiplies_out_to_the_estimate():
    from src.predict import estimate, explain

    car, _ = _default_car()
    ex = explain(car, top_n=5)
    assert len(ex["reasons"]) == 6  # top 5 + "All other features"
    rebuilt = (1 + ex["baseline"]) * math.exp(ex["reasons"]["log_effect"].sum()) - 1
    assert rebuilt == pytest.approx(estimate(car)["estimate"], rel=1e-6)


@needs_model
@pytest.mark.parametrize("name", ["Audi A1", "Opel Corsa", "Renault Espace"])
def test_price_never_rises_with_km_or_age(name):
    from src.predict import depreciation_grid

    car, meta = _default_car(name)
    grid = depreciation_grid(car, meta["app"]["km_by_age"], points=15)
    for _, g in grid.groupby("age"):
        assert np.all(np.diff(g.sort_values("km")["price"].to_numpy()) <= 1e-6)
    # same km, older car -> not more expensive
    same_km = [depreciation_grid({**car}, {a: [20000, 20000]}, points=1)["price"].iat[0] for a in ["1", "2", "3"]]
    assert same_km[0] >= same_km[1] >= same_km[2]


@needs_model
def test_entered_km_is_used_as_is_even_with_a_half_km():
    """Regression test: a default like 24,913.5 km must not be treated as a filled-in value."""
    from src.predict import explain, predict_price

    car, _ = _default_car("Opel Insignia")
    car["km"] = 24913.5
    reasons = " ".join(explain(car)["reasons"]["reason"])
    assert "24,914 km" in reasons or "Kilometres" not in reasons
    assert predict_price(car) == pytest.approx(predict_price({**car, "km": 24914.0}))
