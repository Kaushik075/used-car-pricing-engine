"""The API: endpoints, input validation, and agreement with src/predict.py and the Streamlit app."""
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.predict import deal_check, estimate, explain, high_price_note, load_metadata

client = TestClient(app)
META = load_metadata()
MODELS = sorted(META["app"]["defaults"])
APP_FILE = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def default_car(name: str) -> dict:
    car = dict(META["app"]["defaults"][name])
    car.update(car.pop("equipment"))
    return car


def eur(x):
    return f"€{x:,.0f}"


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "lightgbm"


def test_options_lists_the_models():
    assert client.get("/options").json()["make_model"] == MODELS


def test_minimal_request_fills_the_rest_with_model_defaults():
    r = client.post("/predict", json={"make_model": "Opel Corsa", "age": 1, "km": 30000})
    assert r.status_code == 200
    body = r.json()
    assert body["range"]["low"] <= body["estimate"] <= body["range"]["high"]
    assert body["deal"] is None and len(body["reasons"]) == 6
    assert "Fuel" in body["filled_fields"] and "km" not in body["filled_fields"]


def test_bad_input_lists_every_problem_in_plain_words():
    r = client.post("/predict", json={"make_model": "Tesla Model 3", "age": 6, "km": -5,
                                      "Fuel": "Petrol", "colour": "red", "Extras": ["Jetpack"]})
    assert r.status_code == 422
    problems = {p["field"]: p["message"] for p in r.json()["problems"]}
    assert set(problems) == {"make_model", "age", "km", "Fuel", "colour", "Extras"}
    assert "Audi A3" in problems["make_model"]          # tells you what IS allowed
    assert "between 0 and 3" in problems["age"]
    assert "Benzine" in problems["Fuel"]


def test_missing_required_fields():
    problems = client.post("/predict", json={"age": 1}).json()["problems"]
    assert {p["field"] for p in problems} == {"make_model", "km"}


@pytest.mark.parametrize("name", MODELS)
def test_api_matches_the_prediction_functions(name):
    """API response == calling src/predict.py directly, field by field."""
    car = default_car(name)
    body = client.post("/predict", json={**car, "asking_price": 20000}).json()
    e = estimate(car)
    assert body["estimate"] == pytest.approx(e["estimate"], rel=1e-12)
    assert (body["range"]["low"], body["range"]["high"]) == pytest.approx((e["low"], e["high"]), rel=1e-12)
    assert body["deal"]["verdict"] == deal_check(20000, e["estimate"], e["low"], e["high"])["verdict"]
    assert body["high_price_warning"] == high_price_note(e["estimate"], e["low"], e["high"], META["rules"])
    ex = explain(car)
    assert [r["reason"] for r in body["reasons"]] == list(ex["reasons"]["reason"])
    assert body["filled_fields"] == []


def test_api_matches_what_the_streamlit_app_shows():
    """Drive the real app for every model and compare what it DISPLAYS with the API's answer."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    for name in MODELS:
        at.selectbox[0].set_value(name).run()
        [n for n in at.number_input if n.label.startswith("Seller")][0].set_value(20000).run()
        body = client.post("/predict", json={**default_car(name), "asking_price": 20000}).json()

        markdown = [m.body for m in at.markdown]
        assert at.metric[0].value == eur(body["estimate"]), name
        assert f"Likely range **{eur(body['range']['low'])} – {eur(body['range']['high'])}**" in markdown, name
        badge = {"below range": "Below range", "fair": "Fair price", "overpriced": "Overpriced"}[body["deal"]["verdict"]]
        assert any(badge in m for m in markdown), name
        warnings = [w.value for w in at.warning]
        assert warnings == ([body["high_price_warning"]] if body["high_price_warning"] else []), name
        reasons_md = next(m for m in markdown if m.startswith("Starting point"))
        assert eur(body["baseline"]) in reasons_md
        for r in body["reasons"]:
            pct = abs(r["effect_pct"])
            shown = "barely changes the price" if pct < 0.5 else f"about **{pct:.0f}%**"
            assert f"- {r['reason']}: " in reasons_md and shown in reasons_md, (name, r)
        assert math.isclose(sum(math.log1p(r["effect_pct"] / 100) for r in body["reasons"]),
                            math.log1p(body["estimate"]) - math.log1p(body["baseline"]), rel_tol=1e-9)
