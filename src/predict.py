"""Load the saved PricingModel and answer the app's questions about one car.

Example (from the project root):
    python -m src.predict
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, LIST_COLUMNS, METADATA_PATH, PIPELINE_PATH
from src.explain import explain_car


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(PIPELINE_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def to_frame(car: dict) -> pd.DataFrame:
    """One car (dict) -> the one-row DataFrame the pipelines expect.

    Equipment may be a list (["ABS", "USB"]) or the raw comma-separated string.
    """
    row = dict(car)
    for col in LIST_COLUMNS:
        value = row.get(col, "")
        if isinstance(value, (list, tuple, set)):
            row[col] = ",".join(value)
    # An entered odometer reading is a real reading: make it a whole number, otherwise
    # the km-repair rule (fractional km = filled-in value) would replace it.
    if row.get("km") is not None:
        row["km"] = float(round(row["km"]))
    missing = [c for c in FEATURE_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def estimate(car: dict) -> dict:
    """Estimated price and its likely range (about 80% of real prices fall inside)."""
    est, low, high = load_model().predict_range(to_frame(car))
    return {"estimate": float(est[0]), "low": float(low[0]), "high": float(high[0])}


def predict_price(car: dict) -> float:
    return estimate(car)["estimate"]


def range_pct(est: float, low: float, high: float) -> tuple[float, float]:
    """The range as % around the estimate, e.g. (-8.1, +9.6).

    The ONE place the app's uncertainty percentages come from, so the card,
    the warning and the deal check always quote the same figures.
    """
    return (low / est - 1) * 100, (high / est - 1) * 100


def is_high_price(est: float, rules: dict) -> bool:
    return rules["show_high_price_warning"] and est > rules["high_price_threshold"]


def eur(x: float) -> str:
    return f"€{x:,.0f}"


def high_price_note(est: float, low: float, high: float, rules: dict) -> str | None:
    """The high-price warning text, or None when no warning applies.

    One function for the app and the API, built from THIS car's own range.
    """
    if not is_high_price(est, rules):
        return None
    lo_pct, hi_pct = range_pct(est, low, high)
    return (
        f"High-priced car: above {eur(rules['high_price_threshold'])} the model has fewer examples, "
        f"so ranges are wider. This car's range is {lo_pct:+.0f}% to {hi_pct:+.0f}% around the estimate."
    )


def price_report(car: dict, asking_price: float | None = None, top_n: int = 5) -> dict:
    """Everything the app shows for one car, in one dictionary (used by the API).

    estimate + range, the high-price warning, the deal verdict (if an asking price is
    given) and the top reasons. Built only from the functions above, so the API and
    the Streamlit app always show the same numbers.
    """
    e = estimate(car)
    est, low, high = e["estimate"], e["low"], e["high"]
    lo_pct, hi_pct = range_pct(est, low, high)
    ex = explain(car, top_n=top_n)
    deal = None
    if asking_price is not None:
        d = deal_check(asking_price, est, low, high)
        deal = {"verdict": d["verdict"], "asking_price": float(asking_price), "gap_pct": d["gap_pct"]}
    return {
        "estimate": est,
        "range": {"low": low, "high": high, "low_pct": lo_pct, "high_pct": hi_pct},
        "high_price_warning": high_price_note(est, low, high, load_metadata()["rules"]),
        "deal": deal,
        "baseline": ex["baseline"],
        "reasons": [{"reason": r.reason, "effect_pct": r.effect_pct} for r in ex["reasons"].itertuples()],
    }


def deal_check(asking_price: float, est: float, low: float, high: float) -> dict:
    """Place an asking price against this car's likely range.

    below range -> cheaper than the range: a possible deal (check why)
    fair        -> inside the range: indistinguishable from the market price
    overpriced  -> above the range: room to negotiate
    """
    if asking_price < low:
        verdict = "below range"
    elif asking_price > high:
        verdict = "overpriced"
    else:
        verdict = "fair"
    return {"verdict": verdict, "gap_pct": (asking_price - est) / est * 100, "low": low, "high": high}


def explain(car: dict, top_n: int = 5) -> dict:
    """Top reasons behind this car's estimate (see src/explain.py)."""
    return explain_car(load_model(), to_frame(car), top_n=top_n)


def depreciation_grid(car: dict, km_by_age: dict, points: int = 25) -> pd.DataFrame:
    """Ask the model for this car's price at each age (0-3) across realistic km.

    Everything else about the car stays as entered. For each age, km runs only
    across the 5th-95th percentile of real readings for cars of that age in the
    training data, so the chart never shows combinations the model never saw.
    Returns long format: age, km, price.
    """
    rows = []
    for age, (km_lo, km_hi) in km_by_age.items():
        for km in np.linspace(km_lo, km_hi, points).round(-2):
            rows.append({**car, "age": float(age), "km": float(km)})
    frame = pd.concat([to_frame(r) for r in rows], ignore_index=True)
    out = frame[["age", "km"]].copy()
    out["price"] = load_model().predict(frame)
    return out


if __name__ == "__main__":
    meta = load_metadata()
    car = dict(meta["app"]["defaults"]["Audi A3"])
    car.update(car.pop("equipment"))
    e = estimate(car)
    print(f"Typical Audi A3 -> estimate EUR {e['estimate']:,.0f}, range EUR {e['low']:,.0f}-{e['high']:,.0f}")
    print(deal_check(e["estimate"] * 0.85, e["estimate"], e["low"], e["high"]))
    ex = explain(car)
    print(f"Baseline (average car) EUR {ex['baseline']:,.0f}")
    print(ex["reasons"].round(2).to_string(index=False))
    print(depreciation_grid(car, meta["app"]["km_by_age"], points=3))
