"""Turn a car's SHAP contributions into a few plain-language reasons.

The point model works on log(1 + price), and SHAP splits that log price exactly:
    log(1 + price) = bias + sum of feature contributions
so  1 + price = e^bias  x  e^c1  x  e^c2  x ...
Each feature therefore MULTIPLIES a baseline price; e^c - 1 is its % effect.
The baseline e^bias - 1 is the model's price for an average car in the training data.
"""
import math

import pandas as pd

LABELS = {
    "make_model": "Model", "body_type": "Body type", "vat": "VAT", "Type": "Listing type",
    "Fuel": "Fuel", "Paint_Type": "Paint", "Upholstery_type": "Upholstery",
    "Gearing_Type": "Gearbox", "Drive_chain": "Drive",
}
NUMBER_TEXT = {
    "age": lambda v: f"Age: {v:.0f} year" + ("" if v == 1 else "s"),
    "hp_kW": lambda v: f"Power: {v:,.0f} kW",
    "Gears": lambda v: f"Gears: {v:.0f}",
    "Previous_Owners": lambda v: f"Previous owners: {v:.0f}",
    "Inspection_new": lambda v: "New inspection: " + ("yes" if v else "no"),
    "Displacement_cc": lambda v: f"Engine size: {v:,.0f} cc",
    "Weight_kg": lambda v: f"Weight: {v:,.0f} kg",
    "cons_comb": lambda v: f"Fuel use: {v:.1f} l/100 km",
}


def _reason(feature: str, car: pd.Series, xt: pd.Series) -> str:
    group, _, name = feature.partition("__")
    if group == "cat":
        return f"{LABELS.get(name, name)}: {car[name]}"
    if group == "km":
        return f"Kilometres: {xt['km__km']:,.0f} km"
    if group == "num":
        return NUMBER_TEXT[name](float(car[name]))
    if group == "equip":
        item = name.split("__", 1)[1]
        return ("Has " if xt[feature] == 1 else "No ") + item.lower()
    return feature


def explain_car(model, car_frame: pd.DataFrame, top_n: int = 5) -> dict:
    """Top reasons for ONE car's estimate.

    Returns {"baseline": euros, "estimate": euros, "reasons": DataFrame[reason, effect_pct]}
    where the reasons are the top_n features by size of effect, plus one
    "All other features" row, so the % effects multiply out to the estimate exactly.
    """
    contrib, xt = model.contributions(car_frame)
    c = contrib.iloc[0]
    bias = c.pop("bias")
    # km and its "was imputed" flag describe one input -> report them together
    c["km__km"] = c["km__km"] + c.pop("km__km_was_imputed")

    order = c.abs().sort_values(ascending=False).index
    top, rest = order[:top_n], order[top_n:]
    rows = [{"reason": _reason(f, car_frame.iloc[0], xt.iloc[0]), "log_effect": c[f]} for f in top]
    rows.append({"reason": "All other features", "log_effect": c[rest].sum()})
    reasons = pd.DataFrame(rows)
    reasons["effect_pct"] = (reasons["log_effect"].map(math.exp) - 1) * 100
    return {
        "baseline": math.expm1(bias),
        "estimate": math.expm1(bias + c.sum()),
        "reasons": reasons[["reason", "effect_pct", "log_effect"]],
    }
