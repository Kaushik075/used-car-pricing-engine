"""Used Car Pricing Engine - Streamlit app (Version 2: LightGBM, model-based ranges, explanations).

Run locally from the project root:
    streamlit run streamlit_app.py

Layout: four tabs, no sidebar, so it reads top-to-bottom on a phone.
Columns stack automatically on narrow screens.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.config import LIST_COLUMNS
from src.predict import deal_check, depreciation_grid, estimate, explain, high_price_note, is_high_price, load_metadata, range_pct

st.set_page_config(page_title="Used Car Pricing Engine", page_icon="🚗", layout="centered")

meta = load_metadata()
app, rules = meta["app"], meta["rules"]
cats, rng = app["categories"], app["ranges"]
test = meta["test_metrics"]


def eur(x: float) -> str:
    return f"€{x:,.0f}"


def deal_bar_html(asking: float, low: float, high: float) -> str:
    """A small horizontal bar: below | fair (the likely range) | overpriced, with a marker.

    Colours are semi-transparent and text inherits the theme colour,
    so the bar reads well in both light and dark mode.
    """
    pad = 0.35 * (high - low)
    vmin, vmax = min(low, asking) - pad, max(high, asking) + pad

    def pos(v):
        return (v - vmin) / (vmax - vmin) * 100

    p_low, p_high, p_ask = pos(low), pos(high), pos(asking)
    # keep the marker label on screen near the edges
    shift = "0%" if p_ask < 12 else "-100%" if p_ask > 88 else "-50%"
    zone = "position:absolute;top:0;height:100%;display:flex;align-items:center;justify-content:center;font-size:0.72rem;"
    tick = "position:absolute;top:0;transform:translateX(-50%);font-size:0.75rem;opacity:0.8;white-space:nowrap;"
    return f"""
<div style="position:relative;margin:2.1rem 0 1.6rem 0;color:inherit;" role="img"
     aria-label="Asking price {eur(asking)} against likely range {eur(low)} to {eur(high)}">
  <div style="position:absolute;bottom:calc(100% + 6px);left:{p_ask:.2f}%;transform:translateX({shift});
              font-size:0.8rem;font-weight:600;white-space:nowrap;">Asking {eur(asking)}</div>
  <div style="position:relative;height:28px;border-radius:6px;overflow:hidden;">
    <div style="{zone}left:0;width:{p_low:.2f}%;background:rgba(27,175,122,0.28);">below</div>
    <div style="{zone}left:{p_low:.2f}%;width:{p_high - p_low:.2f}%;background:rgba(90,150,230,0.30);">fair</div>
    <div style="{zone}left:{p_high:.2f}%;right:0;background:rgba(227,73,72,0.28);">overpriced</div>
  </div>
  <div style="position:absolute;top:-4px;left:{p_ask:.2f}%;width:3px;height:36px;margin-left:-1.5px;
              background:currentColor;border-radius:2px;"></div>
  <div style="position:relative;height:1.1rem;margin-top:4px;">
    <span style="{tick}left:{p_low:.2f}%;">{eur(low)}</span>
    <span style="{tick}left:{p_high:.2f}%;">{eur(high)}</span>
  </div>
</div>"""


# Only CSS in the app: trim Streamlit's large default top padding, which wastes a phone's first screen.
st.markdown(
    "<style>.stMainBlockContainer, .block-container {padding-top: 2.5rem;}</style>",
    unsafe_allow_html=True,
)

st.title("Used Car Pricing Engine")
st.caption("Estimate a used car's market price and check whether an asking price is fair.")

tab_price, tab_why, tab_deal, tab_info = st.tabs(["Estimate", "Why", "Deal check", "How it works"])  # short labels fit a phone

# ---------------------------------------------------------------- Tab 1: inputs + estimate
with tab_price:
    result_slot = st.container()  # filled after the inputs, but shown above them

    make_model = st.selectbox("Make and model", sorted(app["defaults"]))
    d = app["defaults"][make_model]
    k = make_model  # widget keys include the model, so defaults reset when it changes

    def choice(label, col, where=st):
        opts = cats[col]
        default = d[col] if d[col] in opts else opts[0]
        return where.selectbox(label, opts, index=opts.index(default), key=f"{col}-{k}")

    def number(label, col, where=st, step=1.0, fmt="%.0f"):
        lo, hi = rng[col]
        return where.number_input(label, lo, hi, float(d[col]), step=step, format=fmt, key=f"{col}-{k}")

    # One st.columns per row, so on a phone (where columns stack) the order stays row by row.
    c1, c2 = st.columns(2)
    age = c1.select_slider("Age (years)", options=[0, 1, 2, 3], value=int(d["age"]), key=f"age-{k}")
    km = number("Kilometres", "km", c2, step=1000.0)
    c1, c2 = st.columns(2)
    hp_kw = number("Power (kW)", "hp_kW", c1)
    car_type = choice("Listing type", "Type", c2)
    c1, c2 = st.columns(2)
    fuel = choice("Fuel", "Fuel", c1)
    gearing = choice("Gearbox", "Gearing_Type", c2)

    with st.expander("More details"):
        e1, e2 = st.columns(2)
        body = choice("Body type", "body_type", e1)
        vat = choice("VAT", "vat", e2)
        gears = number("Gears", "Gears", e1)
        owners = number("Previous owners", "Previous_Owners", e2)
        cc = number("Engine size (cc)", "Displacement_cc", e1, step=10.0)
        weight = number("Weight (kg)", "Weight_kg", e2, step=10.0)
        cons = number("Fuel use (l/100 km)", "cons_comb", e1, step=0.1, fmt="%.1f")
        paint = choice("Paint", "Paint_Type", e2)
        upholstery = choice("Upholstery", "Upholstery_type", e1)
        drive = choice("Drive", "Drive_chain", e2)
        inspection = st.checkbox("New inspection", value=bool(d["Inspection_new"]), key=f"insp-{k}")

    equipment = {}
    with st.expander("Equipment"):
        st.caption("Pre-filled with what most cars of this model have.")
        for col in LIST_COLUMNS:
            equipment[col] = st.multiselect(
                col.replace("_", " & "), app["equipment"][col], default=d["equipment"][col], key=f"{col}-{k}"
            )

    car = {
        "make_model": make_model, "body_type": body, "vat": vat, "Type": car_type, "Fuel": fuel,
        "Paint_Type": paint, "Upholstery_type": upholstery, "Gearing_Type": gearing, "Drive_chain": drive,
        "km": km, "Gears": gears, "age": age, "Previous_Owners": owners, "hp_kW": hp_kw,
        "Inspection_new": int(inspection), "Displacement_cc": cc, "Weight_kg": weight, "cons_comb": cons,
        **equipment,
    }
    res = estimate(car)
    est, low, high = res["estimate"], res["low"], res["high"]
    lo_pct, hi_pct = range_pct(est, low, high)
    high_price = is_high_price(est, rules)
    # One warning text, built from THIS car's own range, used in every tab (and by the API).
    warning_text = high_price_note(est, low, high, rules)

    with result_slot:
        with st.container(border=True):
            st.metric("Estimated market price", eur(est))
            st.markdown(f"Likely range **{eur(low)} – {eur(high)}**")
            st.caption("About 8 in 10 comparable cars are listed within this range.")
            if high_price:
                st.warning(warning_text, icon="⚠️")

# ---------------------------------------------------------------- Tab 2: why this price
with tab_why:
    st.caption(f"For the car in *Estimate*: {make_model} · {age} yr · {km:,.0f} km · {fuel} · {gearing}")

    st.subheader("Top 5 reasons")
    ex = explain(car, top_n=5)
    lines = [f"Starting point: an average car in the data, **{eur(ex['baseline'])}**", ""]
    for r in ex["reasons"].itertuples():
        if abs(r.effect_pct) < 0.5:
            effect = "barely changes the price"
        elif r.effect_pct > 0:
            effect = f"raises the price by about **{r.effect_pct:.0f}%**"
        else:
            effect = f"lowers the price by about **{abs(r.effect_pct):.0f}%**"
        lines.append(f"- {r.reason}: {effect}")
    lines += ["", f"Result: **{eur(ex['estimate'])}**"]
    st.markdown("\n".join(lines))
    st.caption(
        "Each reason compares this car with the average car in the data. For example, a 1-year-old car "
        "is younger than average, so its age raises the price. Effects multiply together to give the "
        "result. Computed with SHAP values from the model itself."
    )

    st.subheader("Value by age and kilometres")
    grid = depreciation_grid(car, app["km_by_age"])
    grid["Age"] = grid["age"].map(lambda a: f"{a:.0f} yr" + ("" if a == 1 else "s"))
    this_car = pd.DataFrame({"km": [km], "price": [est], "label": ["This car"]})
    lines_chart = alt.Chart(grid).mark_line(strokeWidth=2).encode(
        x=alt.X("km:Q", title="Kilometres", axis=alt.Axis(format=",.0f")),
        y=alt.Y("price:Q", title="Estimated price (€)", axis=alt.Axis(format=",.0f"), scale=alt.Scale(zero=False)),
        # four clearly different hues in a fixed order (blue, orange, green, violet)
        color=alt.Color("Age:N", title=None, legend=alt.Legend(orient="top", columns=4, labelLimit=80),
                        scale=alt.Scale(range=["#2a78d6", "#eb6834", "#1baf7a", "#8a6ce0"])),
        tooltip=[alt.Tooltip("Age:N"), alt.Tooltip("km:Q", format=",.0f"), alt.Tooltip("price:Q", title="€", format=",.0f")],
    )
    point = alt.Chart(this_car).mark_point(size=110, filled=True, color="#888888", stroke="white", strokeWidth=1.5).encode(
        x="km:Q", y="price:Q", tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("price:Q", title="€", format=",.0f")]
    )
    st.altair_chart(lines_chart + point, width="stretch")
    st.caption(
        "Prices from this app's model, with everything else held as entered. Each line covers only "
        "the km range real cars of that age have in the data (ages 0–3 only). The dot is this car."
    )

# ---------------------------------------------------------------- Tab 3: deal check
with tab_deal:
    st.caption(f"For the car in *Estimate*: {make_model} · {age} yr · {km:,.0f} km · {fuel} · {gearing}")
    # value=None -> the box starts empty (and returns None) until a price is typed
    asking = st.number_input(
        "Seller's asking price (€)", min_value=1.0, value=None, step=250.0, format="%.0f",
        placeholder="Type the asking price",
    )

    with st.container(border=True):
        if asking is None:
            st.metric("Estimated market price", eur(est))
            st.markdown(f"Likely range **{eur(low)} – {eur(high)}**")
            st.caption("Enter an asking price above to see where it sits.")
        else:
            result = deal_check(asking, est, low, high)
            badge = {
                "below range": ":green-badge[Below range]",
                "fair": ":blue-badge[Fair price]",
                "overpriced": ":red-badge[Overpriced]",
            }[result["verdict"]]
            st.markdown(f"### {badge}")
            st.markdown(deal_bar_html(asking, low, high), unsafe_allow_html=True)
            direction = "below" if result["gap_pct"] < 0 else "above"
            st.markdown(
                f"Asking {eur(asking)} is **{abs(result['gap_pct']):.1f}% {direction}** the estimate of {eur(est)}."
            )
            advice = {
                "below range": "Cheaper than most comparable listings. Worth a look, but check why.",
                "fair": "Within the range where most comparable cars are listed.",
                "overpriced": "Above the range for comparable cars. There should be room to negotiate.",
            }[result["verdict"]]
            st.caption(advice)
            if high_price:
                st.caption(f"⚠️ {warning_text}")

# ---------------------------------------------------------------- Tab 4: how it works
with tab_info:
    lin = meta["test_metrics_linear"]
    cov_all = meta["test_range_coverage"][0]
    lin_band = {r["band"]: r for r in meta["test_errors_by_band"]["linear"]}
    gb_band = {r["band"]: r for r in meta["test_errors_by_band"]["boosting"]}
    cov_band = {r["band"]: r for r in meta["test_range_coverage"][1:]}
    band_rows = "\n".join(
        f"| {b} | {lin_band[b]['mape_pct']:.1f}% | {gb_band[b]['mape_pct']:.1f}% | {cov_band[b]['by_estimate']:.0%} |"
        for b in meta["band_labels"]
    )
    st.markdown(
        f"""
**What it predicts.** The *listing* (asking) price of a used car, not the final sale price.

**Data.** {meta['data']['clean_rows']:,} AutoScout24 listings after removing duplicates:
{len(app['defaults'])} models from Audi, Opel and Renault, all **0–3 years old**. Cars outside that range are guesses.

**Model.** LightGBM gradient boosting on log(price), chosen over linear models and scikit-learn's
HistGradientBoosting by 5-fold cross-validation on the training data. Price can only fall as km or age rise.

**Test results** on {test['n']:,} cars the model never saw:

| | Linear (v1) | LightGBM (v2) |
|---|---|---|
| Average error | {lin['mape_pct']:.1f}% | {test['mape_pct']:.1f}% |
| Average error (€) | {eur(lin['mae_eur'])} | {eur(test['mae_eur'])} |
| R² | {lin['r2']:.3f} | {test['r2']:.3f} |

**By price band** (test set):

| Real price | Linear error | LightGBM error | Real price inside range* |
|---|---|---|---|
{band_rows}

*Range coverage groups cars by estimated price, the only price the app knows. Overall: {cov_all['by_estimate']:.0%} (target 80%).

**Price range.** Two extra LightGBM models predict the 10th and 90th percentile of price for each car,
so the range is wider where the model is less sure. Conformal calibration on training data then adjusts
the width per price band, so about 8 in 10 real prices fall inside.

**Known limits.**
- The most expensive cars (above about €40k) are still usually under-estimated, though less than in v1.
- Accidents, service history and condition are not known to the model.
"""
    )
