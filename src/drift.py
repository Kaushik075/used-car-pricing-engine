"""Drift check: has new data moved away from what the model was trained on?

Run from the project root:
    python -m src.drift --new path\\to\\new_listings.csv

How it works (simple and explainable):
- At training time we save a *reference profile* of the training rows: for every
  feature, the share of cars in each bin (numbers) or each level (categories),
  plus price levels. Only these aggregates are stored, never rows.
- For a new CSV we compute the same shares and compare them with the
  Population Stability Index (PSI):
      PSI = sum over bins of (new% - ref%) * ln(new% / ref%)
  Rule of thumb: below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 large shift.
- We also check the median price, how many rows are outside what the model has
  seen (e.g. a 6-year-old car, or a model it never learned), and big moves in a
  feature's median (PSI can miss those when many values share one bin).

Price-level shifts and out-of-range rows trigger RETRAIN RECOMMENDED. Feature shifts
inside the training range go on the WATCH list: the model already knows how, say,
km affects price, so more km alone does not make it wrong.

The verdict is RETRAIN RECOMMENDED, WATCH or OK, with the reasons listed.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLUMNS,
    DRIFT_REPORT_DIR,
    FEATURE_COLUMNS,
    FEATURE_MEDIAN_SHIFT_PCT,
    LIST_COLUMNS,
    MODEL_DIR,
    OUT_OF_RANGE_SHARE,
    PRICE_MEDIAN_SHIFT_PCT,
    PROFILE_FILE,
    PSI_SHIFT,
    PSI_WATCH,
    TARGET,
)

NUMERIC_DRIFT = ["km", "age", "hp_kW", "Weight_kg", "Displacement_cc", "cons_comb", "Gears", "Previous_Owners"]
DISCRETE_MAX_LEVELS = 12  # numbers with this few distinct values are compared value by value
EPS = 1e-4                # stands in for an empty bin, so the log in PSI stays finite


# ---------------------------------------------------------------- profile (saved at training time)
def _numeric_profile(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    prof = {"min": float(v.min()), "max": float(v.max()), "median": float(v.median()), "mean": float(v.mean())}
    if v.nunique() <= DISCRETE_MAX_LEVELS:
        shares = v.value_counts(normalize=True)
        prof["kind"] = "discrete"
        prof["shares"] = {str(float(k)): float(s) for k, s in shares.items()}
    else:
        edges = np.unique(np.quantile(v, np.linspace(0.1, 0.9, 9)))
        prof["kind"] = "binned"
        prof["edges"] = [float(e) for e in edges]
        prof["shares"] = _bin_shares(v, edges).tolist()
    return prof


def _bin_shares(values, edges) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, float), bins=np.concatenate([[-np.inf], edges, [np.inf]]))
    return counts / max(counts.sum(), 1)


def _equipment_count(df: pd.DataFrame) -> pd.Series:
    def n_items(x):
        return 0 if not isinstance(x, str) or not x.strip() else len([i for i in x.split(",") if i.strip()])
    return sum(df[c].map(n_items) for c in LIST_COLUMNS)


def build_reference_profile(X: pd.DataFrame, y: pd.Series) -> dict:
    """Aggregate summary of the TRAINING rows. Used later to judge new data."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(X)),
        "numeric": {c: _numeric_profile(X[c]) for c in NUMERIC_DRIFT},
        "categorical": {c: {str(k): float(v) for k, v in X[c].value_counts(normalize=True).items()}
                        for c in CATEGORICAL_COLUMNS},
        "price": _numeric_profile(y),
        "equipment_items": _numeric_profile(_equipment_count(X)),
    }


# ---------------------------------------------------------------- comparison
def psi(ref_shares, new_shares) -> float:
    ref = np.clip(np.asarray(ref_shares, float), EPS, None)
    new = np.clip(np.asarray(new_shares, float), EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def _compare_numeric(ref: dict, values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if ref["kind"] == "discrete":
        levels = list(ref["shares"])
        new = v.map(lambda x: str(float(x))).value_counts(normalize=True)
        unseen = float(new[[k for k in new.index if k not in levels]].sum())
        score = psi([ref["shares"][k] for k in levels] + [0.0], [new.get(k, 0.0) for k in levels] + [unseen])
    else:
        score = psi(ref["shares"], _bin_shares(v, np.array(ref["edges"])))
    return {"psi": score, "ref_median": ref["median"], "new_median": float(v.median()),
            "outside_training_range": float(((v < ref["min"]) | (v > ref["max"])).mean())}


def _compare_categorical(ref_shares: dict, values: pd.Series) -> dict:
    new = values.astype(str).value_counts(normalize=True)
    levels = list(ref_shares)
    unseen = float(new[[k for k in new.index if k not in levels]].sum())
    score = psi([ref_shares[k] for k in levels] + [0.0], [new.get(k, 0.0) for k in levels] + [unseen])
    top_ref = max(ref_shares, key=ref_shares.get)
    return {"psi": score, "unseen_share": unseen, "most_common_ref": top_ref,
            "share_ref": ref_shares[top_ref], "share_new": float(new.get(top_ref, 0.0))}


def status(score: float) -> str:
    return "large shift" if score > PSI_SHIFT else "moderate shift" if score > PSI_WATCH else "stable"


def check_drift(new_raw: pd.DataFrame, profile: dict) -> dict:
    """Compare new listings with the training reference profile. Returns a report dict."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in new_raw.columns]
    if missing:
        raise ValueError(f"New data is missing columns the model needs: {missing}")
    df = new_raw.drop_duplicates()

    features = {}
    for c in NUMERIC_DRIFT:
        features[c] = {"type": "number", **_compare_numeric(profile["numeric"][c], df[c])}
    for c in CATEGORICAL_COLUMNS:
        features[c] = {"type": "category", **_compare_categorical(profile["categorical"][c], df[c])}
    features["equipment_items"] = {"type": "number", **_compare_numeric(profile["equipment_items"], _equipment_count(df))}
    for f in features.values():
        f["status"] = status(f["psi"])

    price = _compare_numeric(profile["price"], df[TARGET])
    price["median_shift_pct"] = (price["new_median"] / price["ref_median"] - 1) * 100
    price["status"] = status(price["psi"])

    # Rows the model cannot really price: age/number outside the training range, or an unseen category
    out = pd.Series(False, index=df.index)
    for c in NUMERIC_DRIFT:
        r = profile["numeric"][c]
        v = pd.to_numeric(df[c], errors="coerce")
        out |= (v < r["min"]) | (v > r["max"])
    for c in CATEGORICAL_COLUMNS:
        out |= ~df[c].astype(str).isin(list(profile["categorical"][c]))
    out_share = float(out.mean())

    reasons, watch = [], []
    if price["psi"] > PSI_SHIFT:
        reasons.append(f"price distribution changed a lot (PSI {price['psi']:.2f})")
    if abs(price["median_shift_pct"]) > PRICE_MEDIAN_SHIFT_PCT:
        reasons.append(f"median price moved {price['median_shift_pct']:+.1f}% "
                       f"(€{price['ref_median']:,.0f} → €{price['new_median']:,.0f})")
    for name, f in features.items():
        if f["psi"] > PSI_SHIFT:
            reasons.append(f"{name} shifted a lot (PSI {f['psi']:.2f})")
        elif f["psi"] > PSI_WATCH:
            watch.append(f"{name} moderate shift (PSI {f['psi']:.2f})")
        # PSI can miss a shift when many values sit in one bin (e.g. lots of near-zero km),
        # so a big move in a feature's median is also worth a look.
        if f["type"] == "number" and f["ref_median"] > 0:
            move = (f["new_median"] / f["ref_median"] - 1) * 100
            if abs(move) > FEATURE_MEDIAN_SHIFT_PCT:
                watch.append(f"{name} median moved {move:+.0f}% ({f['ref_median']:,.0f} → {f['new_median']:,.0f})")
    if out_share > OUT_OF_RANGE_SHARE:
        reasons.append(f"{out_share:.1%} of rows are outside what the model has seen")
    if PSI_WATCH < price["psi"] <= PSI_SHIFT:
        watch.append(f"price moderate shift (PSI {price['psi']:.2f})")

    verdict = "RETRAIN RECOMMENDED" if reasons else "WATCH" if watch else "OK"
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_new_rows": int(len(df)),
        "n_reference_rows": profile["n_rows"],
        "verdict": verdict,
        "reasons": reasons,
        "watch": watch,
        "price": price,
        "out_of_range_share": out_share,
        "features": features,
    }


def to_markdown(report: dict, new_name: str) -> str:
    p = report["price"]
    lines = [
        f"# Drift report: {new_name}",
        "",
        f"Checked {report['checked_at']} · {report['n_new_rows']:,} new rows vs "
        f"{report['n_reference_rows']:,} training rows",
        "",
        f"## Verdict: **{report['verdict']}**",
        "",
    ]
    lines += [f"- {r}" for r in report["reasons"]] or ["- No large shifts found."]
    if report["watch"]:
        lines += ["", "To keep an eye on:"] + [f"- {w}" for w in report["watch"]]
    lines += [
        "",
        "## Price",
        "",
        f"Median €{p['ref_median']:,.0f} (training) → €{p['new_median']:,.0f} (new), {p['median_shift_pct']:+.1f}%; "
        f"PSI {p['psi']:.3f} ({p['status']}).",
        f"Rows outside what the model has seen: {report['out_of_range_share']:.1%}.",
        "",
        "## Features (PSI: below 0.10 stable, 0.10–0.25 moderate, above 0.25 large)",
        "",
        "| Feature | PSI | Status | Training | New |",
        "|---|---|---|---|---|",
    ]
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        if f["type"] == "number":
            ref, new = f"median {f['ref_median']:,.1f}", f"median {f['new_median']:,.1f}"
        else:
            ref = f"{f['most_common_ref']}: {f['share_ref']:.0%}"
            new = f"{f['most_common_ref']}: {f['share_new']:.0%}" + (
                f" (+{f['unseen_share']:.0%} unseen levels)" if f["unseen_share"] else "")
        lines.append(f"| {name} | {f['psi']:.3f} | {f['status']} | {ref} | {new} |")
    lines += ["", "PSI = Σ (new% − training%) × ln(new% / training%) over bins or categories."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare a new CSV with the training data.")
    parser.add_argument("--new", required=True, help="CSV with new listings (same columns as the training data)")
    parser.add_argument("--profile", default=str(MODEL_DIR / PROFILE_FILE), help="reference profile of the training data")
    parser.add_argument("--out", default=None, help="where to write the Markdown report")
    args = parser.parse_args()

    profile = json.loads(Path(args.profile).read_text())
    report = check_drift(pd.read_csv(args.new), profile)
    out = Path(args.out) if args.out else DRIFT_REPORT_DIR / f"drift_{Path(args.new).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(report, Path(args.new).name), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))

    print(f"Verdict: {report['verdict']}")
    for r in report["reasons"]:
        print("  -", r)
    for w in report["watch"]:
        print("  (watch)", w)
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
