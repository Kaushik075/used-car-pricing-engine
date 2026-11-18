"""The saved Version 2 model: a price, a price range and the reasons behind the price.

PricingModel bundles three fitted pipelines that share the same preprocessing design:
    point  - predicts the typical price (trained on log price)
    lower  - predicts the 10th percentile of price  (quantile loss)
    upper  - predicts the 90th percentile of price  (quantile loss)

Price range = Conformalized Quantile Regression (CQR):
    The raw 10%/90% quantile models give a range whose WIDTH adapts to each car
    (wider where the data is noisy), but nothing guarantees they really cover 80%.
    Conformal calibration fixes that: on out-of-fold training predictions we
    measure how far real prices fall outside the raw range, and widen (or narrow)
    every range by the amount needed to cover 80%. We do this separately per price
    band ("Mondrian" conformal), so coverage holds within each band, not only on average.
"""
import numpy as np
import pandas as pd

TARGET_COVERAGE = 0.80
QUANTILES = (0.10, 0.90)


# ---------------------------------------------------------------- conformal helpers
def conformity_scores(y, lo, hi) -> np.ndarray:
    """How far each real price is outside its raw range, on the log scale.

    Negative = inside (by how much), positive = outside (by how much).
    """
    z, lo, hi = np.log1p(y), np.log1p(lo), np.log1p(hi)
    return np.maximum(lo - z, z - hi)


def conformal_correction(scores, coverage=TARGET_COVERAGE) -> float:
    """The amount to add to both ends so that `coverage` of scores fall inside.

    Uses the finite-sample quantile level ceil((n+1)*coverage)/n from conformal theory.
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(scores, level, method="higher"))


def assign_band(prices, edges) -> np.ndarray:
    """Band index 0..len(edges) for each price (bands are training-price quintiles)."""
    return np.digitize(np.asarray(prices, dtype=float), edges)


class PricingModel:
    """Everything the app needs, in one object saved with joblib."""

    def __init__(self, point, lower, upper, band_edges, corrections):
        self.point = point
        self.lower = lower
        self.upper = upper
        self.band_edges = list(band_edges)  # price edges between bands
        self.corrections = list(corrections)  # one log-scale correction per band

    # ------------------------------------------------------------ predictions
    def predict(self, X) -> np.ndarray:
        return self.point.predict(X)

    def predict_range(self, X):
        """Return (estimate, low, high) arrays in euros."""
        est = self.point.predict(X)
        lo_raw, hi_raw = self.lower.predict(X), self.upper.predict(X)
        # quantile models are trained separately, so make sure low <= estimate <= high
        lo_raw, hi_raw = np.minimum(lo_raw, est), np.maximum(hi_raw, est)
        c = np.array(self.corrections)[assign_band(est, self.band_edges)]
        low = np.expm1(np.log1p(lo_raw) - c)
        high = np.expm1(np.log1p(hi_raw) + c)
        # a negative correction must never squeeze the range past the estimate
        return est, np.minimum(low, est), np.maximum(high, est)

    # ------------------------------------------------------------ explanations
    def contributions(self, X):
        """Per-feature contributions to log(1 + price), from LightGBM's built-in TreeSHAP.

        Returns (contributions DataFrame incl. a "bias" column, transformed features).

        For each car: bias + sum(contributions) = log(1 + estimated price), exactly.
        So exp(contribution) is the factor by which that feature multiplies the price.
        """
        pipe = self.point.regressor_
        Xt = pipe.named_steps["prep"].transform(X)
        booster = pipe.named_steps["model"].estimator_
        raw = booster.predict(Xt.to_numpy(dtype=float), pred_contrib=True)
        cols = list(Xt.columns) + ["bias"]
        return pd.DataFrame(raw, columns=cols, index=X.index), Xt
