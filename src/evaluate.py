"""Metrics shared by training, notebooks and tests."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """R², MAE in euros and mean absolute percentage error (MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_eur": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(ape.mean() * 100),
        "median_ape_pct": float(np.median(ape) * 100),
        "n": int(len(y_true)),
    }


def error_by_band(y_true, y_pred, threshold: float) -> pd.DataFrame:
    """Compare errors for cars priced below vs above a threshold.

    The band is defined on the TRUE price, so this answers:
    "for cars that really cost more than X, how wrong are we?"
    """
    y_true = pd.Series(np.asarray(y_true, dtype=float))
    y_pred = pd.Series(np.asarray(y_pred, dtype=float))
    band = np.where(y_true > threshold, f"above {threshold:,.0f}", f"up to {threshold:,.0f}")
    rows = []
    for name in sorted(set(band)):
        mask = band == name
        m = regression_metrics(y_true[mask], y_pred[mask])
        rows.append({"band": name, **m})
    return pd.DataFrame(rows).set_index("band")
