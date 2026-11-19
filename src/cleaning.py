"""Dataset-level cleaning: steps that happen once, before the train/test split.

Only rules that do NOT learn anything from the data belong here.
Anything that learns a value (a median, a list of frequent categories)
lives inside the scikit-learn pipeline in features.py, so it is learned
from training rows only.
"""
import numpy as np  # noqa: F401  (type hint below)
import pandas as pd

from src.config import DATA_PATH, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as delivered."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw listings.

    1. Strip stray spaces from text columns.
    2. Drop exact duplicate rows. This is done BEFORE splitting: otherwise the
       same listing could land in both train and test and inflate the score.
    3. Drop rows without a price (none today, but the training code relies on it).
    """
    out = df.copy()
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].str.strip()

    out = out.drop_duplicates().reset_index(drop=True)
    out = out.dropna(subset=[TARGET]).reset_index(drop=True)
    return out


def cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Small summary used by the EDA notebook and README."""
    return {
        "raw_rows": int(len(raw)),
        "exact_duplicates": int(raw.duplicated().sum()),
        "clean_rows": int(len(cleaned)),
        "missing_values": int(cleaned.isna().sum().sum()),
    }


def row_hashes(df: pd.DataFrame) -> "np.ndarray":
    """One 64-bit fingerprint per row (same values -> same fingerprint).

    Saved with each model so the retraining pipeline can tell which rows a model
    was trained on, without storing any raw data.
    """
    cols = sorted(df.columns)
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
