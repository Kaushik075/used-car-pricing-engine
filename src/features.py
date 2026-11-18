"""Feature engineering as scikit-learn transformers.

Everything that LEARNS from data is written as a transformer with fit()/transform(),
so that inside cross-validation and the final fit it only ever sees training rows.
The same objects are saved inside the pipeline, so the app transforms a new car
exactly the way the training data was transformed.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    KM_COLUMN,
    LIST_COLUMNS,
    MIN_CATEGORY_COUNT,
    MIN_FLAG_SHARE,
    NUMERIC_COLUMNS,
)


# ---------------------------------------------------------------------------
# 1. Equipment lists -> 0/1 flags
# ---------------------------------------------------------------------------
class FeatureListBinarizer(BaseEstimator, TransformerMixin):
    """Turn columns like "ABS,Bluetooth,Alloy wheels" into one 0/1 column per item.

    fit()   learns which items are common enough (>= min_share of training rows).
    transform() writes a 1 if the car has that item, else 0. Items never seen in
    training, or too rare, are simply ignored.
    """

    def __init__(self, min_share=MIN_FLAG_SHARE):
        self.min_share = min_share

    @staticmethod
    def _split(value):
        if not isinstance(value, str) or not value.strip():
            return set()
        return {item.strip() for item in value.split(",") if item.strip()}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.vocabulary_ = {}
        for col in self.columns_:
            counts = pd.Series(
                [item for v in X[col] for item in self._split(v)], dtype="object"
            ).value_counts()
            keep = counts[counts / len(X) >= self.min_share].index
            self.vocabulary_[col] = sorted(keep)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        blocks = []
        for col in self.columns_:
            items = X[col].map(self._split)
            for token in self.vocabulary_[col]:
                blocks.append(items.map(lambda s, t=token: int(t in s)).to_numpy())
        return np.column_stack(blocks).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [f"{col}__{tok}" for col in self.columns_ for tok in self.vocabulary_[col]]
        )


# ---------------------------------------------------------------------------
# 2. km with imputed values -> fill with the median km of the car's Type
# ---------------------------------------------------------------------------
class KmImputer(BaseEstimator, TransformerMixin):
    """Repair km values that were filled in upstream.

    In the raw file ~1,000 km values have a fractional part (e.g. 1647.3626).
    Real odometers are whole numbers, and these exact values repeat hundreds of
    times, so they are averages that someone filled in - not real readings.

    We treat any fractional km (or missing km) as unknown and fill it with the
    median km of cars of the same Type (New / Used / ...) in the TRAINING rows.
    A second output column says whether the value was filled (1) or real (0).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        km = self._real_km(X[KM_COLUMN])
        self.type_medians_ = km.groupby(X["Type"]).median().to_dict()
        self.global_median_ = float(km.median())
        return self

    @staticmethod
    def _real_km(km):
        km = pd.to_numeric(km, errors="coerce")
        return km.where(km % 1 == 0)  # fractional -> NaN

    def transform(self, X):
        X = pd.DataFrame(X)
        km = self._real_km(X[KM_COLUMN])
        was_filled = km.isna().astype(float)
        fill = X["Type"].map(self.type_medians_).fillna(self.global_median_)
        km = km.fillna(fill).astype(float)
        return np.column_stack([km.to_numpy(), was_filled.to_numpy()])

    def get_feature_names_out(self, input_features=None):
        return np.array(["km", "km_was_imputed"])


# ---------------------------------------------------------------------------
# 3. Put it all together
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """All preprocessing in one object.

    - Categories -> one-hot. All levels are kept (no dummy dropped): with
      regularisation the redundant column does no harm, and dropping one just to
      lower VIF changes nothing about prediction quality. Levels with fewer than
      MIN_CATEGORY_COUNT training rows are pooled into one "infrequent" column,
      and unseen levels at prediction time go there too instead of crashing.
    - km -> repaired (see KmImputer), then scaled.
    - Other numbers -> scaled so Ridge/Lasso penalise every coefficient fairly.
    - Equipment lists -> 0/1 flags.
    """
    km_steps = Pipeline([("impute", KmImputer()), ("scale", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_COUNT,
                    sparse_output=False,
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("km", km_steps, [KM_COLUMN, "Type"]),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("equip", FeatureListBinarizer(), LIST_COLUMNS),
        ],
        verbose_feature_names_out=True,
    )


def build_linear_pipeline(regressor) -> TransformedTargetRegressor:
    """Preprocessing + model, trained on log(1 + price).

    TransformedTargetRegressor applies log1p to the target before fitting and
    expm1 to predictions, so callers always get prices in euros back.
    """
    return _log_target(Pipeline([("prep", build_preprocessor()), ("model", regressor)]))


# Version 1 name, kept so older code and tests still work.
build_pipeline = build_linear_pipeline


def _log_target(pipe) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


# ---------------------------------------------------------------------------
# 4. Tree models (Version 2)
# ---------------------------------------------------------------------------
# Price may only go DOWN as these go up. Names are the preprocessor's output names.
DECREASING_FEATURES = ("km__km", "num__age")


def build_tree_preprocessor() -> ColumnTransformer:
    """Preprocessing for gradient boosting.

    Differences from the linear version:
    - Categories become integer codes (OrdinalEncoder), not one-hot columns.
      Boosting splits on them natively, and each category stays ONE feature, so
      a per-car explanation says "Model: Renault Espace" instead of nine dummies.
      Rare levels (< MIN_CATEGORY_COUNT training rows) share one code; unseen
      levels become -1, which the models treat as missing.
    - No scaling: trees only care about the order of values.
    Output is a DataFrame so feature names travel with the data.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    min_frequency=MIN_CATEGORY_COUNT,
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("km", KmImputer(), [KM_COLUMN, "Type"]),
            ("num", "passthrough", NUMERIC_COLUMNS),
            ("equip", FeatureListBinarizer(), LIST_COLUMNS),
        ],
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")


class TreeModel(BaseEstimator, RegressorMixin):
    """Wraps a LightGBM or HistGradientBoosting regressor.

    Both libraries need two things that depend on the final column list, which is
    only known after preprocessing is fitted (the equipment flags are learned):
      1. which columns are categorical      -> the "cat__" columns
      2. monotonic constraints              -> price never rises with km or age
    This wrapper works them out at fit time from the column names.
    """

    def __init__(self, estimator=None, decreasing=DECREASING_FEATURES, monotone=True):
        self.estimator = estimator
        self.decreasing = decreasing
        self.monotone = monotone

    def fit(self, X, y):
        names = list(X.columns)
        cats = [i for i, n in enumerate(names) if n.startswith("cat__")]
        mono = [-1 if (self.monotone and n in self.decreasing) else 0 for n in names]
        est = clone(self.estimator)
        X_arr = X.to_numpy(dtype=float)
        constrained = any(mono)  # pass constraints only if one is active (quantile LightGBM rejects any)
        if type(est).__module__.startswith("lightgbm"):
            if constrained:
                est.set_params(monotone_constraints=mono)
            est.fit(X_arr, y, categorical_feature=cats)
        else:  # scikit-learn HistGradientBoostingRegressor
            est.set_params(categorical_features=cats, monotonic_cst=mono if constrained else None)
            est.fit(X_arr, y)
        self.estimator_ = est
        self.feature_names_in_ = np.array(names)
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X, dtype=float))


def build_tree_pipeline(estimator, monotone=True) -> TransformedTargetRegressor:
    """Tree preprocessing + boosting model, trained on log(1 + price) like the linear one."""
    return _log_target(
        Pipeline(
            [
                ("prep", build_tree_preprocessor()),
                ("model", TreeModel(estimator=estimator, monotone=monotone)),
            ]
        )
    )
