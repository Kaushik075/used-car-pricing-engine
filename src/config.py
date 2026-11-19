"""Project-wide settings: paths, column groups and modelling constants.

Keeping these in one place means the training script, the app and the tests
all agree on which columns exist and what each one is used for.
"""
import os  # noqa: F401
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Car_Price_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
# File names inside a model folder (the live model is in MODEL_DIR; challengers get their own folder)
MODEL_FILE = "pricing_model.joblib"        # PricingModel: price, range, explanations
METADATA_FILE = "metadata.json"            # every result, rule and app option
PROFILE_FILE = "reference_profile.json"    # training-data summary used by the drift check
HASHES_FILE = "train_row_hashes.npy"       # fingerprints of training rows (no raw data)
PIPELINE_PATH = MODEL_DIR / MODEL_FILE
METADATA_PATH = MODEL_DIR / METADATA_FILE
CANDIDATES_DIR = MODEL_DIR / "candidates"  # challengers trained by the retraining pipeline
ARCHIVE_DIR = MODEL_DIR / "archive"        # previous champions, kept when a challenger is promoted
PROMOTION_LOG_PATH = MODEL_DIR / "promotion_log.jsonl"
DRIFT_REPORT_DIR = PROJECT_ROOT / "reports" / "drift"

# Experiment tracking: MLflow stores runs in a local SQLite file, artifacts in ./mlruns
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "used-car-pricing"

# ---------- Columns ----------
TARGET = "price"

# Comma-separated lists of equipment ("ABS,Bluetooth,...") -> split into 0/1 flags
LIST_COLUMNS = [
    "Comfort_Convenience",
    "Entertainment_Media",
    "Extras",
    "Safety_Security",
]

CATEGORICAL_COLUMNS = [
    "make_model",
    "body_type",
    "vat",
    "Type",
    "Fuel",
    "Paint_Type",
    "Upholstery_type",
    "Gearing_Type",
    "Drive_chain",
]

# km is handled separately because some of its values were imputed upstream
KM_COLUMN = "km"
NUMERIC_COLUMNS = [
    "Gears",
    "age",
    "Previous_Owners",
    "hp_kW",
    "Inspection_new",
    "Displacement_cc",
    "Weight_kg",
    "cons_comb",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + [KM_COLUMN] + NUMERIC_COLUMNS + LIST_COLUMNS

# ---------- Modelling constants ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# An equipment flag is kept only if at least 1% of TRAINING cars have it.
MIN_FLAG_SHARE = 0.01
# A category level with fewer than 20 TRAINING rows is grouped as "infrequent".
MIN_CATEGORY_COUNT = 20

# ---------- Promotion gate (src/retrain.py) ----------
GATE_MIN_IMPROVEMENT_PP = 0.05     # challenger MAPE must be at least 0.05 points lower
GATE_COVERAGE_RANGE = (0.75, 0.85) # challenger's price ranges must cover 75-85% (target 80%)
GATE_MIN_EVAL_ROWS = 500           # need this many rows NEITHER model trained on

# ---------- Drift check (src/drift.py) ----------
PSI_WATCH = 0.10                   # PSI 0.10-0.25: moderate shift, keep an eye on it
PSI_SHIFT = 0.25                   # PSI above 0.25: large shift
PRICE_MEDIAN_SHIFT_PCT = 5.0       # median price moved more than 5%
OUT_OF_RANGE_SHARE = 0.02          # more than 2% of rows outside what the model has seen
FEATURE_MEDIAN_SHIFT_PCT = 20.0    # a feature's median moved more than 20%: worth watching
