"""Project-wide settings: paths, column groups and modelling constants.

Keeping these in one place means the training script, the app and the tests
all agree on which columns exist and what each one is used for.
"""
from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Car_Price_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
PIPELINE_PATH = MODEL_DIR / "pricing_model.joblib"  # PricingModel: price, range, explanations
METADATA_PATH = MODEL_DIR / "metadata.json"

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
