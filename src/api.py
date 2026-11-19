"""FastAPI service for the pricing model.

Run from the project root (no Docker needed):
    uvicorn src.api:app --port 8000
Then open http://127.0.0.1:8000/docs to try it in the browser.

Endpoints
    GET  /health   - is the service up, and which model is loaded
    GET  /options  - allowed values for every field (what the app's dropdowns offer)
    POST /predict  - estimate, price range, deal verdict and top reasons for one car

Every number comes from src/predict.py, the same code the Streamlit app uses,
so the app and the API cannot disagree (tests/test_api.py checks this).
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from src.config import CATEGORICAL_COLUMNS, LIST_COLUMNS, NUMERIC_COLUMNS
from src.predict import load_metadata, load_model, price_report

META = load_metadata()
OPTIONS = META["app"]
OPTIONAL_FIELDS = [c for c in CATEGORICAL_COLUMNS if c != "make_model"] + [
    c for c in NUMERIC_COLUMNS if c != "age"] + LIST_COLUMNS


# ---------------------------------------------------------------- request / response shapes
class CarRequest(BaseModel):
    """One car. Only make_model, age and km are required.

    Any other field you leave out is filled with the typical value for that model,
    exactly as the app pre-fills its form; the response lists which fields were filled.
    """

    model_config = ConfigDict(
        extra="forbid",  # a misspelt field name is an error, not silently ignored
        json_schema_extra={"examples": [{
            "make_model": "Audi A3", "age": 2, "km": 40000, "Fuel": "Diesel",
            "Gearing_Type": "Automatic", "asking_price": 19500,
        }]},
    )

    make_model: str = Field(description="One of the models in GET /options, e.g. 'Audi A3'")
    age: int = Field(description="Age in years, 0-3 (the only ages the model knows)")
    km: float = Field(description="Kilometres driven")
    asking_price: float | None = Field(default=None, gt=0, description="Optional: seller's asking price in euros")

    body_type: str | None = None
    vat: str | None = None
    Type: str | None = Field(default=None, description="Listing type, e.g. 'Used', 'New'")
    Fuel: str | None = None
    Paint_Type: str | None = None
    Upholstery_type: str | None = None
    Gearing_Type: str | None = None
    Drive_chain: str | None = None
    hp_kW: float | None = None
    Gears: float | None = None
    Previous_Owners: float | None = None
    Inspection_new: int | None = Field(default=None, ge=0, le=1)
    Displacement_cc: float | None = None
    Weight_kg: float | None = None
    cons_comb: float | None = None
    Comfort_Convenience: list[str] | None = None
    Entertainment_Media: list[str] | None = None
    Extras: list[str] | None = None
    Safety_Security: list[str] | None = None

    # Each field is checked on its own, so one bad request lists ALL its problems at once.
    # The limits are the same as the app's form: known categories, training-data ranges, known equipment.
    @field_validator(*CATEGORICAL_COLUMNS)
    @classmethod
    def known_category(cls, value, info: ValidationInfo):
        if value is None:
            return value
        allowed = sorted(OPTIONS["defaults"]) if info.field_name == "make_model" else OPTIONS["categories"][info.field_name]
        if value not in allowed:
            raise ValueError(f"'{value}' is not supported. Choose one of: {', '.join(allowed)}")
        return value

    @field_validator(*NUMERIC_COLUMNS, "km")
    @classmethod
    def within_training_range(cls, value, info: ValidationInfo):
        if value is None:
            return value
        lo, hi = OPTIONS["ranges"][info.field_name]
        if not lo <= value <= hi:
            raise ValueError(f"must be between {lo:g} and {hi:g} (the range seen in the training data)")
        return value

    @field_validator(*LIST_COLUMNS)
    @classmethod
    def known_equipment(cls, value, info: ValidationInfo):
        unknown = sorted(set(value or []) - set(OPTIONS["equipment"][info.field_name]))
        if unknown:
            raise ValueError(f"unknown items {unknown}. See GET /options for the allowed list")
        return value

    def to_car(self) -> tuple[dict, list[str]]:
        """Fill missing fields with this model's typical values. Returns (car, filled field names)."""
        defaults = OPTIONS["defaults"][self.make_model]
        car, filled = {"make_model": self.make_model, "age": self.age, "km": self.km}, []
        for col in OPTIONAL_FIELDS:
            value = getattr(self, col)
            if value is None:
                value = defaults["equipment"][col] if col in LIST_COLUMNS else defaults[col]
                filled.append(col)
            car[col] = value
        return car, filled


class PriceRange(BaseModel):
    low: float
    high: float
    low_pct: float = Field(description="Low end as % difference from the estimate")
    high_pct: float


class Deal(BaseModel):
    verdict: Literal["below range", "fair", "overpriced"]
    asking_price: float
    gap_pct: float = Field(description="Asking price vs estimate, in %")


class Reason(BaseModel):
    reason: str
    effect_pct: float = Field(description="How much this feature raises (+) or lowers (-) the price, in %")


class PredictResponse(BaseModel):
    estimate: float
    range: PriceRange
    high_price_warning: str | None
    deal: Deal | None
    baseline: float = Field(description="Model price for an average car in the data; reasons multiply from here")
    reasons: list[Reason]
    filled_fields: list[str] = Field(description="Fields you left out, filled with this model's typical values")
    model_trained_at: str


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # load once at start-up, so the first request is fast
    yield


app = FastAPI(
    title="Used Car Pricing Engine API",
    description="Listing-price estimate, likely range, deal check and reasons for used cars (Audi, Opel, Renault, 0-3 years).",
    version="3.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def readable_errors(request: Request, exc: RequestValidationError):
    """Turn Pydantic's error list into short 'field: message' lines."""
    problems = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p not in ("body",)) or "request"
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            msg = "unknown field. See http://127.0.0.1:8000/docs for the fields /predict accepts"
        elif err["type"] == "missing":
            msg = "this field is required"
        problems.append({"field": field, "message": msg})
    return JSONResponse(status_code=422, content={"error": "Invalid input", "problems": problems})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": META["selected_model"],
        "model_trained_at": META["trained_at"],
        "test_mape_pct": round(META["test_metrics"]["mape_pct"], 2),
    }


@app.get("/options")
def options():
    return {
        "make_model": sorted(OPTIONS["defaults"]),
        "categories": OPTIONS["categories"],
        "ranges": {**OPTIONS["ranges"], "age": [0, 3]},
        "equipment": OPTIONS["equipment"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: CarRequest):
    car, filled = request.to_car()
    report = price_report(car, asking_price=request.asking_price)
    return {**report, "filled_fields": filled, "model_trained_at": META["trained_at"]}
