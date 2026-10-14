"""FastAPI inference server: POST /predict returns consumption rate and distance-to-empty.

Serves the Ridge model selected in Phase 3 (best overall on CV and test). The client sends the
last >= 1 km of 10 Hz telemetry plus three numbers it tracks for the current trip.

Run with:  .venv/bin/uvicorn server.app:app --port 8000
Docs:      http://127.0.0.1:8000/docs
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.online import SAMPLE_FIELDS, features_now, implied_dte_km

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ridge.joblib"
CARD_PATH = ROOT / "models" / "tflite" / "model_card.json"
STATE: dict = {}


class Sample(BaseModel):
    """One 10 Hz telemetry sample (same units as the training data)."""
    time_s: float
    speed_kmh: float = Field(ge=0)
    accel_long_ms2: float
    elevation_m: float
    battery_voltage_v: float = Field(gt=0)
    battery_current_a: float = Field(description="negative = discharging")
    soc_pct: float = Field(ge=0, le=100)
    ambient_temp_c: float
    battery_temp_c: float
    heater_power_kw: float = Field(ge=0)
    aircon_power_kw: float


class TripContext(BaseModel):
    """Values the client tracks since the trip started."""
    elapsed_s: float = Field(gt=0, description="seconds since trip start")
    km_since_start: float = Field(ge=1.0, description="km driven since trip start (model needs >= 1 km)")
    soc_at_start_pct: float = Field(ge=0, le=100)
    soc_min_pct: float = Field(default=8.0, description="BMS lower SoC limit = 'empty'")


class PredictRequest(BaseModel):
    """Telemetry window (oldest first; the last sample is 'now') + trip context."""
    samples: list[Sample] = Field(min_length=2)
    trip: TripContext


class PredictResponse(BaseModel):
    """Predicted consumption over the next 5 km and the implied distance-to-empty."""
    rate_pct_soc_per_km: float
    dte_km: float
    baseline_rate_pct_soc_per_km: float
    baseline_dte_km: float
    usable_soc_pct: float
    model: str
    server_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and the baseline rate once at startup."""
    STATE["ridge"] = joblib.load(MODEL_PATH)
    STATE["baseline_rate"] = json.loads(CARD_PATH.read_text())["baseline_global_avg_rate"]
    yield
    STATE.clear()


app = FastAPI(title="EV distance-to-empty predictor", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "model_loaded": "ridge" in STATE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict the consumption rate over the next 5 km and the implied distance-to-empty."""
    t0 = time.perf_counter()
    samples = pd.DataFrame([s.model_dump() for s in req.samples])[SAMPLE_FIELDS]
    try:
        f = features_now(samples, req.trip.elapsed_s, req.trip.km_since_start, req.trip.soc_at_start_pct)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    rate = float(STATE["ridge"].predict(f.to_frame().T)[0])
    soc_now = float(samples["soc_pct"].iat[-1])
    base = STATE["baseline_rate"]
    return PredictResponse(
        rate_pct_soc_per_km=rate,
        dte_km=implied_dte_km(soc_now, rate, req.trip.soc_min_pct),
        baseline_rate_pct_soc_per_km=base,
        baseline_dte_km=implied_dte_km(soc_now, base, req.trip.soc_min_pct),
        usable_soc_pct=soc_now - req.trip.soc_min_pct,
        model="ridge (Phase 3 selection)",
        server_ms=(time.perf_counter() - t0) * 1000,
    )
