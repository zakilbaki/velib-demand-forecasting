from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.modeling_config import FEATURE_COLUMNS
from src.serving.feature_builder import build_station_prediction_features
from src.serving.model_loader import LoadedModel, load_latest_final_model
from src.serving.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    StationStatePredictionRequest,
    StationStatePredictionResponse,
)


loaded_model: LoadedModel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global loaded_model
    loaded_model = load_latest_final_model()
    yield


app = FastAPI(
    title="Velib Bike Availability Predictor",
    description="Predict next-hour free bike availability for a Velib station.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if loaded_model is None:
        return HealthResponse(status="degraded", model_loaded=False)

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_run_id=loaded_model.run_id,
        model_run_name=loaded_model.run_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    feature_frame = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    prediction = float(loaded_model.model.predict(feature_frame)[0])

    return PredictionResponse(
        predicted_free_bikes_next_hour=prediction,
        model_run_id=loaded_model.run_id,
        model_run_name=loaded_model.run_name,
        feature_set_name=loaded_model.feature_set_name,
    )


@app.post("/predict/station-state", response_model=StationStatePredictionResponse)
def predict_from_station_state(
    request: StationStatePredictionRequest,
) -> StationStatePredictionResponse:
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        features = build_station_prediction_features(
            station_id=request.station_id,
            hour_timestamp=request.hour_timestamp,
            free_bikes_current=request.free_bikes_current,
            empty_slots_current=request.empty_slots_current,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    feature_frame = pd.DataFrame(
        [
            {
                "free_bikes_current": features.free_bikes_current,
                "empty_slots_current": features.empty_slots_current,
                "hour_of_day": features.hour_of_day,
                "is_weekend": features.is_weekend,
                "latitude": features.latitude,
                "longitude": features.longitude,
                "free_bikes_t_minus_1": features.free_bikes_t_minus_1,
                "free_bikes_t_minus_2": features.free_bikes_t_minus_2,
                "free_bikes_t_minus_3": features.free_bikes_t_minus_3,
                "delta_1h": features.delta_1h,
                "delta_3h": features.delta_3h,
            }
        ]
    )[FEATURE_COLUMNS]

    prediction = float(loaded_model.model.predict(feature_frame)[0])

    return StationStatePredictionResponse(
        station_id=features.station_id,
        resolved_hour_timestamp=features.hour_timestamp.to_pydatetime(),
        predicted_free_bikes_next_hour=prediction,
        model_run_id=loaded_model.run_id,
        model_run_name=loaded_model.run_name,
        feature_set_name=loaded_model.feature_set_name,
    )
