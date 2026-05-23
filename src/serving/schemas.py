from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    free_bikes_current: float = Field(ge=0)
    empty_slots_current: float = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: bool
    latitude: float
    longitude: float
    free_bikes_t_minus_1: float = Field(ge=0)
    free_bikes_t_minus_2: float = Field(ge=0)
    free_bikes_t_minus_3: float = Field(ge=0)
    delta_1h: float
    delta_3h: float


class PredictionResponse(BaseModel):
    predicted_free_bikes_next_hour: float
    model_run_id: str
    model_run_name: str
    feature_set_name: str


class StationStatePredictionRequest(BaseModel):
    station_id: str
    hour_timestamp: datetime
    free_bikes_current: float = Field(ge=0)
    empty_slots_current: float = Field(ge=0)


class StationStatePredictionResponse(PredictionResponse):
    station_id: str
    resolved_hour_timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_run_id: str | None = None
    model_run_name: str | None = None
