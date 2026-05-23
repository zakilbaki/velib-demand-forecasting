from pathlib import Path

import mlflow
import pandas as pd

from src.features.regression_dataset import (
    BEST_DATASET_VERSION,
    CONTINUOUS_BLOCK_END,
    CONTINUOUS_BLOCK_START,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
SERVING_MODEL_DIR = PROJECT_ROOT / "serving_models" / "current"
SERVING_MODEL_METADATA_PATH = SERVING_MODEL_DIR / "metadata.json"
EXPERIMENT_NAME = "velib-demand-forecasting"
MODEL_NAME = "GradientBoostingRegressor"
MODEL_ARTIFACT_NAME = "model"
FEATURE_SET_NAME = "core_plus_lat_lon_plus_lags"
FEATURE_COLUMNS = [
    "free_bikes_current",
    "empty_slots_current",
    "hour_of_day",
    "is_weekend",
    "latitude",
    "longitude",
    "free_bikes_t_minus_1",
    "free_bikes_t_minus_2",
    "free_bikes_t_minus_3",
    "delta_1h",
    "delta_3h",
]
TARGET_COLUMN = "free_bikes_next_hour"
TRAIN_END = pd.Timestamp("2026-03-11 00:00:00", tz="UTC")
VAL_END = pd.Timestamp("2026-03-13 00:00:00", tz="UTC")
BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "random_state": 42,
}


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


__all__ = [
    "BEST_DATASET_VERSION",
    "BEST_PARAMS",
    "CONTINUOUS_BLOCK_END",
    "CONTINUOUS_BLOCK_START",
    "EXPERIMENT_NAME",
    "FEATURE_COLUMNS",
    "FEATURE_SET_NAME",
    "MLFLOW_TRACKING_URI",
    "MODEL_ARTIFACT_NAME",
    "MODEL_NAME",
    "PROJECT_ROOT",
    "SERVING_MODEL_DIR",
    "SERVING_MODEL_METADATA_PATH",
    "TARGET_COLUMN",
    "TRAIN_END",
    "VAL_END",
    "configure_mlflow",
]
