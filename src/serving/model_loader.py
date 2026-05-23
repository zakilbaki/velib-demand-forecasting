import json
from dataclasses import dataclass

import mlflow.sklearn

from src.modeling_config import (
    EXPERIMENT_NAME,
    FEATURE_SET_NAME,
    SERVING_MODEL_DIR,
    SERVING_MODEL_METADATA_PATH,
)


@dataclass
class LoadedModel:
    model: object
    run_id: str
    run_name: str
    experiment_name: str
    feature_set_name: str


def load_latest_final_model() -> LoadedModel:
    if not SERVING_MODEL_DIR.exists():
        raise RuntimeError(
            "Serving model bundle not found. "
            "Run src.training.evaluate_regression first to export the current model."
        )

    model = mlflow.sklearn.load_model(str(SERVING_MODEL_DIR))

    metadata: dict[str, str] = {}
    if SERVING_MODEL_METADATA_PATH.exists():
        metadata = json.loads(SERVING_MODEL_METADATA_PATH.read_text())

    return LoadedModel(
        model=model,
        run_id=metadata.get("run_id", ""),
        run_name=metadata.get("run_name", ""),
        experiment_name=metadata.get("experiment_name", EXPERIMENT_NAME),
        feature_set_name=metadata.get("feature_set_name", FEATURE_SET_NAME),
    )
