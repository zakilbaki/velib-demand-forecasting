import json
import shutil

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.regression_dataset import build_best_regression_training_dataframe
from src.modeling_config import (
    BEST_DATASET_VERSION,
    BEST_PARAMS,
    CONTINUOUS_BLOCK_END,
    CONTINUOUS_BLOCK_START,
    FEATURE_COLUMNS,
    FEATURE_SET_NAME,
    MODEL_ARTIFACT_NAME,
    MODEL_NAME,
    EXPERIMENT_NAME,
    SERVING_CANDIDATES_DIR,
    TARGET_COLUMN,
    TRAIN_END,
    VAL_END,
    configure_mlflow,
)


FINAL_EVALUATION_RUN_NAME = "gbr_final_test_evaluation_q1_to_mid_march"


def _write_serving_metadata(metadata_path, run_id: str, run_name: str) -> None:
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_name,
                "experiment_name": EXPERIMENT_NAME,
                "feature_set_name": FEATURE_SET_NAME,
                "model_name": MODEL_NAME,
            },
            indent=2,
        )
    )


def _export_candidate_bundle(model, run_id: str, run_name: str):
    candidate_dir = SERVING_CANDIDATES_DIR / run_id

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, path=str(candidate_dir))
    _write_serving_metadata(candidate_dir / "metadata.json", run_id, run_name)

    return candidate_dir


def evaluate_best_candidate_on_test() -> None:
    df = build_best_regression_training_dataframe()

    if df.empty:
        raise RuntimeError("Regression evaluation dataset is empty.")

    train_df = df[df["hour_timestamp"] < TRAIN_END]
    val_df = df[
        (df["hour_timestamp"] >= TRAIN_END)
        & (df["hour_timestamp"] < VAL_END)
    ]
    test_df = df[df["hour_timestamp"] >= VAL_END]

    dev_df = (
        pd.concat([train_df, val_df])
        .sort_values("hour_timestamp")
        .reset_index(drop=True)
    )
    test_df = test_df.sort_values("hour_timestamp").reset_index(drop=True)

    if dev_df.empty:
        raise RuntimeError("Development split is empty.")

    if test_df.empty:
        raise RuntimeError("Test split is empty.")

    X_dev = dev_df[FEATURE_COLUMNS]
    y_dev = dev_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = GradientBoostingRegressor(**BEST_PARAMS)

    configure_mlflow()

    with mlflow.start_run(run_name=FINAL_EVALUATION_RUN_NAME) as run:
        mlflow.log_param("dataset_version", BEST_DATASET_VERSION)
        mlflow.log_param("feature_set_name", FEATURE_SET_NAME)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("evaluation_stage", "final_test")
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("continuous_block_start", str(CONTINUOUS_BLOCK_START))
        mlflow.log_param("continuous_block_end", str(CONTINUOUS_BLOCK_END))
        mlflow.log_param("train_end", str(TRAIN_END))
        mlflow.log_param("val_end", str(VAL_END))
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("val_rows", len(val_df))
        mlflow.log_metric("dev_rows", len(dev_df))
        mlflow.log_metric("test_rows", len(test_df))

        for key, value in BEST_PARAMS.items():
            mlflow.log_param(key, value)

        model.fit(X_dev, y_dev)
        test_predictions = model.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)

        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("test_r2", float(test_r2))
        mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT_NAME)
        candidate_dir = _export_candidate_bundle(
            model=model,
            run_id=run.info.run_id,
            run_name=FINAL_EVALUATION_RUN_NAME,
        )

        print(f"Dataset rows: {len(df)}")
        print(f"Development rows: {len(dev_df)}")
        print(f"Test rows: {len(test_df)}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        print(f"Candidate bundle exported to: {candidate_dir}")


def main() -> None:
    evaluate_best_candidate_on_test()


if __name__ == "__main__":
    main()
