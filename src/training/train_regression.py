from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.features.regression_dataset import build_best_regression_training_dataframe


TRAIN_END = pd.Timestamp("2026-03-11 00:00:00", tz="UTC")
VAL_END = pd.Timestamp("2026-03-13 00:00:00", tz="UTC")
EXPERIMENT_NAME = "velib-demand-forecasting"
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
BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 0.8,
    "random_state": 42,
}


def _configure_mlflow() -> None:
    tracking_path = Path.cwd() / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_path}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def train_best_candidate_with_cv() -> None:
    df = build_best_regression_training_dataframe()

    if df.empty:
        raise RuntimeError("Regression training dataset is empty.")

    train_df = df[df["hour_timestamp"] < TRAIN_END]
    val_df = df[
        (df["hour_timestamp"] >= TRAIN_END)
        & (df["hour_timestamp"] < VAL_END)
    ]

    dev_df = (
        pd.concat([train_df, val_df])
        .sort_values("hour_timestamp")
        .reset_index(drop=True)
    )

    X_dev = dev_df[FEATURE_COLUMNS]
    y_dev = dev_df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=3)

    _configure_mlflow()

    fold_rmse = []
    fold_mae = []
    fold_r2 = []

    with mlflow.start_run(run_name="gbr_cv_best_feature_set"):
        mlflow.log_param("dataset_version", "best_regression_training_dataframe")
        mlflow.log_param("feature_set_name", FEATURE_SET_NAME)
        mlflow.log_param("model_name", "GradientBoostingRegressor")
        mlflow.log_param("cv_type", "TimeSeriesSplit")
        mlflow.log_param("n_splits", 3)
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("train_end", str(TRAIN_END))
        mlflow.log_param("val_end", str(VAL_END))
        mlflow.log_metric("train_rows", len(train_df))
        mlflow.log_metric("val_rows", len(val_df))
        mlflow.log_metric("dev_rows", len(dev_df))

        for key, value in BEST_PARAMS.items():
            mlflow.log_param(key, value)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev), start=1):
            X_fold_train = X_dev.iloc[train_idx]
            y_fold_train = y_dev.iloc[train_idx]
            X_fold_val = X_dev.iloc[val_idx]
            y_fold_val = y_dev.iloc[val_idx]

            model = GradientBoostingRegressor(**BEST_PARAMS)
            model.fit(X_fold_train, y_fold_train)

            val_predictions = model.predict(X_fold_val)

            rmse = np.sqrt(mean_squared_error(y_fold_val, val_predictions))
            mae = mean_absolute_error(y_fold_val, val_predictions)
            r2 = r2_score(y_fold_val, val_predictions)

            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_r2.append(r2)

            mlflow.log_metric(f"fold_{fold}_rmse", rmse)
            mlflow.log_metric(f"fold_{fold}_mae", mae)
            mlflow.log_metric(f"fold_{fold}_r2", r2)

        mlflow.log_metric("cv_rmse_mean", float(np.mean(fold_rmse)))
        mlflow.log_metric("cv_mae_mean", float(np.mean(fold_mae)))
        mlflow.log_metric("cv_r2_mean", float(np.mean(fold_r2)))

        print(f"Training rows: {len(df)}")
        print(f"Development rows: {len(dev_df)}")
        print(f"Mean CV RMSE: {np.mean(fold_rmse):.4f}")
        print(f"Mean CV MAE: {np.mean(fold_mae):.4f}")
        print(f"Mean CV R2: {np.mean(fold_r2):.4f}")


def main() -> None:
    train_best_candidate_with_cv()


if __name__ == "__main__":
    main()
