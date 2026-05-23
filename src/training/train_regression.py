import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.features.regression_dataset import build_best_regression_training_dataframe
from src.modeling_config import (
    BEST_DATASET_VERSION,
    BEST_PARAMS,
    CONTINUOUS_BLOCK_END,
    CONTINUOUS_BLOCK_START,
    FEATURE_COLUMNS,
    FEATURE_SET_NAME,
    MODEL_NAME,
    TARGET_COLUMN,
    TRAIN_END,
    VAL_END,
    configure_mlflow,
)


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

    configure_mlflow()

    fold_rmse = []
    fold_mae = []
    fold_r2 = []

    with mlflow.start_run(run_name="gbr_cv_best_feature_set_q1_to_mid_march"):
        mlflow.log_param("dataset_version", BEST_DATASET_VERSION)
        mlflow.log_param("feature_set_name", FEATURE_SET_NAME)
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("cv_type", "TimeSeriesSplit")
        mlflow.log_param("n_splits", 3)
        mlflow.log_param("features", FEATURE_COLUMNS)
        mlflow.log_param("continuous_block_start", str(CONTINUOUS_BLOCK_START))
        mlflow.log_param("continuous_block_end", str(CONTINUOUS_BLOCK_END))
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

        print(f"Dataset rows: {len(df)}")
        print(f"Development rows: {len(dev_df)}")
        print(f"Mean CV RMSE: {np.mean(fold_rmse):.4f}")
        print(f"Mean CV MAE: {np.mean(fold_mae):.4f}")
        print(f"Mean CV R2: {np.mean(fold_r2):.4f}")


def main() -> None:
    train_best_candidate_with_cv()


if __name__ == "__main__":
    main()
