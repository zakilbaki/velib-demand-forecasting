from dataclasses import asdict

import pandas as pd

from src.db import get_db_connection
from src.features.models import RegressionTrainingRow


CONTINUOUS_BLOCK_START = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")
CONTINUOUS_BLOCK_END = pd.Timestamp("2026-03-16 00:00:00", tz="UTC")


REGRESSION_DATASET_QUERY = """
WITH hourly_station_state AS (
    SELECT DISTINCT ON (station_id, date_trunc('hour', "timestamp"))
        ah.station_id,
        date_trunc('hour', "timestamp") AS hour_timestamp,
        ah.free_bikes,
        ah.empty_slots,
        s.latitude,
        s.longitude
    FROM availability_history ah
    JOIN stations s
        ON ah.station_id = s.station_id
    ORDER BY
        ah.station_id,
        date_trunc('hour', ah."timestamp"),
        ah."timestamp" DESC
), training_base AS (
    SELECT
        station_id,
        hour_timestamp,
        free_bikes AS free_bikes_current,
        empty_slots AS empty_slots_current,
        latitude,
        longitude,
        EXTRACT(HOUR FROM hour_timestamp) AS hour_of_day,
        EXTRACT(DOW FROM hour_timestamp) AS day_of_week,
        EXTRACT(DOW FROM hour_timestamp) IN (0, 6) AS is_weekend,
        LEAD(hour_timestamp) OVER (
            PARTITION BY station_id
            ORDER BY hour_timestamp
        ) AS next_hour_timestamp,
        LEAD(free_bikes) OVER (
            PARTITION BY station_id
            ORDER BY hour_timestamp
        ) AS free_bikes_next_hour
    FROM hourly_station_state
)
SELECT
    station_id,
    hour_timestamp,
    free_bikes_current,
    empty_slots_current,
    latitude,
    longitude,
    hour_of_day,
    day_of_week,
    is_weekend,
    free_bikes_next_hour
FROM training_base
WHERE free_bikes_next_hour IS NOT NULL
  AND next_hour_timestamp = hour_timestamp + INTERVAL '1 hour'
ORDER BY station_id, hour_timestamp;
"""


def fetch_regression_training_rows() -> list[RegressionTrainingRow]:
    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(REGRESSION_DATASET_QUERY)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        RegressionTrainingRow(
            station_id=row[0],
            hour_timestamp=row[1],
            free_bikes_current=row[2],
            empty_slots_current=row[3],
            latitude=float(row[4]),
            longitude=float(row[5]),
            hour_of_day=int(row[6]),
            day_of_week=int(row[7]),
            is_weekend=row[8],
            free_bikes_next_hour=row[9],
        )
        for row in rows
    ]


def build_regression_training_dataframe() -> pd.DataFrame:
    rows = fetch_regression_training_rows()
    return pd.DataFrame([asdict(row) for row in rows])


def build_continuous_regression_training_dataframe(
    start: pd.Timestamp = CONTINUOUS_BLOCK_START,
    end: pd.Timestamp = CONTINUOUS_BLOCK_END,
) -> pd.DataFrame:
    df = build_regression_training_dataframe()

    return df[
        (df["hour_timestamp"] >= start)
        & (df["hour_timestamp"] < end)
    ].copy()


def add_free_bike_lag_features(
    df: pd.DataFrame,
    lag_hours: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    lagged_df = df.sort_values(["station_id", "hour_timestamp"]).reset_index(drop=True).copy()

    for lag in lag_hours:
        lag_column = f"free_bikes_t_minus_{lag}"
        lagged_df[lag_column] = (
            lagged_df.groupby("station_id")["free_bikes_current"].shift(lag)
        )

    lagged_df["delta_1h"] = (
        lagged_df["free_bikes_current"] - lagged_df["free_bikes_t_minus_1"]
    )
    lagged_df["delta_3h"] = (
        lagged_df["free_bikes_current"] - lagged_df["free_bikes_t_minus_3"]
    )

    return lagged_df


def build_best_regression_training_dataframe() -> pd.DataFrame:
    continuous_df = build_continuous_regression_training_dataframe()
    lagged_df = add_free_bike_lag_features(continuous_df)

    return lagged_df.dropna(
        subset=[
            "free_bikes_t_minus_1",
            "free_bikes_t_minus_2",
            "free_bikes_t_minus_3",
            "delta_1h",
            "delta_3h",
        ]
    ).copy()
