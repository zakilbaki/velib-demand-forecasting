from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from src.db import get_db_connection


@dataclass
class StationPredictionFeatures:
    station_id: str
    hour_timestamp: pd.Timestamp
    free_bikes_current: float
    empty_slots_current: float
    hour_of_day: int
    is_weekend: bool
    latitude: float
    longitude: float
    free_bikes_t_minus_1: float
    free_bikes_t_minus_2: float
    free_bikes_t_minus_3: float
    delta_1h: float
    delta_3h: float


def normalize_hour_timestamp(hour_timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(hour_timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.floor("h")


def build_station_prediction_features(
    station_id: str,
    hour_timestamp,
    free_bikes_current: float,
    empty_slots_current: float,
) -> StationPredictionFeatures:
    normalized_hour = normalize_hour_timestamp(hour_timestamp)
    history_start = normalized_hour - timedelta(hours=3)
    required_hours = [
        normalized_hour - timedelta(hours=1),
        normalized_hour - timedelta(hours=2),
        normalized_hour - timedelta(hours=3),
    ]

    conn = get_db_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT latitude, longitude
                FROM stations
                WHERE station_id = %s
                """,
                (station_id,),
            )
            station_row = cur.fetchone()

            if station_row is None:
                raise LookupError(f"Station not found: {station_id}")

            latitude, longitude = station_row

            cur.execute(
                """
                WITH hourly_station_state AS (
                    SELECT DISTINCT ON (date_trunc('hour', ah."timestamp"))
                        date_trunc('hour', ah."timestamp") AS hour_timestamp,
                        ah.free_bikes
                    FROM availability_history ah
                    WHERE ah.station_id = %s
                      AND ah."timestamp" >= %s
                      AND ah."timestamp" < %s
                    ORDER BY
                        date_trunc('hour', ah."timestamp"),
                        ah."timestamp" DESC
                )
                SELECT hour_timestamp, free_bikes
                FROM hourly_station_state
                ORDER BY hour_timestamp
                """,
                (station_id, history_start.to_pydatetime(), normalized_hour.to_pydatetime()),
            )
            history_rows = cur.fetchall()
    finally:
        conn.close()

    hourly_history = {pd.Timestamp(row[0]): float(row[1]) for row in history_rows}
    missing_hours = [hour for hour in required_hours if hour not in hourly_history]

    if missing_hours:
        missing_str = ", ".join(str(hour) for hour in missing_hours)
        raise ValueError(
            f"Missing lag history for station {station_id} at hours: {missing_str}"
        )

    free_bikes_t_minus_1 = hourly_history[required_hours[0]]
    free_bikes_t_minus_2 = hourly_history[required_hours[1]]
    free_bikes_t_minus_3 = hourly_history[required_hours[2]]

    return StationPredictionFeatures(
        station_id=station_id,
        hour_timestamp=normalized_hour,
        free_bikes_current=float(free_bikes_current),
        empty_slots_current=float(empty_slots_current),
        hour_of_day=int(normalized_hour.hour),
        is_weekend=bool(normalized_hour.dayofweek >= 5),
        latitude=float(latitude),
        longitude=float(longitude),
        free_bikes_t_minus_1=free_bikes_t_minus_1,
        free_bikes_t_minus_2=free_bikes_t_minus_2,
        free_bikes_t_minus_3=free_bikes_t_minus_3,
        delta_1h=float(free_bikes_current) - free_bikes_t_minus_1,
        delta_3h=float(free_bikes_current) - free_bikes_t_minus_3,
    )
