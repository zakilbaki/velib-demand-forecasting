from src.db import (
    get_db_connection,
    insert_availability_observations,
    upsert_stations,
)

from src.ingestion.client import fetch_velib_data
from src.ingestion.mapper import map_station_record
from src.ingestion.validator import (
    is_valid_availability_observation,
    is_valid_station,
)


def run_ingestion_pipeline() -> dict:
    data = fetch_velib_data()
    raw_stations = data["network"]["stations"]

    valid_stations = []
    valid_observations = []
    invalid_records = []

    for station_data in raw_stations:
        try:
            station, observation = map_station_record(station_data)
        except (KeyError, TypeError, ValueError) as exc:
            invalid_records.append(
                {
                    "station_data": station_data,
                    "error": str(exc),
                }
            )
            continue

        if is_valid_station(station) and is_valid_availability_observation(observation):
            valid_stations.append(station)
            valid_observations.append(observation)
        else:
            invalid_records.append(
                {
                    "station_data": station_data,
                    "error": "validation_failed",
                }
            )

    conn = get_db_connection()
    try:
        upsert_stations(conn, valid_stations)
        insert_availability_observations(conn, valid_observations)
    finally:
        conn.close()

    summary = {
        "total_raw_stations": len(raw_stations),
        "valid_stations": valid_stations,
        "valid_observations": valid_observations,
        "invalid_records": invalid_records
    }

    print(f"Total stations recuperees: {summary['total_raw_stations']}")
    print(f"Stations valides: {len(valid_stations)}")
    print(f"Observations valides: {len(valid_observations)}")
    print(f"Stations invalides: {len(invalid_records)}")

    return summary
