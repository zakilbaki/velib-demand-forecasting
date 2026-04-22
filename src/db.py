import os
from pathlib import Path

import psycopg2
from src.models import AvailabilityObservation, Station


def _load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        key, separator, value = line.partition("=")
        if not separator:
            continue

        os.environ.setdefault(key.strip(), value.strip())


def get_db_connection():
    _load_env_file()

    database = os.getenv("POSTGRES_DB")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    missing_vars = [
        var_name
        for var_name, value in (
            ("POSTGRES_DB", database),
            ("POSTGRES_USER", user),
            ("POSTGRES_PASSWORD", password),
        )
        if not value
    ]

    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        raise RuntimeError(
            f"Missing required database environment variables: {missing_vars_str}"
        )

    return psycopg2.connect(
        database=database,
        user=user,
        password=password,
        host=host,
        port=port,
    )


def upsert_stations(conn, stations: list[Station]) -> None:
    if not stations:
        return

    query = """
        INSERT INTO stations (station_id, name, latitude, longitude)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (station_id)
        DO UPDATE SET
            name = EXCLUDED.name,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude;
    """

    station_rows = [
        (station.station_id, station.name, station.latitude, station.longitude)
        for station in stations
    ]

    with conn.cursor() as cur:
        cur.executemany(query, station_rows)

    conn.commit()


def insert_availability_observations(
    conn, observations: list[AvailabilityObservation]
) -> None:
    if not observations:
        return

    query = """
        INSERT INTO availability_history (station_id, "timestamp", free_bikes, empty_slots)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (station_id, "timestamp")
        DO NOTHING;
    """

    observation_rows = [
        (
            observation.station_id,
            observation.timestamp,
            observation.free_bikes,
            observation.empty_slots,
        )
        for observation in observations
    ]

    with conn.cursor() as cur:
        cur.executemany(query, observation_rows)

    conn.commit()
