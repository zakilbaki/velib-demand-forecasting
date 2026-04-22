from datetime import datetime

from src.models import AvailabilityObservation, Station


def map_station_record(station_data: dict) -> tuple[Station, AvailabilityObservation]:
    station_id = station_data["id"]
    name = station_data["name"]
    latitude = station_data["latitude"]
    longitude = station_data["longitude"]
    timestamp_str = station_data["timestamp"]
    free_bikes = station_data["free_bikes"]
    empty_slots = station_data["empty_slots"]

    if timestamp_str.endswith("Z"):
        timestamp_str = timestamp_str[:-1]

    timestamp = datetime.fromisoformat(timestamp_str)

    station = Station(
        station_id=station_id,
        name=name,
        latitude=latitude,
        longitude=longitude,
    )

    observation = AvailabilityObservation(
        station_id=station_id,
        timestamp=timestamp,
        free_bikes=free_bikes,
        empty_slots=empty_slots,
    )

    return station, observation
