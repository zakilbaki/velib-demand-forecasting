from datetime import timezone

from src.ingestion.mapper import map_station_record
from src.ingestion.validator import (
    is_valid_availability_observation,
    is_valid_station,
)
from src.models import AvailabilityObservation, Station


def test_map_station_record_preserves_utc_timezone() -> None:
    station, observation = map_station_record(
        {
            "id": "station-1",
            "name": "Test Station",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "timestamp": "2026-03-01T08:30:00Z",
            "free_bikes": 8,
            "empty_slots": 12,
        }
    )

    assert station.station_id == "station-1"
    assert observation.timestamp.tzinfo == timezone.utc


def test_station_validation_rejects_missing_identity() -> None:
    assert not is_valid_station(Station("", "Test Station", 48.8, 2.3))
    assert not is_valid_station(Station("station-1", "", 48.8, 2.3))


def test_observation_validation_rejects_negative_capacity() -> None:
    _, valid_observation = map_station_record(
        {
            "id": "station-1",
            "name": "Test Station",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "timestamp": "2026-03-01T08:30:00Z",
            "free_bikes": 8,
            "empty_slots": 12,
        }
    )
    invalid_observation = AvailabilityObservation(
        station_id=valid_observation.station_id,
        timestamp=valid_observation.timestamp,
        free_bikes=-1,
        empty_slots=valid_observation.empty_slots,
    )

    assert is_valid_availability_observation(valid_observation)
    assert not is_valid_availability_observation(invalid_observation)
