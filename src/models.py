from dataclasses import dataclass
from datetime import datetime


@dataclass
class Station:
    station_id: str
    name: str
    latitude: float
    longitude: float


@dataclass
class AvailabilityObservation:
    station_id: str
    timestamp: datetime
    free_bikes: int
    empty_slots: int
