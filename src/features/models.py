from dataclasses import dataclass
from datetime import datetime


@dataclass
class RegressionTrainingRow:
    station_id: str
    hour_timestamp: datetime
    free_bikes_current: int
    empty_slots_current: int
    latitude: float
    longitude: float
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    free_bikes_next_hour: int
