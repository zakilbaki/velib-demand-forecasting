from src.models import AvailabilityObservation, Station


def is_valid_station(station: Station) -> bool:
    if not station.station_id:
        return False
    if not station.name:
        return False
    if station.latitude is None:
        return False
    if station.longitude is None:
        return False
    return True


def is_valid_availability_observation(observation: AvailabilityObservation) -> bool:
    if not observation.station_id:
        return False
    if observation.timestamp is None:
        return False
    if observation.free_bikes < 0:
        return False
    if observation.empty_slots < 0:
        return False
    return True
