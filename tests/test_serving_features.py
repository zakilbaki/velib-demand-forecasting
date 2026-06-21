import pandas as pd

from src.serving.feature_builder import normalize_hour_timestamp


def test_normalize_hour_timestamp_converts_and_floors_to_utc() -> None:
    timestamp = normalize_hour_timestamp("2026-03-01T10:45:00+02:00")

    assert timestamp == pd.Timestamp("2026-03-01T08:00:00Z")


def test_normalize_hour_timestamp_treats_naive_values_as_utc() -> None:
    timestamp = normalize_hour_timestamp("2026-03-01 10:45:00")

    assert timestamp == pd.Timestamp("2026-03-01T10:00:00Z")
