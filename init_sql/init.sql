CREATE TABLE stations (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL
);

CREATE TABLE availability_history (
    station_id TEXT NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL,
    free_bikes SMALLINT NOT NULL CHECK (free_bikes >= 0),
    empty_slots SMALLINT NOT NULL CHECK (empty_slots >= 0),
    PRIMARY KEY (station_id, "timestamp"),
    FOREIGN KEY (station_id) REFERENCES stations (station_id)
);
