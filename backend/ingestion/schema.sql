

CREATE TABLE races (
    id              SERIAL PRIMARY KEY,
    year            INTEGER NOT NULL,
    round           INTEGER NOT NULL,
    name            VARCHAR(100) NOT NULL,   -- "Bahrain Grand Prix"
    circuit         VARCHAR(100) NOT NULL,   -- "Sakhir" (FastF1's event Location field)
    country         VARCHAR(60) NOT NULL,
    
    race_date       DATE NOT NULL,
    total_laps      INTEGER,
    UNIQUE (year, round)
);

CREATE TABLE drivers (
    id              SERIAL PRIMARY KEY,
    code            VARCHAR(3) NOT NULL,     -- "VER"
    full_name       VARCHAR(100) NOT NULL,
    team            VARCHAR(60) NOT NULL,
    car_number      INTEGER,
    year            INTEGER NOT NULL,
    UNIQUE (code, year)
);

CREATE TABLE laps (
    id              SERIAL PRIMARY KEY,
    race_id         INTEGER REFERENCES races(id),
    driver_id       INTEGER REFERENCES drivers(id),
    lap_number      INTEGER NOT NULL,
    lap_time_ms     INTEGER,
    sector1_ms      INTEGER,
    sector2_ms      INTEGER,
    sector3_ms      INTEGER,
    compound        VARCHAR(10),
    tire_age        INTEGER,
    stint_number    INTEGER,
    position        INTEGER,
    is_personal_best BOOLEAN DEFAULT FALSE,
    track_status    VARCHAR(10),
    ambient_temp_c  FLOAT,   -- ADDED: Phase 2 needs this per lap
    track_temp_c    FLOAT,   -- ADDED: Phase 2 needs this per lap
    rainfall        BOOLEAN, -- ADDED: cheap to add now, useful as a limitations flag later
    UNIQUE (race_id, driver_id, lap_number)
);

CREATE TABLE pit_stops (
    id               SERIAL PRIMARY KEY,
    race_id          INTEGER REFERENCES races(id),
    driver_id        INTEGER REFERENCES drivers(id),
    lap_number       INTEGER NOT NULL,
    stop_duration_ms INTEGER,
    new_compound     VARCHAR(10),
    new_tire_age     INTEGER,
    UNIQUE (race_id, driver_id, lap_number)  -- ADDED: required for idempotent upsert
);

CREATE TABLE telemetry_samples (
    id           SERIAL PRIMARY KEY,
    race_id      INTEGER REFERENCES races(id),
    driver_id    INTEGER REFERENCES drivers(id),
    lap_number   INTEGER NOT NULL,
    sample_index INTEGER,
    speed        FLOAT,
    throttle     FLOAT,
    brake        BOOLEAN,
    gear         INTEGER,
    rpm          FLOAT,
    drs          INTEGER,
    x            FLOAT,
    y            FLOAT,
    distance     FLOAT
);