"""
backend/ingestion/ingest.py

Ingests one race session (laps, pit stops, drivers, race metadata, and
optionally downsampled telemetry) from FastF1 into Postgres.

Usage:
    uv run python ingestion/ingest.py --year 2023 --round 1
    uv run python ingestion/ingest.py --year 2023 --round 1 --with-telemetry

Design notes (read before changing this file):

- Identifies races by ROUND NUMBER, not name string. FastF1 will fuzzy-match
  a name like "Britain" against the event schedule, which is fragile and a
  bad source of silent bugs. Round numbers are stable and unambiguous.

- Every insert is an upsert (ON CONFLICT ... DO UPDATE), per the
  architecture doc's idempotency requirement: re-running this for a race
  you've already loaded must be a no-op / safe refresh, not a duplicate
  insert. This is why schema.sql has UNIQUE constraints on every table
  that gets inserted into here (yes, including pit_stops -- see the
  comment in schema.sql about why that constraint was added).

- Telemetry is OFF by default (--with-telemetry to enable). It's the
  most expensive part of ingestion by a wide margin and Phase 1/Phase 2
  don't need it. Backfill it later, per race, only when Phase 8 (CV
  racing line work) actually needs it. Don't pay for data you don't need
  yet.

- Column names below were verified against FastF1's documented
  session.laps / session.results / session.weather_data schemas at the
  time this was written. Library APIs do drift between versions --
  if anything here raises a KeyError, the fix is `print(df.columns)`
  and adjust, not assume the script is wrong.
"""
import argparse
import os

import fastf1
import pandas as pd
from dotenv import load_dotenv
from fastf1.ergast import Ergast
from sqlalchemy import create_engine, text


ERGAST_TO_F1 = {
    "max_verstappen": "VER",
    "hamilton": "HAM",
    "leclerc": "LEC",
    "sainz": "SAI",
    "russell": "RUS",
    "norris": "NOR",
    "perez": "PER",
    "alonso": "ALO",
    "stroll": "STR",
    "bottas": "BOT",
    "albon": "ALB",
    "ocon": "OCO",
    "tsunoda": "TSU",
    "gasly": "GAS",
    "hulkenberg": "HUL",
    "zhou": "ZHO",
    "kevin_magnussen": "MAG",
    "sargeant": "SAR",
    "de_vries": "DEV"
}

def upsert_race(conn, session, year: int) -> int:
    ev = session.event
    round_number = int(ev["RoundNumber"])
    total_laps = int(session.laps["LapNumber"].max())  # derived from real data, not metadata

    result = conn.execute(
        text(
            """
            INSERT INTO races (year, round, name, circuit, country, race_date, total_laps)
            VALUES (:year, :round, :name, :circuit, :country, :race_date, :total_laps)
            ON CONFLICT (year, round) DO UPDATE SET
                total_laps = EXCLUDED.total_laps
            RETURNING id
            """
        ),
        {
            "year": year,
            "round": round_number,
            "name": ev["EventName"],
            "circuit": ev["Location"],
            "country": ev["Country"],
            "race_date": ev["EventDate"].date(),
            "total_laps": total_laps,
        },
    )
    return result.scalar_one()


def upsert_drivers(conn, session, year: int) -> dict[str, int]:
    driver_map: dict[str, int] = {}
    for _, row in session.results.iterrows():
        code = row["Abbreviation"]
        result = conn.execute(
            text(
                """
                INSERT INTO drivers (code, full_name, team, car_number, year)
                VALUES (:code, :full_name, :team, :car_number, :year)
                ON CONFLICT (code, year) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    team = EXCLUDED.team,
                    car_number = EXCLUDED.car_number
                RETURNING id
                """
            ),
            {
                "code": code,
                "full_name": row["FullName"],
                "team": row["TeamName"],
                "car_number": int(row["DriverNumber"]),
                "year": year,
            },
        )
        driver_map[code] = result.scalar_one()
    return driver_map


def _ms(td) -> int | None:
    """Convert a pandas Timedelta to integer milliseconds, NaT -> None."""
    return int(td.total_seconds() * 1000) if pd.notnull(td) else None


def insert_laps(conn, session, race_id: int, driver_map: dict[str, int]) -> int:
    laps_df = session.laps.reset_index(drop=True)

    # session.laps.get_weather_data() returns one weather sample per lap,
    # already time-matched -- row order matches laps_df exactly, so a
    # straight positional concat is safe here (confirmed against FastF1's
    # own documented usage pattern).
    weather_df = laps_df.get_weather_data().reset_index(drop=True)
    laps_df = pd.concat([laps_df, weather_df[["AirTemp", "TrackTemp", "Rainfall"]]], axis=1)

    inserted = 0
    for _, row in laps_df.iterrows():
        code = row["Driver"]
        if code not in driver_map:
            continue  # driver not in session.results (rare edge case) -- skip, don't crash the whole race

        conn.execute(
            text(
                """
                INSERT INTO laps (
                    race_id, driver_id, lap_number, lap_time_ms, sector1_ms,
                    sector2_ms, sector3_ms, compound, tire_age, stint_number,
                    position, is_personal_best, track_status,
                    ambient_temp_c, track_temp_c, rainfall
                )
                VALUES (
                    :race_id, :driver_id, :lap_number, :lap_time_ms, :sector1_ms,
                    :sector2_ms, :sector3_ms, :compound, :tire_age, :stint_number,
                    :position, :is_personal_best, :track_status,
                    :ambient_temp_c, :track_temp_c, :rainfall
                )
                ON CONFLICT (race_id, driver_id, lap_number) DO UPDATE SET
                    lap_time_ms = EXCLUDED.lap_time_ms,
                    compound = EXCLUDED.compound,
                    tire_age = EXCLUDED.tire_age,
                    position = EXCLUDED.position
                """
            ),
            {
                "race_id": race_id,
                "driver_id": driver_map[code],
                "lap_number": int(row["LapNumber"]),
                "lap_time_ms": _ms(row["LapTime"]),
                "sector1_ms": _ms(row["Sector1Time"]),
                "sector2_ms": _ms(row["Sector2Time"]),
                "sector3_ms": _ms(row["Sector3Time"]),
                "compound": row["Compound"],
                "tire_age": int(row["TyreLife"]) if pd.notnull(row["TyreLife"]) else None,
                "stint_number": int(row["Stint"]) if pd.notnull(row["Stint"]) else None,
                "position": int(row["Position"]) if pd.notnull(row["Position"]) else None,
                "is_personal_best": bool(row["IsPersonalBest"]),
                "track_status": str(row["TrackStatus"]),
                "ambient_temp_c": float(row["AirTemp"]) if pd.notnull(row["AirTemp"]) else None,
                "track_temp_c": float(row["TrackTemp"]) if pd.notnull(row["TrackTemp"]) else None,
                "rainfall": bool(row["Rainfall"]) if pd.notnull(row["Rainfall"]) else None,
            },
        )
        inserted += 1
    return inserted


def insert_pit_stops(conn, race_id: int, driver_map: dict[str, int], year: int, round_number: int) -> int:
    ergast = Ergast()
    response = ergast.get_pit_stops(season=year, round=round_number)
    pit_df = response.content[0]

    inserted = 0

    for _, row in pit_df.iterrows():
        ergast_id = row["driverId"]

        # 🔥 normalize
        code = ERGAST_TO_F1.get(ergast_id)
        if not code:
            continue

        if code not in driver_map:
            continue

        conn.execute(
            text("""
                INSERT INTO pit_stops (race_id, driver_id, lap_number, stop_duration_ms)
                VALUES (:race_id, :driver_id, :lap_number, :stop_duration_ms)
                ON CONFLICT (race_id, driver_id, lap_number) DO UPDATE SET
                    stop_duration_ms = EXCLUDED.stop_duration_ms
            """),
            {
                "race_id": race_id,
                "driver_id": driver_map[code],
                "lap_number": int(row["lap"]),
                "stop_duration_ms": _ms(row["duration"]),
            },
        )

        inserted += 1

    return inserted

def insert_telemetry_samples(conn, session, race_id: int, driver_map: dict[str, int], sample_every: int = 5) -> int:
    inserted = 0
    for code, driver_id in driver_map.items():
        driver_laps = session.laps.pick_drivers(code)
        for _, lap in driver_laps.iterlaps():
            try:
                tel = lap.get_telemetry()
            except Exception:
                continue  # some laps (red flag, in/out laps) have no usable telemetry -- skip, don't crash
            if tel.empty:
                continue
            if "Distance" not in tel.columns:
                tel = tel.add_distance()

            sampled = tel.iloc[::sample_every]
            for idx, t in enumerate(sampled.itertuples()):
                conn.execute(
                    text(
                        """
                        INSERT INTO telemetry_samples (
                            race_id, driver_id, lap_number, sample_index, speed,
                            throttle, brake, gear, rpm, drs, x, y, distance
                        )
                        VALUES (
                            :race_id, :driver_id, :lap_number, :sample_index, :speed,
                            :throttle, :brake, :gear, :rpm, :drs, :x, :y, :distance
                        )
                        """
                    ),
                    {
                        "race_id": race_id,
                        "driver_id": driver_id,
                        "lap_number": int(lap["LapNumber"]),
                        "sample_index": idx,
                        "speed": getattr(t, "Speed", None),
                        "throttle": getattr(t, "Throttle", None),
                        "brake": bool(getattr(t, "Brake", False)),
                        "gear": getattr(t, "nGear", None),
                        "rpm": getattr(t, "RPM", None),
                        "drs": getattr(t, "DRS", None),
                        "x": getattr(t, "X", None),
                        "y": getattr(t, "Y", None),
                        "distance": getattr(t, "Distance", None),
                    },
                )
                inserted += 1
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one F1 race weekend into Postgres.")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--round", type=int, required=True, help="Round number, e.g. 1 for Bahrain")
    parser.add_argument(
        "--with-telemetry",
        action="store_true",
        help="Also ingest downsampled telemetry (slow). Skipped by default.",
    )
    args = parser.parse_args()

    load_dotenv()
    engine = create_engine(os.environ["DATABASE_URL"])

    fastf1.Cache.enable_cache("cache")
    session = fastf1.get_session(args.year, args.round, "R")
    session.load()

    with engine.begin() as conn:
        race_id = upsert_race(conn, session, args.year)
        driver_map = upsert_drivers(conn, session, args.year)
        n_laps = insert_laps(conn, session, race_id, driver_map)
        n_pits = insert_pit_stops(conn, race_id, driver_map, args.year, args.round)

        print(f"Round {args.round:>2} {session.event['EventName']:<30} laps={n_laps:<5} pit_stops={n_pits}")

        if args.with_telemetry:
            n_tel = insert_telemetry_samples(conn, session, race_id, driver_map)
            print(f"  + {n_tel} telemetry samples")


if __name__ == "__main__":
    main()