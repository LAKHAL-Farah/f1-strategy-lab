"""
backend/explore.py

Run after ingesting some or all of the 2023 season. Prints, per race, the
lap row count, compound breakdown, and pit stop count -- so you actually
look at the data's shape before Phase 1 trusts it blindly.

Usage:
    uv run python explore.py
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

with engine.connect() as conn:
    races = pd.read_sql(
        text("SELECT id, round, name, total_laps FROM races ORDER BY round"), conn
    )
    print(f"Races ingested: {len(races)} / 22 expected\n")

    for _, race in races.iterrows():
        compound_counts = pd.read_sql(
            text("SELECT compound, count(*) AS n FROM laps WHERE race_id = :rid GROUP BY compound"),
            conn,
            params={"rid": int(race["id"])},
        )
        n_pit_stops = pd.read_sql(
            text("SELECT count(*) AS n FROM pit_stops WHERE race_id = :rid"),
            conn,
            params={"rid": int(race["id"])},
        ).iloc[0]["n"]

        total_lap_rows = int(compound_counts["n"].sum())
        # rough sanity expectation: ~20 drivers x total_laps, will be a bit
        # lower in practice (DNFs, drivers who didn't start, deleted laps)
        expected = int(race["total_laps"]) * 20 if pd.notnull(race["total_laps"]) else None

        print(
            f"Round {race['round']:>2} {race['name']:<30} "
            f"lap_rows={total_lap_rows:>4} (expected ~{expected}) pit_stops={n_pit_stops}"
        )
        compounds_dict = dict(zip(compound_counts["compound"], compound_counts["n"]))
        print(f"    compounds: {compounds_dict}")

        if total_lap_rows == 0:
            print("    *** NO LAPS FOUND -- ingestion may have failed silently for this race ***")