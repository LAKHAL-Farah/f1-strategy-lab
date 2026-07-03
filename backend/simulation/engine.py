"""
backend/simulation/engine.py
"""
from collections import defaultdict

import pandas as pd
from sqlalchemy import text

from simulation.models import Driver, RaceState, TireSet


class SimulationEngine:
    """
    Pure replay only, for now. load_historical() builds the FULL list of
    per-lap RaceState snapshots up front from the database; it contains
    zero business logic -- no degradation modeling, no strategy
    decisions, nothing beyond what's already true in the data. That's
    deliberate: pure replay's only job is to prove the data pipeline and
    this engine agree with reality, exactly. advance_lap_historical() is
    just an index lookup into that pre-built list.

    Known limitations (documented here rather than discovered later):

    - gap_to_leader is APPROXIMATE. It's cumulative summed lap time vs.
      the lap leader's cumulative summed lap time, which ignores pit
      lane time-loss differences and compresses incorrectly during
      safety car / VSC periods (real gaps close up, lap times don't
      reflect that proportionally). Good enough to sanity-check "is this
      driver roughly where they should be"; not accurate enough to
      validate strategy-relevant gap thresholds (e.g. undercut range).
      That's a problem for whichever later phase actually needs it.

    - Any lap with a NULL lap_time_ms (rare -- red flag laps, occasional
      data gaps) contributes 0 to that driver's cumulative time, making
      their gap_to_leader look artificially better than reality for
      every lap after that point. This affects gap accuracy only --
      finishing position comes straight from FastF1's own per-lap
      `position` column, not from anything computed here.

    - The final lap's `position` reflects on-track running order, not
      official classification including any post-race time penalties.
      Identical at the front of the field for most races; if a podium
      test fails for a race you know had a post-race penalty, check
      that before assuming the engine is wrong.
    """

    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.states: list[RaceState] = []

    def load_historical(self, race_id: int) -> None:
        with self.db_engine.connect() as conn:
            total_laps = conn.execute(
                text("SELECT total_laps FROM races WHERE id = :rid"),
                {"rid": race_id},
            ).scalar_one()

            laps_df = pd.read_sql(
                text(
                    """
                    SELECT l.lap_number, l.lap_time_ms, l.compound, l.tire_age,
                           l.stint_number, l.position, l.track_status, l.rainfall,
                           d.code, d.team
                    FROM laps l
                    JOIN drivers d ON d.id = l.driver_id
                    WHERE l.race_id = :rid
                    ORDER BY l.lap_number, l.position
                    """
                ),
                conn,
                params={"rid": race_id},
            )

        if laps_df.empty:
            raise ValueError(
                f"No laps found for race_id={race_id} -- check that ingestion ran for this race"
            )

        # Cumulative race time per driver, used only for the approximate
        # gap_to_leader above. Must sort by (code, lap_number) first --
        # groupby().cumsum() respects existing row order within a group,
        # it does not sort for you.
        laps_df = laps_df.sort_values(["code", "lap_number"]).reset_index(drop=True)
        laps_df["lap_time_ms_filled"] = laps_df["lap_time_ms"].fillna(0)
        laps_df["cumulative_ms"] = laps_df.groupby("code")["lap_time_ms_filled"].cumsum()

        driver_cursor: dict[str, tuple[TireSet, int]] = {}
        stint_history: dict[str, list[TireSet]] = defaultdict(list)
        lap_times_so_far: dict[str, list[float]] = defaultdict(list)

        self.states = []
        for lap_number in range(1, int(total_laps) + 1):
            lap_rows = laps_df[laps_df["lap_number"] == lap_number]
            if lap_rows.empty:
                continue  # no driver has a row for this lap number (rare data gap) -- skip it

            leader_rows = lap_rows[lap_rows["position"] == 1]
            leader_cum_ms = leader_rows["cumulative_ms"].iloc[0] if not leader_rows.empty else None

            drivers: dict[str, Driver] = {}
            for _, row in lap_rows.iterrows():
                code = row["code"]
                stint_number = int(row["stint_number"]) if pd.notnull(row["stint_number"]) else 1

                current_tire = TireSet(
                    compound=row["compound"] or "UNKNOWN",
                    age=int(row["tire_age"]) if pd.notnull(row["tire_age"]) else 0,
                )

                prev = driver_cursor.get(code)
                if prev is not None and prev[1] != stint_number:
                    stint_history[code].append(prev[0])  # archive the stint that just ended
                driver_cursor[code] = (current_tire, stint_number)

                if pd.notnull(row["lap_time_ms"]):
                    lap_times_so_far[code].append(row["lap_time_ms"] / 1000.0)

                gap = (
                    (row["cumulative_ms"] - leader_cum_ms) / 1000.0
                    if leader_cum_ms is not None
                    else 0.0
                )

                drivers[code] = Driver(
                    code=code,
                    team=row["team"],
                    position=int(row["position"]) if pd.notnull(row["position"]) else 99,
                    gap_to_leader=gap,
                    current_tire=current_tire,
                    stint_history=list(stint_history[code]),
                    lap_times=list(lap_times_so_far[code]),
                )

            # TrackStatus codes per FastF1: 1=clear 2=yellow 4=SC 5=red 6=VSC 7=VSC ending.
            # A lap's status can contain more than one code (e.g. "24" if a
            # yellow preceded a safety car within the same lap), so this is
            # a substring containment check, not equality.
            track_status_codes = "".join(
                str(s) for s in lap_rows["track_status"].dropna().unique()
            )

            self.states.append(
                RaceState(
                    race_id=race_id,
                    lap=lap_number,
                    drivers=drivers,
                    safety_car="4" in track_status_codes,
                    vsc="6" in track_status_codes,
                    weather="wet" if bool(lap_rows["rainfall"].fillna(False).any()) else "dry",
                )
            )

    def advance_lap_historical(self, lap: int) -> RaceState:
        """Pure index lookup -- no computation happens here. If you find
        yourself wanting to compute something inside this method, it
        belongs in load_historical instead, so the state list stays
        fully pre-built and every RaceState stays immutable."""
        if not self.states:
            raise RuntimeError("Call load_historical() before advance_lap_historical()")
        if not (1 <= lap <= len(self.states)):
            raise ValueError(
                f"Lap {lap} out of range -- this race has {len(self.states)} laps loaded"
            )
        return self.states[lap - 1]