"""
backend/simulate.py

CLI entry point for Phase 1 pure replay.

Usage:
    uv run python simulate.py --race "Bahrain" --mode replay
"""
import argparse
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from simulation.engine import SimulationEngine


def resolve_race_id(conn, year: int, race_query: str) -> tuple[int, str]:
    """Case-insensitive substring match against races.name.

    Errors loudly on zero or multiple matches rather than silently
    picking one -- the same "don't trust silent success" principle as
    everywhere else in this project, just applied to a CLI argument
    instead of a database row count.
    """
    rows = conn.execute(
        text("SELECT id, name FROM races WHERE year = :year AND name ILIKE :q"),
        {"year": year, "q": f"%{race_query}%"},
    ).all()

    if len(rows) == 0:
        raise SystemExit(f"No race matching '{race_query}' found for {year}. Run explore.py to see what's loaded.")
    if len(rows) > 1:
        names = ", ".join(r.name for r in rows)
        raise SystemExit(f"'{race_query}' matched multiple races: {names}. Be more specific.")
    return rows[0].id, rows[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a race lap by lap from the database.")
    parser.add_argument("--race", required=True, help='Substring match against race name, e.g. "Bahrain"')
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument(
        "--mode",
        choices=["replay"],
        default="replay",
        help="Only 'replay' exists in Phase 1 -- simulated modes arrive in Phase 5",
    )
    args = parser.parse_args()

    load_dotenv()
    db = create_engine(os.environ["DATABASE_URL"])

    with db.connect() as conn:
        race_id, race_name = resolve_race_id(conn, args.year, args.race)

    print(f"Replaying {race_name} ({args.year})\n")

    engine = SimulationEngine(db)
    engine.load_historical(race_id)

    for state in engine.states:
        flags = ""
        if state.safety_car:
            flags += "  [SAFETY CAR]"
        if state.vsc:
            flags += "  [VSC]"
        print(f"--- Lap {state.lap} ---{flags}")

        ordered = sorted(state.drivers.values(), key=lambda d: d.position)
        for d in ordered:
            print(
                f"  P{d.position:>2}  {d.code}  +{d.gap_to_leader:6.1f}s  "
                f"{d.current_tire.compound:<8} (age {d.current_tire.age})"
            )

    final = engine.states[-1]
    podium = sorted(final.drivers.values(), key=lambda d: d.position)[:3]
    print("\nFinal podium (replay):")
    for d in podium:
        print(f"  P{d.position}  {d.code}")


if __name__ == "__main__":
    main()