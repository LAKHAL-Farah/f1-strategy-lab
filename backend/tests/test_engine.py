"""
backend/tests/test_engine.py

This is Phase 1's done condition, made executable. If this fails, the
bug is in load_historical() or in the ingested data -- nothing in
Phase 2 onward should be trusted until this passes, since everything
later sits on top of this engine.

Run from backend/:
    uv run pytest
"""
import os

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from simulation.engine import SimulationEngine

load_dotenv()


@pytest.fixture(scope="module")
def db_engine():
    return create_engine(os.environ["DATABASE_URL"])


def test_bahrain_2023_replay_matches_real_result(db_engine):
    """
    Verify the actual classified top 3 yourself (explore.py, or the
    official FIA result) before trusting this assertion -- don't take it
    from a comment written by someone who wasn't watching the race.
    """
    with db_engine.connect() as conn:
        race_id = conn.execute(
            text("SELECT id FROM races WHERE year = 2023 AND name ILIKE :q"),
            {"q": "%Bahrain%"},
        ).scalar_one()

    engine = SimulationEngine(db_engine)
    engine.load_historical(race_id)

    assert len(engine.states) > 0, "No laps loaded -- check ingestion ran for this race first"

    final_state = engine.states[-1]
    podium = sorted(final_state.drivers.values(), key=lambda d: d.position)[:3]
    podium_codes = [d.code for d in podium]

    assert podium_codes == ["VER", "PER", "ALO"], (
        f"Expected VER/PER/ALO, got {podium_codes}. If you've confirmed the "
        f"real result differs from this, update the assertion -- don't just "
        f"delete the test."
    )