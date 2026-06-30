"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""
from pathlib import Path

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# migrations/versions/0001_initial_schema.py -> parents[2] is backend/
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "ingestion" / "schema.sql"


def upgrade() -> None:
    op.execute(SCHEMA_PATH.read_text())


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS telemetry_samples;
        DROP TABLE IF EXISTS pit_stops;
        DROP TABLE IF EXISTS laps;
        DROP TABLE IF EXISTS drivers;
        DROP TABLE IF EXISTS races;
        """
    )