"""increase compound varchar size in laps

Revision ID: fa572eea4bfc
Revises: 0001
Create Date: 2026-07-03 13:49:19.428136

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fa572eea4bfc'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.alter_column(
        "laps",
        "compound",
        type_=sa.String(length=20),
        existing_type=sa.String(length=10),
        nullable=True,
    )


def downgrade():
    op.alter_column(
        "laps",
        "compound",
        type_=sa.String(length=10),
        existing_type=sa.String(length=20),
        nullable=True,
    )
