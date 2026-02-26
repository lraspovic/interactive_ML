"""add sensors and glcm_config columns to projects; make resolution_m nullable

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # sensors: list[{sensor_type, bands}] JSON — nullable for old rows
    op.add_column(
        "projects",
        sa.Column("sensors", postgresql.JSONB(), nullable=True),
    )
    # glcm_config: {enabled, bands, window_size, statistics} JSON — nullable
    op.add_column(
        "projects",
        sa.Column("glcm_config", postgresql.JSONB(), nullable=True),
    )
    # resolution_m was NOT NULL; relax to nullable so the column persists but
    # the ML pipeline is not constrained by it.
    op.alter_column(
        "projects",
        "resolution_m",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    op.drop_column("projects", "sensors")
    op.drop_column("projects", "glcm_config")
    # Restore NOT NULL constraint (existing NULLs would need back-filling
    # before running downgrade in production).
    op.alter_column(
        "projects",
        "resolution_m",
        existing_type=sa.Integer(),
        nullable=False,
        server_default="10",
    )
