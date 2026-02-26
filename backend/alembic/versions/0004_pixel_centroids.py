"""add pixel_centroids column to training_features

Revision ID: 0004
Revises: 0003
Create Date: 2026-02-26

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "training_features",
        sa.Column("pixel_centroids", sa.LargeBinary(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_features", "pixel_centroids")
