"""add training_features table

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-24

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "training_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "training_sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("training_samples.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # Which STAC scene the features were extracted from
        sa.Column("item_id", sa.Text(), nullable=False),
        sa.Column("collection", sa.String(128), nullable=False),
        # Ordered list of feature names (bands + indices), e.g. ["red","nir","NDVI"]
        sa.Column("feature_names", postgresql.JSONB(), nullable=False),
        # Number of valid pixels extracted from the polygon
        sa.Column("n_pixels", sa.Integer(), nullable=False),
        # (n_pixels, n_features) float32 array serialised with numpy.save()
        sa.Column("feature_data", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # Fast lookup: all features for a project+scene
    op.create_index(
        "idx_training_features_project_scene",
        "training_features",
        ["project_id", "item_id"],
    )
    # One row per sample per scene (upsert semantics enforced in application code)
    op.create_index(
        "idx_training_features_sample_scene",
        "training_features",
        ["training_sample_id", "item_id", "collection"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("idx_training_features_sample_scene", table_name="training_features")
    op.drop_index("idx_training_features_project_scene", table_name="training_features")
    op.drop_table("training_features")
