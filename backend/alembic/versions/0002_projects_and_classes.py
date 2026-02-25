"""add projects and classes tables, link training_samples

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-23

"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # projects
    # ------------------------------------------------------------------
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task_type", sa.String(64), nullable=False, server_default="classification"),
        sa.Column(
            "aoi_geometry",
            geoalchemy2.types.Geometry(geometry_type="POLYGON", srid=4326),
            nullable=True,
        ),
        sa.Column("imagery_url", sa.Text(), nullable=True),
        sa.Column("available_bands", postgresql.JSONB(), nullable=True),
        sa.Column("enabled_indices", postgresql.JSONB(), nullable=True),
        sa.Column("resolution_m", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("model_config", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_projects_name", "projects", ["name"])

    # ------------------------------------------------------------------
    # classes
    # ------------------------------------------------------------------
    op.create_table(
        "classes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("color", sa.String(7), nullable=False),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default="0"),
    )

    op.create_index("idx_classes_project_id", "classes", ["project_id"])

    # ------------------------------------------------------------------
    # training_samples — add project_id FK (nullable for existing rows)
    # ------------------------------------------------------------------
    op.add_column(
        "training_samples",
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=True,
        ),
    )

    op.create_index("idx_training_samples_project_id", "training_samples", ["project_id"])


def downgrade() -> None:
    op.drop_index("idx_training_samples_project_id", table_name="training_samples")
    op.drop_column("training_samples", "project_id")

    op.drop_index("idx_classes_project_id", table_name="classes")
    op.drop_table("classes")

    op.drop_index("idx_projects_name", table_name="projects")
    op.drop_table("projects")
