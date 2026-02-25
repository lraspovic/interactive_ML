from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class ClassCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    color: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")
    display_order: int = 0


class ClassRead(BaseModel):
    id: int
    project_id: UUID
    name: str
    color: str
    display_order: int

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    task_type: str = "classification"
    aoi_geometry: dict[str, Any] | None = Field(
        None, description="GeoJSON Polygon for the area of interest"
    )
    imagery_url: str | None = None
    available_bands: list[str] = ["blue", "green", "red"]
    enabled_indices: list[str] = []
    resolution_m: int = 10
    model_config_: dict[str, Any] | None = Field(None, alias="model_config")
    classes: list[ClassCreate] = Field(..., min_length=2)


class ProjectRead(BaseModel):
    id: UUID
    name: str
    description: str | None
    task_type: str
    aoi_geometry: dict[str, Any] | None
    imagery_url: str | None
    available_bands: list[str]
    enabled_indices: list[str]
    resolution_m: int
    model_config: dict[str, Any] | None
    classes: list[ClassRead]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProjectSummary(BaseModel):
    """Lightweight listing — no classes, no geometry."""
    id: UUID
    name: str
    description: str | None
    task_type: str
    created_at: datetime
    updated_at: datetime
    class_count: int

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Training samples
# ---------------------------------------------------------------------------

class TrainingSampleCreate(BaseModel):
    """GeoJSON-style polygon + class label POSTed by the frontend."""

    geometry: dict[str, Any] = Field(
        ...,
        description="GeoJSON Geometry object (type: Polygon)",
        examples=[{
            "type": "Polygon",
            "coordinates": [[[16.0, 48.0], [16.1, 48.0], [16.1, 48.1], [16.0, 48.0]]],
        }],
    )
    label: str = Field(..., description="Human-readable class name, e.g. 'Forest'")
    class_id: int | None = None
    project_id: UUID | None = None
    image_ref: str | None = None
    metadata: dict[str, Any] | None = None


class TrainingSampleRead(BaseModel):
    id: UUID
    geometry: dict[str, Any]
    label: str
    class_id: int | None
    project_id: UUID | None
    image_ref: str | None
    metadata: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingSampleFeature(BaseModel):
    """A single GeoJSON Feature wrapping a training sample."""

    type: str = "Feature"
    id: UUID
    geometry: dict[str, Any]
    properties: dict[str, Any]


class FeatureCollection(BaseModel):
    """Standard GeoJSON FeatureCollection."""

    type: str = "FeatureCollection"
    features: list[TrainingSampleFeature]
