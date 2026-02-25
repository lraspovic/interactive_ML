from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import mapping, shape
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import TrainingSample
from app.schemas import (
    FeatureCollection,
    TrainingSampleCreate,
    TrainingSampleFeature,
    TrainingSampleRead,
)

router = APIRouter(prefix="/training-samples", tags=["training-samples"])


def _sample_to_feature(sample: TrainingSample) -> TrainingSampleFeature:
    """Convert a SQLAlchemy row to a GeoJSON Feature dict."""
    geom = mapping(to_shape(sample.geometry))
    return TrainingSampleFeature(
        id=sample.id,
        geometry=geom,
        properties={
            "label": sample.label,
            "class_id": sample.class_id,
            "project_id": str(sample.project_id) if sample.project_id else None,
            "image_ref": sample.image_ref,
            "metadata": sample.extra,
            "created_at": sample.created_at.isoformat(),
        },
    )


@router.post(
    "",
    response_model=TrainingSampleRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_training_sample(
    body: TrainingSampleCreate,
    db: AsyncSession = Depends(get_db),
):
    """Save a new annotated polygon to the database."""
    try:
        shapely_geom = shape(body.geometry)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid GeoJSON geometry: {exc}",
        )

    sample = TrainingSample(
        geometry=from_shape(shapely_geom, srid=4326),
        label=body.label,
        class_id=body.class_id,
        project_id=body.project_id,
        image_ref=body.image_ref,
        extra=body.metadata,
    )
    db.add(sample)
    await db.commit()
    await db.refresh(sample)

    return TrainingSampleRead(
        id=sample.id,
        geometry=mapping(to_shape(sample.geometry)),
        label=sample.label,
        class_id=sample.class_id,
        project_id=sample.project_id,
        image_ref=sample.image_ref,
        metadata=sample.extra,
        created_at=sample.created_at,
    )


@router.get("", response_model=FeatureCollection)
async def list_training_samples(
    project_id: UUID | None = Query(None, description="Filter by project"),
    db: AsyncSession = Depends(get_db),
):
    """Return training samples as a GeoJSON FeatureCollection, optionally scoped to a project."""
    q = select(TrainingSample).order_by(TrainingSample.created_at.desc())
    if project_id:
        q = q.where(TrainingSample.project_id == project_id)
    result = await db.execute(q)
    samples = result.scalars().all()
    return FeatureCollection(features=[_sample_to_feature(s) for s in samples])


@router.delete("/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_sample(
    sample_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Remove a training sample by ID."""
    result = await db.execute(
        delete(TrainingSample).where(TrainingSample.id == sample_id).returning(TrainingSample.id)
    )
    deleted = result.fetchone()
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found")
    await db.commit()
