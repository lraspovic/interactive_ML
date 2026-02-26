from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import mapping, shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Class, Project
from app.schemas import ProjectCreate, ProjectRead, ProjectSummary

router = APIRouter(prefix="/projects", tags=["projects"])


def _geom_to_dict(geom) -> dict[str, Any] | None:
    if geom is None:
        return None
    return mapping(to_shape(geom))


def _project_to_read(project: Project) -> ProjectRead:
    return ProjectRead(
        id=project.id,
        name=project.name,
        description=project.description,
        task_type=project.task_type,
        aoi_geometry=_geom_to_dict(project.aoi_geometry),
        imagery_url=project.imagery_url,
        available_bands=project.available_bands or [],
        enabled_indices=project.enabled_indices or [],
        resolution_m=project.resolution_m,
        sensors=project.sensors,
        glcm_config=project.glcm_config,
        model_config=project.model_config,
        classes=[
            {
                "id": c.id,
                "project_id": c.project_id,
                "name": c.name,
                "color": c.color,
                "display_order": c.display_order,
            }
            for c in sorted(project.classes, key=lambda c: c.display_order)
        ],
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.post("", response_model=ProjectRead, status_code=status.HTTP_201_CREATED)
async def create_project(body: ProjectCreate, db: AsyncSession = Depends(get_db)):
    """Create a new project with its class definitions in one transaction."""
    # Parse AOI geometry if provided
    aoi_geom = None
    if body.aoi_geometry:
        try:
            aoi_geom = from_shape(shape(body.aoi_geometry), srid=4326)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid AOI geometry: {exc}",
            )

    # Derive available_bands from the union of sensor bands so both legacy
    # callers (which set available_bands directly) and new wizard submissions
    # (which set sensors) are handled correctly.
    sensors_payload = [
        s.model_dump() for s in (body.sensors or [])
    ]
    if sensors_payload:
        # Build the union of all sensor bands, preserving order and deduplicating
        seen: set[str] = set()
        derived_bands: list[str] = []
        for sc in body.sensors:
            for b in sc.bands:
                if b not in seen:
                    seen.add(b)
                    derived_bands.append(b)
        # Prefer the derived list; fall back to the explicit list only if
        # the derived list is empty (shouldn't happen in practice).
        effective_bands = derived_bands or body.available_bands
    else:
        effective_bands = body.available_bands
        sensors_payload = None

    project = Project(
        id=uuid.uuid4(),
        name=body.name,
        description=body.description,
        task_type=body.task_type,
        aoi_geometry=aoi_geom,
        imagery_url=body.imagery_url,
        available_bands=effective_bands,
        enabled_indices=body.enabled_indices,
        resolution_m=body.resolution_m,
        sensors=sensors_payload,
        glcm_config=body.glcm_config,
        model_config=body.model_config_,
    )
    db.add(project)
    await db.flush()  # get project.id before inserting classes

    for order, cls in enumerate(body.classes):
        db.add(Class(
            project_id=project.id,
            name=cls.name,
            color=cls.color,
            display_order=order,
        ))

    await db.commit()

    # Reload with classes eagerly
    result = await db.execute(
        select(Project).options(selectinload(Project.classes)).where(Project.id == project.id)
    )
    project = result.scalar_one()
    return _project_to_read(project)


@router.get("", response_model=list[ProjectSummary])
async def list_projects(db: AsyncSession = Depends(get_db)):
    """Return all projects with class counts."""
    result = await db.execute(
        select(Project).options(selectinload(Project.classes)).order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()
    return [
        ProjectSummary(
            id=p.id,
            name=p.name,
            description=p.description,
            task_type=p.task_type,
            created_at=p.created_at,
            updated_at=p.updated_at,
            class_count=len(p.classes),
        )
        for p in projects
    ]


@router.get("/{project_id}", response_model=ProjectRead)
async def get_project(project_id: UUID, db: AsyncSession = Depends(get_db)):
    """Return a single project with its classes."""
    result = await db.execute(
        select(Project).options(selectinload(Project.classes)).where(Project.id == project_id)
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return _project_to_read(project)
