"""
Training endpoints.

POST /train          — kick off background model training for a project + scene
GET  /train/status   — poll progress / metrics
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import tempfile
import uuid
from typing import Any
from uuid import UUID

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from geoalchemy2.shape import to_shape
from pydantic import BaseModel
from shapely.geometry import Point, mapping
from sqlalchemy import delete, select
from starlette.background import BackgroundTask

from app.database import AsyncSessionLocal
from app.ml.features import extract_pixel_features
from app.ml.trainer import fit_model
from app.models import Project, TrainingFeatures, TrainingSample
from app.routers.imagery import _get_signed_assets

router = APIRouter(prefix="/train", tags=["train"])
logger = logging.getLogger(__name__)

# In-memory job status keyed by str(project_id).
# For a multi-process deployment this would need an external store,
# but for a single-worker setup this is fine.
_TRAIN_JOBS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    project_id: UUID
    item_id: str
    collection: str = "sentinel-2-l2a"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
async def trigger_train(body: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model retraining for *project_id* using pixel values extracted from
    the STAC scene identified by *item_id* / *collection*.

    Returns immediately; poll GET /train/status?project_id=... for progress.
    """
    key = str(body.project_id)
    if _TRAIN_JOBS.get(key, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Training already in progress for this project")

    _TRAIN_JOBS[key] = {
        "status": "running",
        "progress": 0.0,
        "metrics": None,
        "error": None,
    }
    background_tasks.add_task(
        _run_training, body.project_id, body.item_id, body.collection
    )
    return {"status": "queued", "project_id": key}


@router.get("/status")
async def train_status(project_id: UUID = Query(...)):
    """Return the current training job state for a project."""
    key = str(project_id)
    job = _TRAIN_JOBS.get(
        key, {"status": "idle", "progress": 0.0, "metrics": None, "error": None}
    )
    return {"project_id": key, **job}


@router.get("/features")
async def feature_means(
    project_id: UUID = Query(...),
    item_id: str = Query(...),
    collection: str = Query("sentinel-2-l2a"),
):
    """
    Return per-polygon feature means for a project + scene combination.

    Only polygons that have already been extracted (cached in training_features)
    are returned — call POST /train first to populate the cache.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(TrainingFeatures, TrainingSample.label, TrainingSample.class_id)
            .join(TrainingSample, TrainingFeatures.training_sample_id == TrainingSample.id)
            .where(
                TrainingFeatures.project_id == project_id,
                TrainingFeatures.item_id == item_id,
                TrainingFeatures.collection == collection,
            )
            .order_by(TrainingSample.label, TrainingFeatures.created_at)
        )
        rows = result.all()

    features = []
    for tf, label, class_id in rows:
        X = np.load(io.BytesIO(tf.feature_data))   # (n_pixels, n_features)
        means = {name: round(float(X[:, i].mean()), 4)
                 for i, name in enumerate(tf.feature_names)}
        features.append({
            "sample_id": str(tf.training_sample_id),
            "label": label,
            "class_id": class_id,
            "n_pixels": tf.n_pixels,
            "means": means,
        })

    return {"item_id": item_id, "collection": collection, "features": features}


@router.get("/features/download")
async def download_features_gpkg(
    project_id: UUID = Query(...),
    item_id: str = Query(...),
    collection: str = Query("sentinel-2-l2a"),
):
    """
    Download a GeoPackage with two layers:

    - ``training_polygons``: one row per polygon, polygon geometry,
      label / class_id / n_pixels / mean+std per feature band.
    - ``pixel_values``: one row per pixel, **Point geometry at the pixel centre**,
      label / class_id / sample_id / pixel_idx / all raw feature values.

    Open in QGIS: drag the .gpkg onto the map; select a layer in the dialog.
    CRS: EPSG:4326.
    """
    import geopandas as gpd
    import pandas as pd

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                TrainingFeatures,
                TrainingSample.label,
                TrainingSample.class_id,
                TrainingSample.geometry,
            )
            .join(TrainingSample, TrainingFeatures.training_sample_id == TrainingSample.id)
            .where(
                TrainingFeatures.project_id == project_id,
                TrainingFeatures.item_id == item_id,
                TrainingFeatures.collection == collection,
            )
            .order_by(TrainingSample.label, TrainingFeatures.created_at)
        )
        rows = result.all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No extracted features found for this scene. Run training first.",
        )

    poly_records = []
    pixel_records = []

    for tf, label, class_id, geom_wkb in rows:
        X = np.load(io.BytesIO(tf.feature_data))  # (n_pixels, n_features)
        names: list[str] = tf.feature_names
        geom = to_shape(geom_wkb)
        sid = str(tf.training_sample_id)

        # Deserialise pixel centroids (may be NULL for rows extracted before migration)
        coords_wgs84: np.ndarray | None = None
        if tf.pixel_centroids is not None:
            coords_wgs84 = np.load(io.BytesIO(tf.pixel_centroids))  # (n_pixels, 2)

        # ── polygon summary row ──────────────────────────────────────────────
        poly_row: dict = {
            "geometry": geom,
            "sample_id": sid,
            "label": label,
            "class_id": class_id,
            "n_pixels": tf.n_pixels,
        }
        for i, name in enumerate(names):
            col = X[:, i]
            poly_row[f"{name}_mean"] = round(float(col.mean()), 6)
            poly_row[f"{name}_std"] = round(float(col.std()), 6)
            poly_row[f"{name}_min"] = round(float(col.min()), 6)
            poly_row[f"{name}_max"] = round(float(col.max()), 6)
        poly_records.append(poly_row)

        # ── per-pixel rows with Point geometry ─────────────────────────
        for px_idx in range(X.shape[0]):
            if coords_wgs84 is not None:
                lon, lat = coords_wgs84[px_idx]
                px_geom = Point(lon, lat)
            else:
                # Legacy fallback: polygon centroid (no per-pixel coords stored)
                px_geom = geom.centroid

            px_row: dict = {
                "geometry": px_geom,
                "sample_id": sid,
                "pixel_idx": px_idx,
                "label": label,
                "class_id": class_id,
            }
            for i, name in enumerate(names):
                px_row[name] = round(float(X[px_idx, i]), 6)
            pixel_records.append(px_row)

    poly_gdf = gpd.GeoDataFrame(poly_records, crs="EPSG:4326")
    pixel_gdf = gpd.GeoDataFrame(pixel_records, crs="EPSG:4326")

    tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
    tmp.close()

    poly_gdf.to_file(tmp.name, driver="GPKG", layer="training_polygons", engine="pyogrio")
    pixel_gdf.to_file(tmp.name, driver="GPKG", layer="pixel_values", engine="pyogrio")

    safe_item = item_id.replace("/", "_")[:30]
    filename = f"features_{str(project_id)[:8]}_{safe_item}.gpkg"

    return FileResponse(
        tmp.name,
        media_type="application/geopackage+sqlite3",
        filename=filename,
        background=BackgroundTask(os.unlink, tmp.name),
    )


# ---------------------------------------------------------------------------
# Polygon-level train / test split
# ---------------------------------------------------------------------------

def _polygon_split(
    poly_data: list[tuple["np.ndarray", int, float]],
    train_frac: float = 0.80,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Area-weighted, per-class polygon-level split.

    Rules
    -----
    - Every class always has at least one polygon in training.
    - Classes with a single polygon: that polygon goes to train only.
    - For multi-polygon classes: polygons are shuffled (fixed seed) then
      greedily assigned to train until *train_frac* of the class area is
      reached; the remainder goes to test.
    - Using polygon area in EPSG:4326 degrees² as a proxy for pixel count
      is fine because we only use it for proportional ranking within a class.
    - The seed is fixed so that switching model types later reuses the same
      splits.
    """
    import random
    from collections import defaultdict

    class_to_idx: dict[int, list[int]] = defaultdict(list)
    for idx, (_, cls, _) in enumerate(poly_data):
        class_to_idx[cls].append(idx)

    train_idx: list[int] = []
    test_idx: list[int] = []
    rng = random.Random(seed)

    for cls, indices in sorted(class_to_idx.items()):
        if len(indices) == 1:
            # Single polygon — always train, never penalise test
            train_idx.append(indices[0])
            continue

        shuffled = indices[:]
        rng.shuffle(shuffled)

        areas = [poly_data[i][2] for i in shuffled]
        total = sum(areas) or 1.0          # guard against zero-area edge case
        target = total * train_frac

        cum = 0.0
        cls_train: list[int] = []
        cls_test: list[int] = []
        for i, idx in enumerate(shuffled):
            if cum < target:
                cls_train.append(idx)
                cum += areas[i]
            else:
                cls_test.append(idx)

        # Guarantee at least one polygon in test (move last train poly if needed)
        if not cls_test and len(cls_train) > 1:
            cls_test.append(cls_train.pop())

        train_idx.extend(cls_train)
        test_idx.extend(cls_test)

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_training(project_id: UUID, item_id: str, collection: str) -> None:
    key = str(project_id)
    try:
        # ── 1. Load project config and training samples from DB ──────────────
        async with AsyncSessionLocal() as db:
            project = await db.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found")

            available_bands: list[str] = project.available_bands or ["blue", "green", "red"]
            enabled_indices: list[str] = project.enabled_indices or []
            model_config: dict = project.model_config or {}

            result = await db.execute(
                select(TrainingSample).where(TrainingSample.project_id == project_id)
            )
            samples = result.scalars().all()

        if not samples:
            raise ValueError("No training samples found — draw some polygons first")

        # ── 2. Fetch signed asset hrefs (cached) ─────────────────────────────
        signed_assets = _get_signed_assets(collection, item_id)

        # ── 3. Extract & persist features per polygon ────────────────────────
        # poly_data entries: (X_poly, class_id, area_deg2)
        # Keyed by position so the split indices map back to the same arrays.
        poly_data: list[tuple[np.ndarray, int, float]] = []
        feature_names: list[str] = []
        n = len(samples)

        for i, sample in enumerate(samples):
            sample_id = sample.id
            label = int(sample.class_id) if sample.class_id is not None else 0
            area_deg2 = float(to_shape(sample.geometry).area)

            # Check DB cache first ─────────────────────────────────────────
            async with AsyncSessionLocal() as db:
                cached = await db.execute(
                    select(TrainingFeatures).where(
                        TrainingFeatures.training_sample_id == sample_id,
                        TrainingFeatures.item_id == item_id,
                        TrainingFeatures.collection == collection,
                    )
                )
                cached_row = cached.scalar_one_or_none()

            if cached_row is not None:
                X_poly = np.load(io.BytesIO(cached_row.feature_data))
                if not feature_names:
                    feature_names = cached_row.feature_names
                logger.debug(
                    "Loaded %d pixels from DB cache for sample %s", len(X_poly), sample_id
                )
            else:
                geometry = mapping(to_shape(sample.geometry))
                X_poly, names, coords = await asyncio.to_thread(
                    extract_pixel_features,
                    geometry,
                    signed_assets,
                    available_bands,
                    enabled_indices,
                )
                if X_poly is None or len(X_poly) == 0:
                    logger.warning("No valid pixels for sample %s — skipping", sample_id)
                    _TRAIN_JOBS[key]["progress"] = round((i + 1) / n * 0.80, 3)
                    continue

                if not feature_names:
                    feature_names = names

                buf = io.BytesIO()
                np.save(buf, X_poly.astype(np.float32))
                feature_bytes = buf.getvalue()

                centroid_bytes: bytes | None = None
                if coords is not None:
                    cbuf = io.BytesIO()
                    np.save(cbuf, coords.astype(np.float64))
                    centroid_bytes = cbuf.getvalue()

                async with AsyncSessionLocal() as db:
                    await db.execute(
                        delete(TrainingFeatures).where(
                            TrainingFeatures.training_sample_id == sample_id,
                            TrainingFeatures.item_id == item_id,
                            TrainingFeatures.collection == collection,
                        )
                    )
                    db.add(TrainingFeatures(
                        id=uuid.uuid4(),
                        training_sample_id=sample_id,
                        project_id=project_id,
                        item_id=item_id,
                        collection=collection,
                        feature_names=names,
                        n_pixels=len(X_poly),
                        feature_data=feature_bytes,
                        pixel_centroids=centroid_bytes,
                    ))
                    await db.commit()

                logger.info(
                    "Saved %d pixels for sample %s (scene %s)", len(X_poly), sample_id, item_id
                )

            poly_data.append((X_poly, label, area_deg2))
            _TRAIN_JOBS[key]["progress"] = round((i + 1) / n * 0.80, 3)

        if not poly_data:
            raise ValueError(
                "No valid pixels could be extracted from the training polygons. "
                "Make sure the selected scene overlaps your drawn polygons."
            )

        if len({cls for _, cls, _ in poly_data}) < 2:
            raise ValueError(
                "Only one class found in training data. "
                "Draw polygons for at least 2 different classes."
            )

        # ── 4. Polygon-level train / test split (fixed seed=42) ──────────────
        train_idx, test_idx = _polygon_split(poly_data, train_frac=0.80, seed=42)

        def _stack(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
            Xs = [poly_data[i][0] for i in indices]
            ys = [np.full(len(poly_data[i][0]), poly_data[i][1], dtype=np.int32)
                  for i in indices]
            return np.vstack(Xs), np.concatenate(ys)

        X_train, y_train = _stack(train_idx)
        X_test,  y_test  = _stack(test_idx) if test_idx else (None, None)

        logger.info(
            "Split: %d train polygons (%d px) | %d test polygons (%d px)",
            len(train_idx), len(X_train),
            len(test_idx),  len(X_test) if X_test is not None else 0,
        )

        _TRAIN_JOBS[key]["progress"] = 0.85

        # ── 5. Phase 1 — fit on train split, evaluate on test split ──────────
        version = uuid.uuid4().hex[:8]
        _, metrics, artifact_path = await asyncio.to_thread(
            fit_model, X_train, y_train, X_test, y_test, model_config, version
        )
        metrics["feature_names"] = feature_names
        metrics["n_train_polygons"] = len(train_idx)
        metrics["n_test_polygons"] = len(test_idx)

        # Expose metrics to the frontend immediately so the user can see
        # accuracy while phase 2 runs in the background.
        _TRAIN_JOBS[key] = {
            "status": "done",
            "phase": "refitting",
            "progress": 1.0,
            "metrics": metrics,
            "error": None,
            "artifact": str(artifact_path),
        }
        logger.info("Phase 1 complete for project %s: %s", project_id, metrics)

        # ── 6. Phase 2 — refit on ALL data, overwrite _latest.pkl ────────────
        X_all, y_all = _stack(list(range(len(poly_data))))
        _, _, final_path = await asyncio.to_thread(
            fit_model, X_all, y_all, None, None, model_config, f"{version}_full"
        )
        latest_path = final_path.parent / f"model_{project_id}_latest.pkl"
        shutil.copy(final_path, latest_path)

        _TRAIN_JOBS[key]["phase"] = "complete"
        logger.info(
            "Phase 2 complete for project %s — full model saved to %s",
            project_id, latest_path,
        )

    except Exception as exc:
        logger.exception("Training failed for project %s", project_id)
        _TRAIN_JOBS[key] = {
            "status": "failed",
            "progress": 0.0,
            "metrics": None,
            "error": str(exc),
        }
