"""
Prediction endpoints.

POST /predict               — kick off a background prediction job
GET  /predict/status        — poll prediction job status
GET  /predict/image/{id}    — serve classification PNG
GET  /predict/uncertainty/{id} — serve uncertainty heatmap PNG
"""
from __future__ import annotations

import asyncio
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.ml.predictor import (
    extract_bbox_features,
    read_raw_bands_tiff,
    render_prediction_png,
    render_uncertainty_png,
)
from app.ml.trainer import MODEL_ARTIFACTS_DIR
from app.models import Class, Project
from app.routers.imagery import _get_signed_assets

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

# In-memory job state — same pattern as _TRAIN_JOBS in train.py.
_PREDICT_JOBS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    project_id: UUID
    item_id: str
    collection: str = "sentinel-2-l2a"
    bbox: list[float]   # [minlon, minlat, maxlon, maxlat]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
async def trigger_predict(body: PredictRequest, background_tasks: BackgroundTasks):
    """
    Start a prediction job.  Returns immediately; poll GET /predict/status.

    The bbox is the map viewport at the time the user clicks "Run Prediction".
    Prediction and uncertainty PNGs are saved to disk and overwritten each run.
    """
    key = str(body.project_id)
    if _PREDICT_JOBS.get(key, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Prediction already in progress")

    _PREDICT_JOBS[key] = {
        "status":      "running",
        "progress":    0.0,
        "bbox":        body.bbox,
        "actual_bbox": None,
        "item_id":     body.item_id,
        "collection":  body.collection,
        "timestamp":   None,
        "error":       None,
    }
    background_tasks.add_task(
        _run_prediction,
        body.project_id,
        body.item_id,
        body.collection,
        body.bbox,
    )
    return {"status": "queued", "project_id": key}


@router.get("/status")
async def predict_status(project_id: UUID = Query(...)):
    """Return the current prediction job state for a project."""
    key = str(project_id)
    job = _PREDICT_JOBS.get(key, {
        "status":      "idle",
        "progress":    0.0,
        "bbox":        None,
        "actual_bbox": None,
        "timestamp":   None,
        "error":       None,
    })
    return {"project_id": key, **job}


@router.get("/image/{project_id}")
async def prediction_image(project_id: str):
    """Serve the latest classification PNG for a project."""
    path = MODEL_ARTIFACTS_DIR / f"pred_{project_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No prediction found. Run a prediction first.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/uncertainty/{project_id}")
async def uncertainty_image(project_id: str):
    """Serve the latest uncertainty heatmap PNG for a project."""
    path = MODEL_ARTIFACTS_DIR / f"uncertainty_{project_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No uncertainty map found.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/debug-tiff/{project_id}")
async def prediction_debug_tiff(project_id: str):
    """
    Download the raw band raster used for the last prediction as a georeferenced
    multi-band GeoTIFF (float32, EPSG:4326).  One band per available_bands entry;
    band names stored as raster tags.  Open in QGIS to inspect pixel values.
    """
    from uuid import UUID as _UUID
    from fastapi.responses import Response as _Response

    job = _PREDICT_JOBS.get(project_id)
    if not job or job.get("status") not in ("done", "failed", "running"):
        raise HTTPException(
            status_code=404,
            detail="No prediction job found for this project. Run a prediction first.",
        )

    bbox = job.get("actual_bbox") or job.get("bbox")
    item_id = job.get("item_id")
    collection = job.get("collection", "sentinel-2-l2a")

    if not bbox or not item_id:
        raise HTTPException(
            status_code=404,
            detail="Scene / bbox info not available — start a new prediction first.",
        )

    try:
        pid = _UUID(project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="Invalid project_id format.")

    async with AsyncSessionLocal() as db:
        project = await db.get(Project, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        available_bands: list[str] = project.available_bands or ["blue", "green", "red"]
        enabled_indices: list[str] = project.enabled_indices or []

    signed_assets = await asyncio.to_thread(_get_signed_assets, collection, item_id)
    tiff_bytes = await asyncio.to_thread(
        read_raw_bands_tiff, bbox, signed_assets, available_bands, enabled_indices
    )

    if not tiff_bytes:
        raise HTTPException(
            status_code=500,
            detail="Could not read any band data for this bbox/scene.",
        )

    short_id = project_id[:8]
    filename = f"debug_{short_id}_{item_id[:20]}.tif"
    return _Response(
        content=tiff_bytes,
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_prediction(
    project_id: UUID,
    item_id: str,
    collection: str,
    bbox: list[float],
) -> None:
    key = str(project_id)
    try:
        # ── 1. Load project config ────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            project = await db.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found")
            available_bands: list[str] = project.available_bands or ["blue", "green", "red"]
            enabled_indices: list[str] = project.enabled_indices or []

        # ── 2. Load trained model ─────────────────────────────────────────────
        latest_path = MODEL_ARTIFACTS_DIR / f"model_{project_id}_latest.pkl"
        if not latest_path.exists():
            raise ValueError(
                "No trained model found. Train the model before running prediction."
            )
        with open(latest_path, "rb") as fh:
            data = pickle.load(fh)
        model = data["model"]

        _PREDICT_JOBS[key]["progress"] = 0.15

        # ── 3. Fetch signed asset hrefs (cached) ──────────────────────────────
        signed_assets = await asyncio.to_thread(_get_signed_assets, collection, item_id)

        _PREDICT_JOBS[key]["progress"] = 0.25

        # ── 4. Extract bbox features ──────────────────────────────────────────
        X, H, W, valid_mask, feature_names, actual_bbox = await asyncio.to_thread(
            extract_bbox_features,
            bbox,
            signed_assets,
            available_bands,
            enabled_indices,
        )
        # Fall back to the requested bbox if rio_tiler didn't report actual bounds
        actual_bbox = actual_bbox or bbox

        if X is None or H == 0 or W == 0:
            raise ValueError(
                "Could not read any pixel data from the scene for the given viewport. "
                "Try zooming in or selecting a different scene."
            )

        n_valid = int(valid_mask.sum())
        logger.info(
            "Prediction features: %d×%d px, %d valid (%.1f%%), %d features, "
            "value range [%.1f, %.1f]",
            H, W, n_valid, 100 * n_valid / (H * W),
            X.shape[1],
            float(X[valid_mask].min()) if n_valid else 0,
            float(X[valid_mask].max()) if n_valid else 0,
        )

        _PREDICT_JOBS[key]["progress"] = 0.70

        # ── 5. Run prediction + uncertainty ───────────────────────────────────
        # Only predict on valid pixels (nodata pixels have all-zero features and
        # were excluded during training — feeding them to the model causes them
        # to be assigned an arbitrary class, polluting the result).
        n_pixels = H * W
        n_classes = len(model.classes_)
        predictions = np.full(n_pixels, -1, dtype=np.int32)
        probas      = np.zeros((n_pixels, n_classes), dtype=np.float32)

        if valid_mask.any():
            X_valid = X[valid_mask]
            predictions[valid_mask] = model.predict(X_valid)
            probas[valid_mask]      = model.predict_proba(X_valid)

        _PREDICT_JOBS[key]["progress"] = 0.85

        # ── 6. Load class colors ──────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Class).where(Class.project_id == project_id)
            )
            class_rows = result.scalars().all()
        class_colors: dict[int, str] = {c.id: c.color for c in class_rows}

        # ── 7. Render and save PNGs ───────────────────────────────────────────
        pred_png = render_prediction_png(predictions, class_colors, H, W, valid_mask)
        unc_png  = render_uncertainty_png(probas, H, W, valid_mask)

        MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (MODEL_ARTIFACTS_DIR / f"pred_{project_id}.png").write_bytes(pred_png)
        (MODEL_ARTIFACTS_DIR / f"uncertainty_{project_id}.png").write_bytes(unc_png)

        _PREDICT_JOBS[key] = {
            "status":      "done",
            "progress":    1.0,
            "bbox":        bbox,
            "actual_bbox": actual_bbox,
            "item_id":     item_id,
            "collection":  collection,
            "timestamp":   int(time.time()),
            "error":       None,
            "image_size":  [W, H],
        }
        logger.info(
            "Prediction complete for project %s — %d×%d px, %d classes",
            project_id, W, H, len(class_colors),
        )

    except Exception as exc:
        logger.exception("Prediction failed for project %s", project_id)
        _PREDICT_JOBS[key] = {
            "status":      "failed",
            "progress":    0.0,
            "bbox":        bbox,
            "actual_bbox": None,
            "item_id":     item_id,
            "collection":  collection,
            "timestamp":   None,
            "error":       str(exc),
        }
