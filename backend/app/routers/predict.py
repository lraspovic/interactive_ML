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
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.ml.predictor import (
    extract_bbox_features,
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
        "status": "running",
        "progress": 0.0,
        "bbox": body.bbox,
        "timestamp": None,
        "error": None,
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
        "status": "idle",
        "progress": 0.0,
        "bbox": None,
        "timestamp": None,
        "error": None,
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
        X, H, W, valid_mask, feature_names = await asyncio.to_thread(
            extract_bbox_features,
            bbox,
            signed_assets,
            available_bands,
            enabled_indices,
        )

        if X is None or H == 0 or W == 0:
            raise ValueError(
                "Could not read any pixel data from the scene for the given viewport. "
                "Try zooming in or selecting a different scene."
            )

        _PREDICT_JOBS[key]["progress"] = 0.70

        # ── 5. Run prediction + uncertainty ───────────────────────────────────
        predictions = model.predict(X)        # (H*W,) int
        probas      = model.predict_proba(X)  # (H*W, n_classes)

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
            "status":     "done",
            "progress":   1.0,
            "bbox":       bbox,
            "timestamp":  int(time.time()),
            "error":      None,
            "image_size": [W, H],
        }
        logger.info(
            "Prediction complete for project %s — %d×%d px, %d classes",
            project_id, W, H, len(class_colors),
        )

    except Exception as exc:
        logger.exception("Prediction failed for project %s", project_id)
        _PREDICT_JOBS[key] = {
            "status":    "failed",
            "progress":  0.0,
            "bbox":      bbox,
            "timestamp": None,
            "error":     str(exc),
        }
