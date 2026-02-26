"""
Prediction endpoints.

POST /predict                                  — kick off a background prediction job
GET  /predict/status                           — poll prediction job status
GET  /predict/tiles/{z}/{x}/{y}                — serve classification map tiles (colourised on the fly)
GET  /predict/uncertainty/tiles/{z}/{x}/{y}    — serve uncertainty heatmap tiles
GET  /predict/download/{project_id}            — download prediction COG (raw class IDs, uint16)
GET  /predict/uncertainty/download/{project_id}— download uncertainty COG (normalised entropy, float32)
"""
from __future__ import annotations

import asyncio
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from io import BytesIO
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.ml.predictor import (
    extract_bbox_features,
    MAX_SIZE,
    read_raw_bands_tiff,
    write_prediction_cog,
    write_uncertainty_cog,
)
from app.ml.trainer import MODEL_ARTIFACTS_DIR
from app.models import Class, Project
from app.routers.imagery import _get_signed_assets

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

# In-memory job state — same pattern as _TRAIN_JOBS in train.py.
_PREDICT_JOBS: dict[str, dict[str, Any]] = {}

# Short-lived class-colour cache so tile endpoints don't hit the DB every request.
_CLASS_COLOR_CACHE: dict[str, tuple[float, dict[int, str]]] = {}
_CLASS_COLOR_CACHE_TTL = 300  # seconds


async def _get_class_colors(project_id: UUID) -> dict[int, str]:
    """Return {class_id: "#RRGGBB"} for a project, cached for 5 minutes."""
    key = str(project_id)
    cached = _CLASS_COLOR_CACHE.get(key)
    if cached:
        ts, colors = cached
        if time.time() - ts < _CLASS_COLOR_CACHE_TTL:
            return colors
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Class).where(Class.project_id == project_id))
        rows = result.scalars().all()
    colors: dict[int, str] = {c.id: c.color for c in rows}
    _CLASS_COLOR_CACHE[key] = (time.time(), colors)
    return colors


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SceneRef(BaseModel):
    """Reference to one STAC scene used for feature extraction."""
    collection: str
    item_id: str


class PredictRequest(BaseModel):
    project_id: UUID
    bbox: list[float]   # [minlon, minlat, maxlon, maxlat]
    # New multi-sensor API: pass a list of scenes.
    # Legacy API: pass item_id + collection (converted to scenes automatically).
    scenes: list[SceneRef] = []
    item_id: str | None = None          # legacy
    collection: str = "sentinel-2-l2a"  # legacy

    def effective_scenes(self) -> list[SceneRef]:
        if self.scenes:
            return self.scenes
        if self.item_id:
            return [SceneRef(collection=self.collection, item_id=self.item_id)]
        return []


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
    scenes = body.effective_scenes()
    if not scenes:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one scene via 'scenes' or 'item_id'.",
        )

    key = str(body.project_id)
    if _PREDICT_JOBS.get(key, {}).get("status") == "running":
        raise HTTPException(status_code=409, detail="Prediction already in progress")

    _PREDICT_JOBS[key] = {
        "status":      "running",
        "progress":    0.0,
        "bbox":        body.bbox,
        "actual_bbox": None,
        # Store scenes for status reporting (first scene for legacy compat)
        "item_id":     scenes[0].item_id,
        "collection":  scenes[0].collection,
        "timestamp":   None,
        "error":       None,
    }
    background_tasks.add_task(
        _run_prediction,
        body.project_id,
        scenes,
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


@router.get("/tiles/{z}/{x}/{y}")
async def prediction_tile(z: int, x: int, y: int, project_id: UUID = Query(...)):
    """
    Serve a 256×256 RGBA PNG map tile for the latest prediction.
    Class IDs are read from the COG and colourised on the fly using the
    class table so colour changes don't require re-running prediction.
    """
    path = MODEL_ARTIFACTS_DIR / f"pred_{project_id}.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No prediction found. Run a prediction first.")

    class_colors = await _get_class_colors(project_id)

    def _render() -> bytes:
        from PIL import Image
        from rio_tiler.io import COGReader

        try:
            with COGReader(str(path)) as cog:
                img = cog.tile(x, y, z, resampling_method="nearest")
        except Exception as exc:
            if "TileOutsideBounds" in type(exc).__name__ or "outside bounds" in str(exc).lower():
                buf = BytesIO()
                Image.fromarray(np.zeros((256, 256, 4), dtype=np.uint8), mode="RGBA").save(buf, format="PNG")
                return buf.getvalue()
            raise

        class_arr = img.data[0]   # (256, 256) uint16 class IDs
        mask = img.mask           # (256, 256) uint8: 255=valid, 0=nodata
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        for class_id, hex_color in class_colors.items():
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            rgba[(class_arr == class_id) & (mask > 0)] = [r, g, b, 210]

        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    png = await asyncio.to_thread(_render)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/uncertainty/tiles/{z}/{x}/{y}")
async def uncertainty_tile(z: int, x: int, y: int, project_id: UUID = Query(...)):
    """
    Serve a 256×256 RGBA PNG tile for the uncertainty heatmap.
    Normalised entropy (0–1) is read from the COG and mapped
    through a blue→yellow→red colour ramp.
    """
    path = MODEL_ARTIFACTS_DIR / f"uncertainty_{project_id}.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No uncertainty map found.")

    def _render() -> bytes:
        from PIL import Image
        from rio_tiler.io import COGReader

        try:
            with COGReader(str(path)) as cog:
                img = cog.tile(x, y, z, resampling_method="bilinear")
        except Exception as exc:
            if "TileOutsideBounds" in type(exc).__name__ or "outside bounds" in str(exc).lower():
                buf = BytesIO()
                Image.fromarray(np.zeros((256, 256, 4), dtype=np.uint8), mode="RGBA").save(buf, format="PNG")
                return buf.getvalue()
            raise

        ent = img.data[0].astype(np.float32)  # (256, 256) normalised entropy 0–1
        mask = img.mask                        # (256, 256) uint8

        low  = np.array([30,  130, 220], dtype=np.float32)  # blue
        mid  = np.array([250, 220,  30], dtype=np.float32)  # yellow
        high = np.array([220,  30,  30], dtype=np.float32)  # red

        t = ent.ravel()[:, None]
        colors = np.where(
            t <= 0.5,
            low + (mid - low) * (t / 0.5),
            mid + (high - mid) * ((t - 0.5) / 0.5),
        ).astype(np.uint8).reshape(256, 256, 3)

        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[:, :, :3] = colors
        rgba[:, :, 3] = np.where(mask.reshape(256, 256) > 0, 200, 0).astype(np.uint8)

        buf = BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    png = await asyncio.to_thread(_render)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/download/{project_id}")
async def download_prediction(project_id: str):
    """
    Download the prediction COG (uint16 class IDs, EPSG:3857, deflate-compressed).
    Open in QGIS, GDAL, or any GIS tool.  Apply the project class→colour table
    to render the classes.
    """
    path = MODEL_ARTIFACTS_DIR / f"pred_{project_id}.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No prediction found. Run a prediction first.")
    short = project_id[:8]
    return FileResponse(
        str(path),
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="prediction_{short}.tif"'},
    )


@router.get("/uncertainty/download/{project_id}")
async def download_uncertainty(project_id: str):
    """
    Download the uncertainty COG (float32 normalised entropy 0–1, EPSG:3857).
    Values near 1 indicate high uncertainty; values near 0 indicate high confidence.
    """
    path = MODEL_ARTIFACTS_DIR / f"uncertainty_{project_id}.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No uncertainty map found.")
    short = project_id[:8]
    return FileResponse(
        str(path),
        media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="uncertainty_{short}.tif"'},
    )


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
        read_raw_bands_tiff, bbox, {collection: signed_assets}, available_bands, enabled_indices
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
    scenes: list[SceneRef],
    bbox: list[float],
) -> None:
    key = str(project_id)
    first_scene = scenes[0]
    try:
        # ── 1. Load project config ────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            project = await db.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project_id} not found")
            available_bands: list[str] = project.available_bands or ["blue", "green", "red"]
            enabled_indices: list[str] = project.enabled_indices or []
            glcm_config: dict | None = project.glcm_config
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

        # ── 3. Fetch signed asset hrefs per collection (cached) ──────────────
        sensors_assets: dict[str, dict] = {}
        for scene in scenes:
            sensors_assets[scene.collection] = await asyncio.to_thread(
                _get_signed_assets, scene.collection, scene.item_id
            )

        _PREDICT_JOBS[key]["progress"] = 0.25

        # ── 4. Extract bbox features ─────────────────────────────────────────
        X, H, W, valid_mask, feature_names, actual_bbox, utm_transform, native_crs = (
            await asyncio.to_thread(
                extract_bbox_features,
                bbox,
                sensors_assets,
                available_bands,
                enabled_indices,
                MAX_SIZE,
                glcm_config,
            )
        )
        # Fall back to the requested bbox if feature extraction had no data
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

        # ── 6. Write prediction and uncertainty COGs ─────────────────────────────
        MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            write_prediction_cog,
            predictions, H, W, valid_mask, utm_transform, native_crs,
            MODEL_ARTIFACTS_DIR / f"pred_{project_id}.tif",
        )
        await asyncio.to_thread(
            write_uncertainty_cog,
            probas, H, W, valid_mask, utm_transform, native_crs,
            MODEL_ARTIFACTS_DIR / f"uncertainty_{project_id}.tif",
        )

        _PREDICT_JOBS[key] = {
            "status":      "done",
            "progress":    1.0,
            "bbox":        bbox,
            "actual_bbox": actual_bbox,
            "item_id":     first_scene.item_id,
            "collection":  first_scene.collection,
            "timestamp":   int(time.time()),
            "error":       None,
            "image_size":  [W, H],
        }
        logger.info(
            "Prediction complete for project %s — %d×%d px",
            project_id, W, H,
        )

    except Exception as exc:
        logger.exception("Prediction failed for project %s", project_id)
        _PREDICT_JOBS[key] = {
            "status":      "failed",
            "progress":    0.0,
            "bbox":        bbox,
            "actual_bbox": None,
            "item_id":     first_scene.item_id,
            "collection":  first_scene.collection,
            "timestamp":   None,
            "error":       str(exc),
        }
