"""
Satellite imagery endpoints.

- GET /imagery/stac-search              — query Planetary Computer STAC, return scene list
- GET /imagery/stac-item/{coll}/{id}    — fetch + sign one STAC item (JSON proxy)
- GET /imagery/stac/tiles/{z}/{x}/{y}  — serve map tiles directly from COG assets
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta
from typing import List, Optional

import httpx
import numpy as np
import planetary_computer
import pystac
import pystac_client
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/imagery", tags=["imagery"])

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Optional PC subscription key (not required for Sentinel-2 L2A, but widens
# rate limits on other collections).
PC_SDK_SUBSCRIPTION_KEY = os.getenv("PC_SDK_SUBSCRIPTION_KEY", "")

# ---------------------------------------------------------------------------
# In-process TTL cache for signed STAC item assets
# PC signed URLs are valid for ~1 hour; we cache for 50 min so every tile
# in a viewport shares one PC API round-trip instead of one per tile.
# ---------------------------------------------------------------------------
import time as _time
from concurrent.futures import ThreadPoolExecutor

_ITEM_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_ITEM_CACHE_TTL = 50 * 60  # 50 minutes

# ---------------------------------------------------------------------------
# Tile-level PNG cache — keyed by (item_id, collection, assets, z, x, y,
# rescale, color_formula).  Bounded to 1 000 entries; oldest evicted when full.
# At ~150 KB average per tile this caps memory usage at ~150 MB.
# ---------------------------------------------------------------------------
_TILE_CACHE: dict[tuple, bytes] = {}
_TILE_CACHE_MAX = 1000


def _tile_cache_get(key: tuple) -> bytes | None:
    return _TILE_CACHE.get(key)


def _tile_cache_set(key: tuple, png: bytes) -> None:
    if len(_TILE_CACHE) >= _TILE_CACHE_MAX:
        # Evict the oldest 10 % of entries
        evict = list(_TILE_CACHE.keys())[:_TILE_CACHE_MAX // 10]
        for k in evict:
            _TILE_CACHE.pop(k, None)
    _TILE_CACHE[key] = png


def _get_signed_assets(collection: str, item_id: str) -> dict:
    """Return signed asset href dict for the item, using a short-lived cache."""
    key = (collection, item_id)
    cached = _ITEM_CACHE.get(key)
    if cached:
        ts, asset_map = cached
        if _time.monotonic() - ts < _ITEM_CACHE_TTL:
            return asset_map
    item_url = f"{PC_STAC_URL}/collections/{collection}/items/{item_id}"
    resp = httpx.get(item_url, timeout=15)
    resp.raise_for_status()
    signed = planetary_computer.sign(resp.json())
    asset_map = signed.get("assets", {})
    _ITEM_CACHE[key] = (_time.monotonic(), asset_map)
    return asset_map


def _pc_client() -> pystac_client.Client:
    """Open a pystac Client against Planetary Computer."""
    modifier = planetary_computer.sign_inplace
    return pystac_client.Client.open(PC_STAC_URL, modifier=modifier)


# ---------------------------------------------------------------------------
# Scene search
# ---------------------------------------------------------------------------

@router.get("/stac-search")
async def stac_search(
    bbox: str = Query(..., description="minlon,minlat,maxlon,maxlat"),
    collection: str = Query("sentinel-2-l2a"),
    date_from: Optional[str] = Query(
        None,
        description="ISO date YYYY-MM-DD, defaults to 60 days ago",
    ),
    date_to: Optional[str] = Query(
        None,
        description="ISO date YYYY-MM-DD, defaults to today",
    ),
    max_cloud: float = Query(20.0, ge=0, le=100),
    limit: int = Query(12, ge=1, le=50),
):
    """
    Search Planetary Computer for scenes intersecting the given bbox.
    Returns a list of lightweight scene descriptors; assets are NOT downloaded.
    """
    try:
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        bbox_list = parts  # [minlon, minlat, maxlon, maxlat]
    except ValueError:
        raise HTTPException(status_code=422, detail="bbox must be 'minlon,minlat,maxlon,maxlat'")

    # Date range defaults
    to_date = date_to or date.today().isoformat()
    from_date = date_from or (date.today() - timedelta(days=60)).isoformat()
    datetime_range = f"{from_date}/{to_date}"

    try:
        catalog = _pc_client()
        search = catalog.search(
            collections=[collection],
            bbox=bbox_list,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            limit=limit,
            sortby=["-properties.datetime"],
        )
        items = list(search.items())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"STAC search failed: {exc}")

    results = []
    for item in items[:limit]:
        props = item.properties
        # thumbnail — PC provides a rendered_preview asset for many collections
        thumb = None
        if "rendered_preview" in item.assets:
            thumb = item.assets["rendered_preview"].href
        elif "overview" in item.assets:
            thumb = item.assets["overview"].href

        results.append(
            {
                "id": item.id,
                "collection": collection,
                "date": props.get("datetime", "")[:10],
                "cloud_pct": round(props.get("eo:cloud_cover", -1), 1),
                "thumbnail": thumb,
                # self link — used to build the titiler proxy URL
                "self_url": item.get_self_href() or f"{PC_STAC_URL}/collections/{collection}/items/{item.id}",
                # available raster assets (skip metadata-only assets)
                "assets": [
                    k for k, v in item.assets.items()
                    if v.media_type and "tiff" in v.media_type
                ],
            }
        )

    return results


# ---------------------------------------------------------------------------
# Signed item proxy (used as ?url= for titiler)
# ---------------------------------------------------------------------------

@router.get("/stac-item/{collection}/{item_id}")
async def stac_item(collection: str, item_id: str):
    """
    Fetch a STAC item from Planetary Computer, sign all asset hrefs,
    and return the signed item JSON.

    titiler calls this endpoint when the frontend passes
    ?url=http://backend:8000/imagery/stac-item/{collection}/{item_id}
    """
    item_url = f"{PC_STAC_URL}/collections/{collection}/items/{item_id}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(item_url)
            resp.raise_for_status()
            item_dict = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"Item not found: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch STAC item: {exc}")

    # Sign all asset hrefs in-place
    try:
        signed = planetary_computer.sign(item_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signing failed: {exc}")

    return JSONResponse(content=signed)


# ---------------------------------------------------------------------------
# COG tile endpoint — reads directly from signed PC assets via rio_tiler
# ---------------------------------------------------------------------------

@router.get("/stac/tiles/{z}/{x}/{y}")
async def stac_tile(
    z: int,
    x: int,
    y: int,
    item_id: str = Query(..., description="STAC item ID"),
    collection: str = Query("sentinel-2-l2a"),
    assets: List[str] = Query(default=["B04", "B03", "B02"]),
    rescale: str = Query("0,3000", description="'min,max' used to stretch to uint8"),
    color_formula: Optional[str] = Query(None, description="rio_tiler color formula"),
):
    """
    Serve a 256×256 PNG map tile for given STAC item + asset combination.
    Signs item server-side so no credentials reach the browser.
    """
    try:
        rmin, rmax = [float(v.strip()) for v in rescale.split(",")]
    except ValueError:
        raise HTTPException(status_code=422, detail="rescale must be 'min,max'")

    # Check tile cache first — cache key captures everything that affects the output
    cache_key = (item_id, collection, tuple(assets), z, x, y, rescale, color_formula or "")
    cached_png = _tile_cache_get(cache_key)
    if cached_png is not None:
        return Response(
            content=cached_png,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600", "X-Tile-Cache": "HIT"},
        )

    def _read_band(asset_name: str):
        """Read one COG band — runs in a thread-pool worker."""
        from rio_tiler.io import COGReader

        signed_assets = _get_signed_assets(collection, item_id)
        asset_info = signed_assets.get(asset_name)
        if asset_info is None:
            raise ValueError(f"Asset '{asset_name}' not found in item '{item_id}'")
        with COGReader(asset_info["href"]) as cog:
            return cog.tile(x, y, z, resampling_method="bilinear", nodata=0)

    def _render_tile() -> bytes:
        from rio_tiler.models import ImageData

        # Read all bands in parallel — each COG is a separate HTTP round-trip,
        # so parallelising cuts wall-clock time by ~N-bands.
        with ThreadPoolExecutor(max_workers=len(assets)) as pool:
            band_images = list(pool.map(_read_band, assets))

        if len(band_images) == 1:
            merged = band_images[0]
        else:
            all_data = np.vstack([b.data for b in band_images])   # (bands, 256, 256)
            combined_mask = np.min(
                np.stack([b.mask for b in band_images], axis=0), axis=0
            ).astype(np.uint8)

            # Any pixel where all bands are 0 is outside the swath — force transparent
            outside_swath = np.all(all_data == 0, axis=0)
            combined_mask[outside_swath] = 0

            merged = ImageData(all_data, combined_mask)

        merged.rescale(
            in_range=[(rmin, rmax)] * len(assets),
            out_range=[(0, 255)] * len(assets),
        )
        if color_formula:
            merged.apply_color_formula(color_formula)

        return merged.render(img_format="PNG")

    try:
        png_bytes = await asyncio.to_thread(_render_tile)
        _tile_cache_set(cache_key, png_bytes)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=str(exc))
    except Exception as exc:
        # Tile outside bounds is normal at edges — return transparent tile
        if "TileOutsideBounds" in type(exc).__name__ or "outside bounds" in str(exc).lower():
            from rio_tiler.models import ImageData
            empty = np.zeros((len(assets), 256, 256), dtype=np.uint8)
            alpha = np.zeros((256, 256), dtype=np.uint8)
            png_bytes = ImageData(empty, alpha).render(img_format="PNG")
        else:
            raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )
