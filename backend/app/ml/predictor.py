"""
Bbox-based feature extraction and COG rendering for the prediction pipeline.

All pixel data stays in the Sentinel-2 scene’s native UTM CRS throughout —
no intermediate reprojection to EPSG:3857.  This means:
  • Feature values fed to the model are taken straight from the source pixels.
  • The written COGs have an exact, lossless affine transform.
  • rio_tiler re-projects each 256×256 tile from UTM to Web Mercator on the
    fly at serve-time, which is identical to what it does for the S-2 imagery
    tiles — guaranteeing perfect alignment.

extract_bbox_features: reads all configured bands in parallel at native pixel
  resolution (no GSD assumption), with an optional max_size cap for prediction
  performance.  Nearest-neighbour resampling preserves original source DN values.
  Bands + spectral indices are stacked into a (H*W, n_features) float32 matrix.
  Returns utm_transform (Affine) and native_crs (rasterio.crs.CRS) so downstream
  writers receive the exact grid.

write_prediction_cog: uint16 tiled COG, class IDs as pixel values, native UTM.
write_uncertainty_cog: float32 tiled COG, normalised Shannon entropy, native UTM.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from app.ml.features import S2_BAND_TO_ASSET, SPECTRAL_INDICES

logger = logging.getLogger(__name__)

# Maximum pixels along the longer axis when reading a bbox COG region.
# Keeps prediction < 1 s and the PNG file small.  Users can zoom in and
# re-predict for higher-resolution output.
MAX_SIZE = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_band_part(
    href: str,
    bbox: list[float],
    band_name: str,
    max_size: int | None,
) -> tuple[str, np.ndarray | None, np.ndarray | None, tuple | None, object | None]:
    """
    Read a rectangular bbox window from one COG asset in its native UTM CRS.

    The output shape is derived from the native window pixel count so that
    pixel values are read without interpolation (nearest-neighbour resampling).
    When *max_size* is set and the native dimensions exceed it, the output is
    downsampled while preserving aspect ratio — still with nearest-neighbour
    so values remain original DNs rather than blended artefacts.

    Returns
    -------
    band_name      : str
    arr            : float32 (H, W) array in native UTM, or None on failure
    mask           : uint8 (H, W) — 255 = valid, 0 = nodata
    actual_native  : (left, bottom, right, top) bounds in native UTM metres
    native_crs     : rasterio.crs.CRS of the source dataset
    """
    import rasterio
    import rasterio.enums
    from rasterio import windows, warp

    try:
        with rasterio.open(href) as ds:
            # 1. Find the window in native UTM that covers the requested WGS84 bbox
            minlon, minlat, maxlon, maxlat = bbox
            native_bounds = warp.transform_bounds(
                "epsg:4326", ds.crs, minlon, minlat, maxlon, maxlat
            )
            win = windows.from_bounds(*native_bounds, transform=ds.transform)
            win = win.intersection(windows.Window(0, 0, ds.width, ds.height))
            if win.width <= 0 or win.height <= 0:
                return band_name, None, None, None, None

            # Snapped bounds in native UTM — pixel-grid-aligned
            actual_native = windows.bounds(win, ds.transform)
            nodata_val = float(ds.nodata) if ds.nodata is not None else 0.0
            native_crs = ds.crs

            # Derive output dimensions from the native pixel count of the window.
            # This gives exact source DN values (no interpolation required).
            out_h = max(1, round(win.height))
            out_w = max(1, round(win.width))

            # Optional cap: downsample for prediction performance while preserving
            # aspect ratio.  Still uses nearest-neighbour to avoid blended values.
            if max_size is not None and max(out_w, out_h) > max_size:
                scale = max_size / max(out_w, out_h)
                out_w = max(1, round(out_w * scale))
                out_h = max(1, round(out_h * scale))

            # nearest preserves original source DN values
            arr = ds.read(
                1, window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.float32)

        mask = np.where(arr != nodata_val, np.uint8(255), np.uint8(0))
        return band_name, arr, mask, actual_native, native_crs
    except Exception as exc:
        logger.warning("Part read failed for band %s: %s", band_name, exc)
        return band_name, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bbox_features(
    bbox: list[float],
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str],
    max_size: int = MAX_SIZE,
) -> tuple:
    """
    Extract a (H*W, n_features) feature matrix for a rectangular bounding box.

    Bands are read at native pixel resolution (nearest-neighbour resampling) so
    pixel values fed to the classifier are identical to the source scene values.
    Mixed-resolution bands (e.g. 10 m + 20 m S2) are aligned to the shared
    pixel grid by snapping to the minimum common dimensions.
    An optional *max_size* cap limits the pixel dimensions for prediction
    performance (nearest-neighbour is used even when downsampling).

    All bands are read in native UTM (no CRS change).  The shared utm_transform
    + native_crs are returned so the caller can write COGs with an exact affine.

    Returns
    -------
    X                 : (H*W, n_features) float32, or None on failure
    H, W              : pixel dimensions of the raster window
    valid_mask        : (H*W,) bool — True where all bands had valid data
    feature_names     : list[str]
    actual_bounds_wgs84 : [west, south, east, north] EPSG:4326 (for status reporting)
    utm_transform     : rasterio Affine for the (H, W) grid in native UTM
    native_crs        : rasterio.crs.CRS of the source scene
    """
    # Resolve band names → signed asset hrefs
    band_hrefs: dict[str, str] = {}
    for logical in available_bands:
        asset_key = S2_BAND_TO_ASSET.get(logical)
        if not asset_key:
            continue
        asset_info = signed_assets.get(asset_key)
        if asset_info and asset_info.get("href"):
            band_hrefs[logical] = asset_info["href"]

    if not band_hrefs:
        logger.warning("No matching assets found for bands %s", available_bands)
        return None, 0, 0, None, [], None, None, None

    # Read all bands in parallel
    band_arrays: dict[str, np.ndarray] = {}
    band_masks: dict[str, np.ndarray] = {}
    shapes: list[tuple[int, int]] = []
    actual_native_utm: tuple | None = None  # (left, bottom, right, top) in native UTM
    native_crs = None

    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_part, href, bbox, name, max_size): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, mask, a_native, crs = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                band_masks[band_name] = mask
                shapes.append(arr.shape)
                if actual_native_utm is None:
                    actual_native_utm = a_native
                    native_crs = crs

    if not band_arrays:
        return None, 0, 0, None, [], None, None, None

    # Snap all bands to the common (minimum) pixel dimensions.
    # Since all bands cover the same actual_native_utm extent, the snapped
    # array corresponds to a coarser GSD but the same spatial footprint —
    # we recompute the affine from the fixed extent + snapped dims so it is exact.
    H = min(s[0] for s in shapes)
    W = min(s[1] for s in shapes)
    for bn in list(band_arrays.keys()):
        band_arrays[bn] = band_arrays[bn][:H, :W]
        band_masks[bn] = band_masks[bn][:H, :W]

    # Exact affine for the snapped (H, W) grid in native UTM
    from rasterio.transform import from_bounds as _from_bounds
    from rasterio import warp as _warp
    utm_transform = _from_bounds(*actual_native_utm, W, H)

    # WGS84 bounds for job-status reporting (human-readable, used by the frontend)
    west, south, east, north = _warp.transform_bounds(
        native_crs, "epsg:4326", *actual_native_utm
    )
    actual_bounds_wgs84 = [west, south, east, north]

    # A pixel is valid only if ALL bands report valid data there
    combined_mask = np.ones((H, W), dtype=bool)
    for mask in band_masks.values():
        combined_mask &= (mask > 0)

    # Build flat feature columns (same ordering as extract_pixel_features)
    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    for logical in available_bands:
        if logical in band_arrays:
            feature_cols.append(band_arrays[logical].ravel())
            feature_names.append(logical)

    band_1d = {k: v.ravel() for k, v in band_arrays.items()}
    for idx_name in enabled_indices:
        spec = SPECTRAL_INDICES.get(idx_name.upper())
        if spec is None:
            continue
        if not spec["requires"].issubset(set(band_arrays.keys())):
            logger.debug("Skipping %s — missing required bands", idx_name)
            continue
        idx_values = spec["fn"]({k: band_1d[k] for k in band_1d}).astype(np.float32)
        feature_cols.append(idx_values)
        feature_names.append(idx_name.upper())

    if not feature_cols:
        return None, H, W, combined_mask.ravel(), feature_names, actual_bounds_wgs84, utm_transform, native_crs

    X = np.column_stack(feature_cols)   # (H*W, n_features)
    return X, H, W, combined_mask.ravel(), feature_names, actual_bounds_wgs84, utm_transform, native_crs


def read_raw_bands_tiff(
    bbox: list[float],
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str] | None = None,
    max_size: int | None = None,
) -> bytes | None:
    """
    Read raw band pixel data for a bbox and return a georeferenced multi-band
    GeoTIFF as bytes.  Raw bands come first, then any computable spectral
    indices (NDVI, NDWI, …) as additional bands.  Band names are stored as
    raster tags.  CRS: native UTM of the source Sentinel-2 scene.

    *max_size* defaults to None (no downsampling) so the downloaded TIFF
    contains the exact native pixel values at the scene's true resolution.
    Pass an explicit value only when download size needs to be limited.

    Open in QGIS/ArcGIS to inspect the exact pixel values fed to the model.
    """
    import rasterio
    from rasterio.io import MemoryFile

    band_hrefs: dict[str, str] = {}
    for logical in available_bands:
        asset_key = S2_BAND_TO_ASSET.get(logical)
        if not asset_key:
            continue
        asset_info = signed_assets.get(asset_key)
        if asset_info and asset_info.get("href"):
            band_hrefs[logical] = asset_info["href"]

    if not band_hrefs:
        logger.warning("No matching assets found for bands %s", available_bands)
        return None

    band_arrays: dict[str, np.ndarray] = {}
    actual_bounds_utm: tuple | None = None
    native_crs = None

    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_part, href, bbox, name, max_size): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, _mask, a_native, crs = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                if actual_bounds_utm is None:
                    actual_bounds_utm = a_native
                    native_crs = crs

    if not band_arrays:
        return None

    H = min(a.shape[0] for a in band_arrays.values())
    W = min(a.shape[1] for a in band_arrays.values())
    band_order = [b for b in available_bands if b in band_arrays]
    arrays = [band_arrays[b][:H, :W] for b in band_order]

    # Compute spectral indices and append as extra bands
    if enabled_indices:
        band_1d = {b: band_arrays[b][:H, :W].ravel() for b in band_order}
        for idx_name in enabled_indices:
            spec = SPECTRAL_INDICES.get(idx_name.upper())
            if spec is None:
                continue
            if not spec["requires"].issubset(set(band_order)):
                logger.debug("TIFF: skipping %s — missing required bands", idx_name)
                continue
            idx_arr = spec["fn"]({k: band_1d[k] for k in band_1d}).astype(np.float32).reshape(H, W)
            arrays.append(idx_arr)
            band_order.append(idx_name.upper())

    # UTM affine consistent with the snapped (H, W) grid — no WGS84 roundtrip
    from rasterio.transform import from_bounds as _fb
    transform = _fb(*actual_bounds_utm, W, H)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=H,
            width=W,
            count=len(arrays),
            dtype="float32",
            crs=native_crs,
            transform=transform,
            nodata=0.0,
        ) as ds:
            for i, (arr, name) in enumerate(zip(arrays, band_order), start=1):
                ds.write(arr, i)
                ds.update_tags(i, name=name)
        return memfile.read()


def write_prediction_cog(
    predictions: np.ndarray,   # (H*W,) int32 — predicted class ID per pixel
    H: int,
    W: int,
    valid_mask: np.ndarray,    # (H*W,) bool — True where data is valid
    native_transform,          # rasterio Affine for the (H, W) pixel grid
    native_crs,                # rasterio.crs.CRS — native UTM of the scene
    out_path: Path | str,
) -> None:
    """
    Write a Cloud-Optimised GeoTIFF (uint16) where each pixel holds its
    predicted class ID.  nodata=0; class IDs from the DB start at 1.

    CRS: native UTM of the source Sentinel-2 scene (read from ds.crs).
    rio_tiler reprojects to Web Mercator at tile-serve time, guaranteeing
    pixel-perfect alignment with the S-2 imagery tiles.
    Overviews use nearest-neighbour resampling to preserve class boundaries.
    """
    import rasterio
    import rasterio.shutil as rio_shutil
    from rasterio.enums import Resampling

    NODATA_VAL: int = 0
    tmp_path = str(out_path) + ".tmp.tif"

    data = predictions.reshape(H, W).astype(np.uint16)
    data[~valid_mask.reshape(H, W)] = NODATA_VAL

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            height=H, width=W,
            count=1, dtype="uint16",
            crs=native_crs,
            transform=native_transform,
            nodata=NODATA_VAL,
            tiled=True, blockxsize=256, blockysize=256,
        ) as ds:
            ds.write(data, 1)
            ds.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
            ds.update_tags(ns="rio_overview", resampling="nearest")

        with rasterio.open(tmp_path) as src:
            rio_shutil.copy(
                src, str(out_path),
                driver="GTiff",
                copy_src_overviews=True,
                tiled=True, blockxsize=512, blockysize=512,
                compress="deflate",
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_uncertainty_cog(
    probas: np.ndarray,   # (H*W, n_classes) float32 — predict_proba output
    H: int,
    W: int,
    valid_mask: np.ndarray,  # (H*W,) bool
    native_transform,        # rasterio Affine for the (H, W) pixel grid
    native_crs,              # rasterio.crs.CRS — native UTM of the scene
    out_path: Path | str,
) -> None:
    """
    Write a Cloud-Optimised GeoTIFF (float32) with per-pixel normalised Shannon
    entropy in [0, 1].  nodata=-1.  Overviews use average resampling.

    CRS: native UTM of the source Sentinel-2 scene.
    The blue→yellow→red colour ramp is applied at tile-serve time so the
    colourmap can be changed without rerunning prediction.
    """
    import rasterio
    import rasterio.shutil as rio_shutil
    from rasterio.enums import Resampling

    NODATA_VAL: float = -1.0
    tmp_path = str(out_path) + ".tmp.tif"

    p = np.clip(probas, 1e-10, 1.0)
    entropy = (-np.sum(p * np.log(p), axis=1)).astype(np.float32)
    max_e = float(np.log(max(probas.shape[1], 2)))
    norm_entropy = np.clip(entropy / max_e, 0.0, 1.0)
    norm_entropy[~valid_mask] = NODATA_VAL

    data = norm_entropy.reshape(H, W)

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            height=H, width=W,
            count=1, dtype="float32",
            crs=native_crs,
            transform=native_transform,
            nodata=NODATA_VAL,
            tiled=True, blockxsize=256, blockysize=256,
        ) as ds:
            ds.write(data, 1)
            ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            ds.update_tags(ns="rio_overview", resampling="average")

        with rasterio.open(tmp_path) as src:
            rio_shutil.copy(
                src, str(out_path),
                driver="GTiff",
                copy_src_overviews=True,
                tiled=True, blockxsize=512, blockysize=512,
                compress="deflate",
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
