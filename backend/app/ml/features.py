"""
Feature extraction from COG assets for training polygon geometries.

For each polygon:
  - fetch pixel values for each configured band from the appropriate STAC
    collection (Sentinel-2 L2A or Sentinel-1 RTC) via rasterio range requests
  - compute enabled spectral indices from those raw values using the full
    ASI catalogue in spectral_catalogue.py
  - optionally compute GLCM texture features (per polygon, broadcast to pixels)
  - return (n_valid_pixels, n_features) float32 array + feature_names + WGS84
    pixel centroid coordinates

Supports multi-sensor training: when both Sentinel-2 and Sentinel-1 scenes are
provided their bands are extracted against the same polygon, resampled to a
shared pixel grid (defined by the first-band native pixel count of the primary
sensor), and column-stacked into a single feature matrix.

COG reads use HTTP range requests — no full imagery download is required.
Band reads for a single polygon are parallelised with a ThreadPoolExecutor.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from app.ml.spectral_catalogue import (
    COLLECTION_BAND_TO_ASSET,
    LEGACY_BAND_ALIASES,
    compute_index,
    SPECTRAL_INDEX_CATALOGUE,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10

# Maximum number of pixels to use for GLCM computation per polygon.
# GLCM is O(n_pixels × window_size²) — cap keeps training time predictable.
GLCM_MAX_PIXELS = 500


def _read_band_pixels(
    href: str,
    geometry: dict,
    band_name: str,
    target_shape: tuple[int, int] | None = None,
) -> tuple[str, np.ndarray | None, np.ndarray | None, Any | None, Any | None]:
    """
    Read pixels inside *geometry* from one COG asset.

    When *target_shape* is None the output shape is derived from the native
    window pixel count (``round(win.height) × round(win.width)``), which
    preserves the original DN values with no interpolation.  When a shared
    *target_shape* is supplied (to align a 20 m band onto a 10 m grid) nearest-
    neighbour resampling is used so each output pixel still carries an original
    source DN — never an interpolated value.

    Uses HTTP range requests — only the bytes covering the polygon bbox
    are downloaded from Azure Blob Storage.

    Returns (band_name, 2D float32 array, 2D bool valid_mask, out_transform, native_crs)
    or (band_name, None, None, None, None) on failure.
    """
    import rasterio
    import rasterio.features as rio_feat
    from rasterio import windows, warp
    from rasterio.transform import from_bounds as transform_from_bounds
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    try:
        with rasterio.open(href) as ds:
            # Transform polygon bbox from WGS84 to native COG CRS
            aoi_bounds = rio_feat.bounds(geometry)   # (minx, miny, maxx, maxy) WGS84
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            win = windows.from_bounds(*native_bounds, transform=ds.transform)
            win = win.intersection(windows.Window(0, 0, ds.width, ds.height))
            if win.width <= 0 or win.height <= 0:
                return band_name, None, None, None, None

            # Determine output shape.
            # When no target is supplied, use the native pixel count of the window
            # so the read is lossless (no interpolation, exact source DN values).
            # When a shared target is supplied (aligning a 20 m band to a 10 m
            # grid), nearest-neighbour resampling is used so each output pixel
            # still holds an original source value.
            if target_shape is None:
                tgt_h = max(1, round(win.height))
                tgt_w = max(1, round(win.width))
                target_shape = (tgt_h, tgt_w)

            out_h, out_w = target_shape

            # nearest preserves original DN values; no interpolation artefacts
            data = ds.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.float32)

            # Transform for the resampled output grid
            out_transform = transform_from_bounds(
                native_bounds[0], native_bounds[1],
                native_bounds[2], native_bounds[3],
                out_w, out_h,
            )

            # Build polygon mask in native CRS at output resolution
            t = Transformer.from_crs(4326, ds.crs, always_xy=True)
            native_geom = shp_transform(t.transform, shape(geometry))
            poly_mask = rio_feat.geometry_mask(
                [mapping(native_geom)],
                out_shape=(out_h, out_w),
                transform=out_transform,
                invert=True,
            )
            nodata_val = ds.nodata if ds.nodata is not None else 0
            valid = poly_mask & (data != nodata_val)
            if not valid.any():
                return band_name, None, None, None, None

            crs = ds.crs

        return band_name, data, valid, out_transform, crs
    except Exception as exc:
        logger.warning("Feature read failed for band %s: %s", band_name, exc)
        return band_name, None, None, None, None


def compute_glcm_features(
    band_data: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int,
    statistics: list[str],
    levels: int = 64,
) -> dict[str, float]:
    """
    Compute GLCM texture statistics over the valid region of a 2-D band patch.

    Angles 0°, 45°, 90°, 135° are always used; results are averaged across
    angles so each statistic produces one scalar per band.

    Parameters
    ----------
    band_data:    2D float32 array (full window including invalid pixels).
    valid_mask:   2D bool array — True where pixels are inside the polygon.
    window_size:  Neighbourhood size for GLCM (must be odd; 3 / 5 / 7 / 9).
    statistics:   Subset of {"contrast","dissimilarity","homogeneity",
                  "energy","correlation","ASM"} to compute.
    levels:       Number of grey levels to quantise to (default 64 for speed).

    Returns
    -------
    Dict of {stat_name: scalar_float}.  Empty dict on failure.
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
    except ImportError:
        logger.warning("scikit-image not installed — GLCM features skipped")
        return {}

    try:
        roi = band_data[valid_mask]
        if roi.size == 0:
            return {}
        vmin, vmax = float(roi.min()), float(roi.max())
        if vmax == vmin:
            # Constant patch — return zeros for all requested statistics
            return {s: 0.0 for s in statistics}

        # Quantise the entire patch to [0, levels-1]
        quantised = np.clip(
            ((band_data - vmin) / (vmax - vmin) * (levels - 1)).astype(np.uint8),
            0, levels - 1,
        )

        half = window_size // 2
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        stats_accum: dict[str, list[float]] = {s: [] for s in statistics}

        # Slide the window over all valid pixel positions.
        # Caller is responsible for capping pixel count before calling.
        rows, cols = np.where(valid_mask)
        for r, c in zip(rows, cols):
            r0, r1 = max(0, r - half), min(band_data.shape[0], r + half + 1)
            c0, c1 = max(0, c - half), min(band_data.shape[1], c + half + 1)
            patch = quantised[r0:r1, c0:c1]
            if patch.size < 4:
                continue
            glcm = graycomatrix(
                patch, distances=[1], angles=angles,
                levels=levels, symmetric=True, normed=True,
            )
            for stat in statistics:
                val = float(graycoprops(glcm, stat).mean())
                stats_accum[stat].append(val)

        return {s: float(np.mean(vs)) if vs else 0.0 for s, vs in stats_accum.items()}

    except Exception as exc:
        logger.warning("GLCM computation failed: %s", exc)
        return {}


def extract_pixel_features(
    geometry: dict,
    sensors_assets: dict[str, dict],
    available_bands: list[str],
    enabled_indices: list[str],
    glcm_config: dict | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray | None]:
    """
    Extract a (n_valid_pixels, n_features) float32 feature matrix for one polygon.

    Parameters
    ----------
    geometry:
        GeoJSON Polygon geometry dict (EPSG:4326).
    sensors_assets:
        ``{collection_id: signed_asset_dict}`` — may contain one or both of
        ``"sentinel-2-l2a"`` and ``"sentinel-1-rtc"``.
        For backward compatibility a plain ``{asset_key: asset_info}`` dict
        (legacy single-sensor format) is also accepted; it is treated as
        Sentinel-2 L2A.
    available_bands:
        Ordered list of logical band names configured for the project (union of
        all sensor bands).
    enabled_indices:
        List of spectral index short names to compute (e.g. ``["NDVI", "NBR"]``).
    glcm_config:
        Optional GLCM configuration dict::

            {
                "enabled": True,
                "bands":       ["nir", "sar_vv"],
                "window_size": 5,
                "statistics":  ["contrast", "dissimilarity", "homogeneity",
                                 "energy", "correlation"],
            }

    Returns
    -------
    (X, feature_names, coords_wgs84)
    X is None if no valid pixels were found.
    coords_wgs84 is a (n_pixels, 2) float64 ``[lon, lat]`` array or None.
    """
    # ── 0. Backwards-compat: if legacy flat dict is passed, wrap it ────────
    # A legacy call passes signed_assets = {"B02": {"href": ...}, ...}.
    # The new API expects {"sentinel-2-l2a": {"B02": {"href": ...}, ...}}.
    # Detect by checking if any key matches a known collection name.
    known_collections = set(COLLECTION_BAND_TO_ASSET.keys())
    if not any(k in known_collections for k in sensors_assets):
        # Legacy format — wrap as S2
        sensors_assets = {"sentinel-2-l2a": sensors_assets}

    # ── 1. Resolve band → href via per-collection asset maps ────────────────
    band_hrefs: dict[str, str] = {}

    for collection, signed_assets in sensors_assets.items():
        band_map = COLLECTION_BAND_TO_ASSET.get(collection, {})
        for band_name in available_bands:
            if band_name in band_hrefs:
                continue
            # Apply legacy alias (e.g. "red_edge" → "nir_2")
            resolved = LEGACY_BAND_ALIASES.get(band_name, band_name)
            asset_key = band_map.get(resolved) or band_map.get(band_name)
            if not asset_key:
                continue
            asset_info = signed_assets.get(asset_key)
            if asset_info and asset_info.get("href"):
                band_hrefs[band_name] = asset_info["href"]

    if not band_hrefs:
        logger.warning("No matching assets found for bands %s", available_bands)
        return None, [], None

    # ── 2. Pre-compute shared target_shape from first band ─────────────────
    target_shape: tuple[int, int] | None = None
    try:
        import rasterio
        import rasterio.features as rio_feat
        from rasterio import warp, windows as rio_windows
        first_href = next(iter(band_hrefs.values()))
        aoi_bounds = rio_feat.bounds(geometry)
        with rasterio.open(first_href) as ds:
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            win0 = rio_windows.from_bounds(*native_bounds, transform=ds.transform)
            win0 = win0.intersection(rio_windows.Window(0, 0, ds.width, ds.height))
        tgt_h = max(1, round(win0.height))
        tgt_w = max(1, round(win0.width))
        target_shape = (tgt_h, tgt_w)
    except Exception as exc:
        logger.warning("Could not pre-compute target_shape, falling back to native: %s", exc)

    # ── 3. Fetch all bands in parallel ──────────────────────────────────────
    band_arrays: dict[str, np.ndarray] = {}
    band_masks:  dict[str, np.ndarray] = {}
    band_2d:     dict[str, np.ndarray] = {}   # kept for GLCM
    pixel_grid_transform = None
    pixel_grid_crs = None

    with ThreadPoolExecutor(max_workers=max(1, len(band_hrefs))) as pool:
        futures = {
            pool.submit(_read_band_pixels, href, geometry, name, target_shape): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            name, arr, valid, t, crs = future.result()
            if arr is not None:
                band_arrays[name] = arr
                band_masks[name]  = valid
                band_2d[name]     = arr
                if pixel_grid_transform is None:
                    pixel_grid_transform = t
                    pixel_grid_crs = crs

    if not band_arrays:
        return None, [], None

    # ── 4. Combined validity mask ────────────────────────────────────────────
    combined_mask: np.ndarray = np.ones(
        next(iter(band_masks.values())).shape, dtype=bool
    )
    for m in band_masks.values():
        combined_mask &= m

    n_pixels = int(combined_mask.sum())
    if n_pixels == 0:
        return None, [], None

    # ── 5. Build per-pixel feature vectors ──────────────────────────────────
    flat_pixels: dict[str, np.ndarray] = {
        b: arr[combined_mask].astype(np.float32)
        for b, arr in band_arrays.items()
    }

    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    # Raw bands (in project band order)
    for band in available_bands:
        if band in flat_pixels:
            feature_cols.append(flat_pixels[band])
            feature_names.append(band)

    # Spectral indices from ASI catalogue
    for idx_name in enabled_indices:
        key = idx_name.upper()
        if key not in SPECTRAL_INDEX_CATALOGUE:
            continue
        result = compute_index(key, flat_pixels)
        if result is not None:
            feature_cols.append(result)
            feature_names.append(key)
        else:
            logger.debug("Skipping index %s — missing required bands", key)

    if not feature_cols:
        return None, feature_names or [], None

    X = np.column_stack(feature_cols)  # (n_pixels, n_raw_features)

    # ── 6. GLCM texture features (per-polygon, broadcast to all pixels) ─────
    glcm_enabled = (
        glcm_config is not None
        and glcm_config.get("enabled", False)
        and n_pixels > 0
    )
    if glcm_enabled:
        glcm_bands = glcm_config.get("bands", [])
        window_size = int(glcm_config.get("window_size", 5))
        statistics  = glcm_config.get(
            "statistics",
            ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"],
        )

        # Sub-sample mask to cap GLCM compute time
        compute_mask = combined_mask
        if n_pixels > GLCM_MAX_PIXELS:
            rows, cols = np.where(combined_mask)
            idx_choice = np.random.choice(len(rows), GLCM_MAX_PIXELS, replace=False)
            compute_mask = np.zeros_like(combined_mask)
            compute_mask[rows[idx_choice], cols[idx_choice]] = True
            logger.debug(
                "GLCM: capping pixel count %d → %d for polygon texture",
                n_pixels, GLCM_MAX_PIXELS,
            )

        glcm_scalar_names: list[str] = []
        glcm_scalar_vals:  list[float] = []

        for band in glcm_bands:
            if band not in band_2d:
                continue
            stats = compute_glcm_features(
                band_2d[band], compute_mask, window_size, statistics
            )
            for stat, val in stats.items():
                glcm_scalar_names.append(f"glcm_{band}_{stat}")
                glcm_scalar_vals.append(val)

        if glcm_scalar_names:
            glcm_row = np.array(glcm_scalar_vals, dtype=np.float32)
            glcm_cols = np.tile(glcm_row, (n_pixels, 1))  # (n_pixels, n_glcm)
            X = np.hstack([X, glcm_cols])
            feature_names.extend(glcm_scalar_names)

    # ── 7. Compute WGS84 pixel centroids ────────────────────────────────────
    coords_wgs84: np.ndarray | None = None
    if pixel_grid_transform is not None and pixel_grid_crs is not None:
        try:
            from rasterio.transform import xy as rio_xy
            from pyproj import Transformer as ProjTransformer

            rows_idx, cols_idx = np.where(combined_mask)
            xs_native, ys_native = rio_xy(pixel_grid_transform, rows_idx, cols_idx)
            to_wgs84 = ProjTransformer.from_crs(
                pixel_grid_crs, "epsg:4326", always_xy=True
            )
            lons, lats = to_wgs84.transform(xs_native, ys_native)
            coords_wgs84 = np.column_stack(
                [np.asarray(lons, dtype=np.float64),
                 np.asarray(lats, dtype=np.float64)]
            )
        except Exception as exc:
            logger.warning("Could not compute pixel centroids: %s", exc)

    return X, feature_names, coords_wgs84
