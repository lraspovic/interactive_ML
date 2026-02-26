"""
Bbox-based feature extraction and PNG rendering for the prediction pipeline.

extract_bbox_features: reads each configured band for a rectangular bbox using
COGReader.part() (HTTP range requests, no full download), then stacks bands +
spectral indices into a flat (H*W, n_features) matrix — same feature order and
computation as extract_pixel_features() so train/predict stay in sync.

render_prediction_png: colours each pixel by its predicted class_id.
render_uncertainty_png: colours each pixel by prediction entropy (blue→red).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

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
    max_size: int,
) -> tuple[str, np.ndarray | None, np.ndarray | None, list[float] | None]:
    """
    Read a rectangular bbox window from one COG asset.

    Returns (band_name, float32 2-D array (H, W), uint8 mask (H, W),
             actual_bounds [minx, miny, maxx, maxy])
    or (band_name, None, None, None) on failure.

    The actual_bounds may differ slightly from the requested bbox because
    rio_tiler snaps reads to the native pixel grid.
    """
    import rasterio
    import rasterio.crs
    import rasterio.enums
    from affine import Affine
    from rasterio import windows, warp

    # Reproject to EPSG:3857 (Web Mercator) so the output pixel grid matches
    # Leaflet's tile layer CRS.  Leaflet's ImageOverlay stretches the PNG
    # linearly between the WGS84 corner bounds in Mercator space — if the
    # pixels were on a UTM grid the features away from centre would misalign.
    _DST_CRS = rasterio.crs.CRS.from_epsg(3857)

    try:
        with rasterio.open(href) as ds:
            # 1. Find the UTM window covering the requested WGS84 bbox
            minlon, minlat, maxlon, maxlat = bbox
            native_bounds = warp.transform_bounds(
                "epsg:4326", ds.crs, minlon, minlat, maxlon, maxlat
            )
            win = windows.from_bounds(*native_bounds, transform=ds.transform)
            win = win.intersection(windows.Window(0, 0, ds.width, ds.height))
            if win.width <= 0 or win.height <= 0:
                return band_name, None, None, None

            actual_native = windows.bounds(win, ds.transform)
            win_transform = windows.transform(win, ds.transform)
            nodata_val = float(ds.nodata) if ds.nodata is not None else 0.0
            src_crs = ds.crs

            # 2. Read at a consistent 10 m GSD regardless of native band resolution.
            # Sentinel-2 bands are 10 m (B02/03/04/08) or 20 m (B8A/B11/B12).
            # Using win.height/width for 20 m bands gives half the pixels of 10 m
            # bands, so all 20 m bands end up at 20 m after reprojection.
            # Instead, compute output dimensions from the UTM extent ÷ 10 m,
            # then cap at max_size so very large viewports don't blow memory.
            TARGET_GSD = 10.0  # metres
            extent_x = actual_native[2] - actual_native[0]
            extent_y = actual_native[3] - actual_native[1]
            tgt_w = max(1, round(extent_x / TARGET_GSD))
            tgt_h = max(1, round(extent_y / TARGET_GSD))
            # Apply max_size cap (keep aspect ratio)
            gsd_scale = max_size / max(tgt_w, tgt_h)
            src_w = max(1, round(tgt_w * gsd_scale)) if gsd_scale < 1.0 else tgt_w
            src_h = max(1, round(tgt_h * gsd_scale)) if gsd_scale < 1.0 else tgt_h

            src_arr = ds.read(
                1, window=win,
                out_shape=(src_h, src_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)

        # 3. Compute EPSG:3857 output transform; cap at max_size but never upscale
        dst_transform, dst_w, dst_h = warp.calculate_default_transform(
            src_crs, _DST_CRS, src_w, src_h, *actual_native
        )
        scale = min(1.0, max_size / max(dst_w, dst_h))
        out_w = max(1, round(dst_w * scale))
        out_h = max(1, round(dst_h * scale))
        # Scale the transform to the actual output resolution
        out_transform = dst_transform * Affine.scale(dst_w / out_w, dst_h / out_h)

        # 4. Reproject native UTM pixels → EPSG:3857
        dst_arr = np.full((out_h, out_w), nodata_val, dtype=np.float32)
        warp.reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=win_transform,
            src_crs=src_crs,
            dst_transform=out_transform,
            dst_crs=_DST_CRS,
            src_nodata=nodata_val,
            dst_nodata=nodata_val,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # 5. Nodata mask + convert Mercator bounds to WGS84 for Leaflet
        mask = np.where(dst_arr != nodata_val, np.uint8(255), np.uint8(0))
        out_left   = out_transform.c
        out_top    = out_transform.f
        out_right  = out_left + out_transform.a * out_w
        out_bottom = out_top  + out_transform.e * out_h  # e is negative
        west, south, east, north = warp.transform_bounds(
            _DST_CRS, "epsg:4326", out_left, out_bottom, out_right, out_top
        )
        actual_bounds = [west, south, east, north]

        return band_name, dst_arr, mask, actual_bounds
    except Exception as exc:
        logger.warning("Part read failed for band %s: %s", band_name, exc)
        return band_name, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bbox_features(
    bbox: list[float],
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str],
    max_size: int = MAX_SIZE,
) -> tuple[np.ndarray | None, int, int, np.ndarray | None, list[str], list[float] | None]:
    """
    Extract a (H*W, n_features) feature matrix for a rectangular bounding box.

    Uses the same band-order and spectral-index logic as extract_pixel_features()
    so the feature vector fed to model.predict() is identical to training.

    Returns
    -------
    X            : (H*W, n_features) float32, or None on failure
    H, W         : pixel dimensions of the raster window
    valid_mask   : (H*W,) bool — True where all bands had valid data
    feature_names: list of feature name strings
    actual_bounds: actual raster extent [minx, miny, maxx, maxy] (may differ
                   slightly from the requested bbox due to pixel snapping)
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
        return None, 0, 0, None, []

    # Read all bands in parallel
    band_arrays: dict[str, np.ndarray] = {}
    band_masks: dict[str, np.ndarray] = {}
    shapes: list[tuple[int, int]] = []
    actual_bounds: list[float] | None = None  # taken from the first successful read

    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_part, href, bbox, name, max_size): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, mask, bounds = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                band_masks[band_name] = mask
                shapes.append(arr.shape)
                if actual_bounds is None and bounds is not None:
                    actual_bounds = bounds

    if not band_arrays:
        return None, 0, 0, None, [], None

    # Snap all bands to the common (minimum) size
    H = min(s[0] for s in shapes)
    W = min(s[1] for s in shapes)
    for bn in list(band_arrays.keys()):
        band_arrays[bn] = band_arrays[bn][:H, :W]
        band_masks[bn] = band_masks[bn][:H, :W]

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
        return None, H, W, combined_mask.ravel(), feature_names, actual_bounds

    X = np.column_stack(feature_cols)   # (H*W, n_features)
    return X, H, W, combined_mask.ravel(), feature_names, actual_bounds


def read_raw_bands_tiff(
    bbox: list[float],
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str] | None = None,
    max_size: int = MAX_SIZE,
) -> bytes | None:
    """
    Read raw band pixel data for a bbox and return a georeferenced multi-band
    GeoTIFF as bytes.  Raw bands come first, then any computable spectral
    indices (NDVI, NDWI, …) as additional bands.  Band names are stored as
    raster tags.  CRS: EPSG:3857 (Web Mercator, matching the prediction PNG
    grid — open in QGIS/ArcGIS to inspect the exact values fed to the model).
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

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
    actual_bounds_wgs84: list[float] | None = None

    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_part, href, bbox, name, max_size): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, _mask, bounds_wgs84 = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                if actual_bounds_wgs84 is None and bounds_wgs84 is not None:
                    actual_bounds_wgs84 = bounds_wgs84

    if not band_arrays:
        return None

    actual_bounds_wgs84 = actual_bounds_wgs84 or bbox
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

    # Data is on an EPSG:3857 grid (reprojected by _read_band_part).
    # Convert WGS84 actual bounds back to Mercator for the TIFF geotransform.
    from rasterio import warp as _warp
    west, south, east, north = actual_bounds_wgs84
    left_m, bottom_m, right_m, top_m = _warp.transform_bounds(
        "epsg:4326", "epsg:3857", west, south, east, north
    )
    transform = from_bounds(left_m, bottom_m, right_m, top_m, W, H)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=H,
            width=W,
            count=len(arrays),
            dtype="float32",
            crs="EPSG:3857",
            transform=transform,
            nodata=0.0,
        ) as ds:
            for i, (arr, name) in enumerate(zip(arrays, band_order), start=1):
                ds.write(arr, i)
                ds.update_tags(i, name=name)
        return memfile.read()


def render_prediction_png(
    predictions: np.ndarray,         # (H*W,) int — predicted class_id per pixel
    class_colors: dict[int, str],     # {class_id: "#RRGGBB"}
    H: int,
    W: int,
    valid_mask: np.ndarray,           # (H*W,) bool
) -> bytes:
    """
    Render a fully-opaque RGBA PNG where each pixel is the color of its
    predicted class.  Pixels outside the valid_mask are transparent.
    """
    from PIL import Image

    rgba = np.zeros((H * W, 4), dtype=np.uint8)

    for class_id, hex_color in class_colors.items():
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        where = (predictions == class_id) & valid_mask
        rgba[where] = [r, g, b, 210]

    # Pixels with no valid data stay transparent

    img = Image.fromarray(rgba.reshape(H, W, 4), mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_uncertainty_png(
    probas: np.ndarray,       # (H*W, n_classes) float — predict_proba output
    H: int,
    W: int,
    valid_mask: np.ndarray,   # (H*W,) bool
) -> bytes:
    """
    Render a heatmap PNG based on per-pixel Shannon entropy.

    Colour scale:  cool blue (certain) → yellow (mid) → hot red (uncertain).
    Invalid pixels are transparent.
    """
    from PIL import Image

    p = np.clip(probas, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)   # (H*W,)
    max_entropy = np.log(max(probas.shape[1], 2))
    norm = np.clip(entropy / max_entropy, 0.0, 1.0)   # (H*W,) in [0, 1]

    # Three-stop colour ramp: blue → yellow → red
    low  = np.array([30,  130, 220], dtype=np.float32)   # blue
    mid  = np.array([250, 220,  30], dtype=np.float32)   # yellow
    high = np.array([220,  30,  30], dtype=np.float32)   # red

    t = norm[:, None]   # (H*W, 1) for broadcasting
    colors = np.where(
        t <= 0.5,
        low + (mid - low) * (t / 0.5),
        mid + (high - mid) * ((t - 0.5) / 0.5),
    ).astype(np.uint8)

    rgba = np.zeros((H * W, 4), dtype=np.uint8)
    rgba[:, :3] = colors
    rgba[:, 3] = np.where(valid_mask, 200, 0).astype(np.uint8)

    img = Image.fromarray(rgba.reshape(H, W, 4), mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
