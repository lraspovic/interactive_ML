"""
Feature extraction from COG assets for training polygon geometries.

For each polygon:
  - fetch pixel values for each configured band via COGReader.feature()
  - compute enabled spectral indices from those raw values
  - return (n_valid_pixels, n_features) float32 array + ordered feature_names list

COG reads use HTTP range requests (only bytes inside the polygon bbox are fetched),
so no full imagery download is required.  Band reads for a single polygon are
parallelised with a ThreadPoolExecutor — same pattern used by the tile endpoint.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Target ground sample distance used for all feature reads.
# Every band (10 m or 20 m native) is resampled to this resolution so that
# pixel indices are spatially aligned across all bands.
TARGET_GSD = 10.0  # metres

# Logical band name (from wizard Step 3) → Sentinel-2 L2A STAC asset key.
# Only Sentinel-2 L2A is supported for now; extend this map for other collections.
S2_BAND_TO_ASSET: dict[str, str] = {
    "blue":      "B02",
    "green":     "B03",
    "red":       "B04",
    "nir":       "B08",
    "red_edge":  "B8A",
    "swir1":     "B11",
    "swir2":     "B12",
}

_EPS = 1e-10

# Spectral index registry.
# "requires": set of logical band names that must be present.
# "fn": callable that receives {band_name: 1D float32 array} → 1D float32 array.
SPECTRAL_INDICES: dict[str, dict[str, Any]] = {
    "NDVI": {
        "requires": {"nir", "red"},
        "fn": lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS),
    },
    "NDWI": {
        "requires": {"green", "nir"},
        "fn": lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS),
    },
    "NDBI": {
        "requires": {"swir1", "nir"},
        "fn": lambda b: (b["swir1"] - b["nir"]) / (b["swir1"] + b["nir"] + _EPS),
    },
    "BSI": {
        "requires": {"swir1", "red", "nir", "blue"},
        "fn": lambda b: (
            (b["swir1"] + b["red"]) - (b["nir"] + b["blue"])
        ) / (
            (b["swir1"] + b["red"]) + (b["nir"] + b["blue"]) + _EPS
        ),
    },
    "EVI": {
        "requires": {"nir", "red", "blue"},
        "fn": lambda b: 2.5 * (b["nir"] - b["red"]) / (
            b["nir"] + 6 * b["red"] - 7.5 * b["blue"] + 1 + _EPS
        ),
    },
}


def _read_band_pixels(
    href: str,
    geometry: dict,
    band_name: str,
    target_shape: tuple[int, int] | None = None,
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Read pixels inside *geometry* from one COG asset.

    All bands are resampled to *target_shape* (rows, cols) so that pixel
    indices are spatially aligned regardless of native band resolution
    (e.g. 10 m vs 20 m Sentinel-2 bands).

    Uses HTTP range requests — only the bytes covering the polygon bbox
    are downloaded from Azure Blob Storage.

    Returns (band_name, 2D float32 array, 2D bool valid_mask)
    or (band_name, None, None) on failure.
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
                return band_name, None, None

            # Determine output shape: use caller-supplied target or derive from
            # the native extent at TARGET_GSD so every band lands on the same grid.
            if target_shape is None:
                extent_x = native_bounds[2] - native_bounds[0]
                extent_y = native_bounds[3] - native_bounds[1]
                tgt_w = max(1, round(extent_x / TARGET_GSD))
                tgt_h = max(1, round(extent_y / TARGET_GSD))
                target_shape = (tgt_h, tgt_w)

            out_h, out_w = target_shape

            # Read at target_shape — rasterio bilinear-resamples 20 m → 10 m
            data = ds.read(
                1,
                window=win,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
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
                return band_name, None, None

        return band_name, data, valid
    except Exception as exc:
        logger.warning("Feature read failed for band %s: %s", band_name, exc)
        return band_name, None, None


def extract_pixel_features(
    geometry: dict,
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str],
) -> tuple[np.ndarray | None, list[str]]:
    """
    Extract a (n_valid_pixels, n_features) float32 feature matrix for one polygon.

    Parameters
    ----------
    geometry:       GeoJSON Polygon geometry dict (EPSG:4326)
    signed_assets:  signed asset href dict from _get_signed_assets()
    available_bands: ordered list of logical band names configured for the project
    enabled_indices: list of spectral index names to compute (e.g. ["NDVI"])

    Returns
    -------
    (X, feature_names) — X is None if no valid pixels were found inside the polygon.
    """
    # Resolve logical band names to signed asset hrefs
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
        return None, []

    # Pre-compute a shared target_shape at TARGET_GSD from the first available
    # band's native UTM extent.  All bands (10 m and 20 m native) will be read
    # at this same grid so pixel indices are spatially consistent.
    target_shape: tuple[int, int] | None = None
    try:
        import rasterio
        import rasterio.features as rio_feat
        from rasterio import warp
        first_href = next(iter(band_hrefs.values()))
        aoi_bounds = rio_feat.bounds(geometry)
        with rasterio.open(first_href) as ds:
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        extent_x = native_bounds[2] - native_bounds[0]
        extent_y = native_bounds[3] - native_bounds[1]
        tgt_w = max(1, round(extent_x / TARGET_GSD))
        tgt_h = max(1, round(extent_y / TARGET_GSD))
        target_shape = (tgt_h, tgt_w)
    except Exception as exc:
        logger.warning("Could not pre-compute target_shape, falling back to native: %s", exc)

    # Fetch all bands in parallel, resampled to the shared target_shape
    band_arrays: dict[str, np.ndarray] = {}   # 2D float32
    band_masks:  dict[str, np.ndarray] = {}   # 2D bool (valid pixels)
    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_pixels, href, geometry, band_name, target_shape): band_name
            for band_name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, valid = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                band_masks[band_name]  = valid

    if not band_arrays:
        return None, []

    # Intersect all valid masks so every feature row refers to the same ground pixel.
    # A pixel is included only if it is valid (inside polygon, not nodata) in ALL bands.
    combined_mask: np.ndarray = np.ones(next(iter(band_masks.values())).shape, dtype=bool)
    for m in band_masks.values():
        combined_mask &= m

    n_pixels = int(combined_mask.sum())
    if n_pixels == 0:
        return None, []

    # Build feature columns: raw bands first (in project order), then indices
    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    # 1D pixel vectors extracted at the shared mask
    flat_pixels: dict[str, np.ndarray] = {
        b: arr[combined_mask].astype(np.float32)
        for b, arr in band_arrays.items()
    }

    for logical in available_bands:
        if logical in flat_pixels:
            feature_cols.append(flat_pixels[logical])
            feature_names.append(logical)

    for idx_name in enabled_indices:
        spec = SPECTRAL_INDICES.get(idx_name.upper())  # wizard stores lowercase, registry is uppercase
        if spec is None:
            continue
        if not spec["requires"].issubset(set(flat_pixels.keys())):
            logger.debug("Skipping %s — missing required bands", idx_name)
            continue
        idx_values = spec["fn"](flat_pixels)
        feature_cols.append(idx_values.astype(np.float32))
        feature_names.append(idx_name.upper())  # normalize to uppercase for display

    if not feature_cols:
        return None, feature_names

    X = np.column_stack(feature_cols)  # (n_pixels, n_features)
    return X, feature_names
