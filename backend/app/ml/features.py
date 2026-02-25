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
) -> tuple[str, np.ndarray | None]:
    """
    Read pixels inside *geometry* from one COG asset.

    Uses HTTP range requests — only the bytes covering the polygon bbox
    are downloaded from Azure Blob Storage.

    Returns (band_name, 1D float32 pixel array) or (band_name, None) on failure.
    """
    from rio_tiler.io import COGReader

    # rio_tiler.feature() accepts a GeoJSON Feature dict
    geojson_feature = {"type": "Feature", "geometry": geometry, "properties": {}}
    try:
        with COGReader(href) as cog:
            img = cog.feature(geojson_feature, nodata=0)
        # img.mask: (H, W) uint8 — 255 = valid pixel, 0 = nodata / outside polygon
        valid = img.mask.ravel() > 0
        if not valid.any():
            return band_name, None
        pixels = img.data[0].ravel()[valid].astype(np.float32)
        return band_name, pixels
    except Exception as exc:
        logger.warning("Feature read failed for band %s: %s", band_name, exc)
        return band_name, None


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

    # Fetch all bands in parallel
    band_pixels: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_pixels, href, geometry, band_name): band_name
            for band_name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, pixels = future.result()
            if pixels is not None:
                band_pixels[band_name] = pixels

    if not band_pixels:
        return None, []

    # Snap all bands to the minimum pixel count (guards against edge-case mismatches)
    n_pixels = min(len(p) for p in band_pixels.values())
    if n_pixels == 0:
        return None, []

    # Build feature columns: raw bands first (in project order), then indices
    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []

    for logical in available_bands:
        if logical in band_pixels:
            feature_cols.append(band_pixels[logical][:n_pixels])
            feature_names.append(logical)

    for idx_name in enabled_indices:
        spec = SPECTRAL_INDICES.get(idx_name.upper())  # wizard stores lowercase, registry is uppercase
        if spec is None:
            continue
        if not spec["requires"].issubset(set(band_pixels.keys())):
            logger.debug("Skipping %s — missing required bands", idx_name)
            continue
        idx_values = spec["fn"]({k: v[:n_pixels] for k, v in band_pixels.items()})
        feature_cols.append(idx_values.astype(np.float32))
        feature_names.append(idx_name.upper())  # normalize to uppercase for display

    if not feature_cols:
        return None, feature_names

    X = np.column_stack(feature_cols)  # (n_pixels, n_features)
    return X, feature_names
