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
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Read a rectangular bbox window from one COG asset.

    Returns (band_name, float32 2-D array (H, W), uint8 mask (H, W))
    or (band_name, None, None) on failure.
    """
    from rio_tiler.io import COGReader

    try:
        with COGReader(href) as cog:
            img = cog.part(bbox, max_size=max_size)
        arr = img.data[0].astype(np.float32)   # (H, W)
        mask = img.mask                         # (H, W) uint8: 255=valid 0=nodata
        return band_name, arr, mask
    except Exception as exc:
        logger.warning("Part read failed for band %s: %s", band_name, exc)
        return band_name, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_bbox_features(
    bbox: list[float],
    signed_assets: dict,
    available_bands: list[str],
    enabled_indices: list[str],
    max_size: int = MAX_SIZE,
) -> tuple[np.ndarray | None, int, int, np.ndarray | None, list[str]]:
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

    with ThreadPoolExecutor(max_workers=len(band_hrefs)) as pool:
        futures = {
            pool.submit(_read_band_part, href, bbox, name, max_size): name
            for name, href in band_hrefs.items()
        }
        for future in as_completed(futures):
            band_name, arr, mask = future.result()
            if arr is not None:
                band_arrays[band_name] = arr
                band_masks[band_name] = mask
                shapes.append(arr.shape)

    if not band_arrays:
        return None, 0, 0, None, []

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
        return None, H, W, combined_mask.ravel(), feature_names

    X = np.column_stack(feature_cols)   # (H*W, n_features)
    return X, H, W, combined_mask.ravel(), feature_names


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
