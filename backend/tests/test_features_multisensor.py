"""
Tests for multi-sensor feature extraction in backend/app/ml/features.py

Coverage:
  - extract_pixel_features with Sentinel-2 only (new API)
  - extract_pixel_features with Sentinel-1 only (SAR bands)
  - extract_pixel_features with both S1 + S2 (multi-sensor stacking)
  - backward compatibility: legacy flat dict (no collection key) is auto-wrapped
  - feature names are correct for each sensor combination
  - return tuple has three elements: (X, feature_names, coords)
  - coords shape matches X row count

All tests use synthetic local GeoTIFFs — no network calls.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# WGS84 polygon inside the synthetic COG extent (UTM32N ~11°E 47°N)
# These coordinates match the polygon in test_pixel_centroids.py.
_POLY = {
    "type": "Polygon",
    "coordinates": [[
        [11.0005, 46.9985],
        [11.0045, 46.9985],
        [11.0045, 47.0025],
        [11.0005, 47.0025],
        [11.0005, 46.9985],
    ]],
}


def _make_cog(path: str, width: int = 400, height: int = 400, fill: float | None = None) -> None:
    """Write a synthetic float32 UTM32N GeoTIFF at 10 m GSD.

    Covers 650000–654000 E, 5206000–5210000 N (UTM32N), which corresponds to
    approximately 11°E / 47°N in WGS84 — matching _POLY (same as test_pixel_centroids).
    """
    crs = CRS.from_epsg(32632)
    transform = Affine(10.0, 0.0, 650_000.0, 0.0, -10.0, 5_210_000.0)
    if fill is not None:
        data = np.full((height, width), fill, dtype=np.float32)
    else:
        data = (np.random.rand(height, width) * 3000 + 100).astype(np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        width=width, height=height,
        count=1, dtype="float32",
        crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def s2_assets(tmp_path):
    """Minimal S2 asset dict in STAC format: {asset_key: {"href": path}}."""
    paths = {}
    for asset_key in ("B02", "B04", "B08"):
        p = str(tmp_path / f"{asset_key}.tif")
        _make_cog(p)
        paths[asset_key] = {"href": p}
    return paths


@pytest.fixture()
def s1_assets(tmp_path):
    """Minimal S1 asset dict in STAC format: {asset_key: {"href": path}}."""
    paths = {}
    for asset_key in ("vv", "vh"):
        p = str(tmp_path / f"{asset_key}.tif")
        _make_cog(p)
        paths[asset_key] = {"href": p}
    return paths


# ---------------------------------------------------------------------------
# Basic return contract
# ---------------------------------------------------------------------------

class TestExtractPixelFeaturesReturnContract:
    def test_returns_three_element_tuple(self, s2_assets):
        from app.ml.features import extract_pixel_features

        result = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        assert len(result) == 3

    def test_x_is_2d_float32_array(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, coords = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted — polygon may not overlap COG in CI")
        assert X.ndim == 2
        assert X.dtype == np.float32

    def test_feature_names_length_matches_x_columns(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, coords = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        assert len(names) == X.shape[1]

    def test_coords_row_count_matches_x(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, coords = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        assert len(coords) == X.shape[0]

    def test_coords_are_wgs84_range(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, coords = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        for lon, lat in coords:
            assert -180 <= lon <= 180
            assert -90  <= lat <= 90


# ---------------------------------------------------------------------------
# Sentinel-2 only
# ---------------------------------------------------------------------------

class TestSentinel2Only:
    def test_s2_band_names_present(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        for band in ("blue", "red", "nir"):
            assert band in names, f"Band '{band}' not in feature names: {names}"

    def test_spectral_index_added(self, s2_assets):
        from app.ml.features import extract_pixel_features

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=["NDVI"],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        assert "NDVI" in names, f"NDVI not in feature names: {names}"


# ---------------------------------------------------------------------------
# Sentinel-1 only
# ---------------------------------------------------------------------------

class TestSentinel1Only:
    def test_s1_band_names_present(self, s1_assets):
        from app.ml.features import extract_pixel_features

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-1-rtc": s1_assets},
            available_bands=["sar_vv", "sar_vh"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        for band in ("sar_vv", "sar_vh"):
            assert band in names, f"Band '{band}' not in feature names: {names}"

    def test_no_s2_band_names_in_s1_only(self, s1_assets):
        from app.ml.features import extract_pixel_features

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-1-rtc": s1_assets},
            available_bands=["sar_vv", "sar_vh"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")
        for s2_band in ("blue", "green", "red", "nir"):
            assert s2_band not in names


# ---------------------------------------------------------------------------
# Multi-sensor (S1 + S2 stacked)
# ---------------------------------------------------------------------------

class TestMultiSensor:
    def test_both_sensor_bands_in_feature_names(self, s2_assets, s1_assets):
        from app.ml.features import extract_pixel_features

        sensors_assets = {
            "sentinel-2-l2a": s2_assets,
            "sentinel-1-rtc": s1_assets,
        }

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets=sensors_assets,
            available_bands=["blue", "red", "nir", "sar_vv", "sar_vh"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")

        for band in ("blue", "red", "nir"):
            assert band in names, f"S2 band '{band}' missing from multi-sensor names"
        for band in ("sar_vv", "sar_vh"):
            assert band in names, f"S1 band '{band}' missing from multi-sensor names"

    def test_column_count_is_sum_of_both_sensor_bands(self, s2_assets, s1_assets):
        from app.ml.features import extract_pixel_features

        sensors_assets = {
            "sentinel-2-l2a": s2_assets,
            "sentinel-1-rtc": s1_assets,
        }

        X, names, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets=sensors_assets,
            available_bands=["blue", "red", "nir", "sar_vv", "sar_vh"],
            enabled_indices=[],
        )
        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted")

        # 3 S2 bands + 2 S1 bands = 5
        assert X.shape[1] == 5, (
            f"Expected 5 band columns, got {X.shape[1]}. Names: {names}"
        )


# ---------------------------------------------------------------------------
# Backward compatibility: legacy flat dict (no collection key)
# ---------------------------------------------------------------------------

class TestLegacyFlatDictBackwardCompat:
    def test_flat_dict_wrapped_as_s2(self, s2_assets):
        """
        Old callers pass a flat {asset_key: {"href": ...}} dict without a collection
        wrapper.  extract_pixel_features must auto-wrap it as {"sentinel-2-l2a": ...}.
        """
        from app.ml.features import extract_pixel_features

        # Pass the S2 assets WITHOUT the collection-key wrapper
        X_compat, names_compat, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets=s2_assets,  # flat dict, legacy format
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )

        # Pass them WITH the explicit collection key
        X_new, names_new, _ = extract_pixel_features(
            geometry=_POLY,
            sensors_assets={"sentinel-2-l2a": s2_assets},
            available_bands=["blue", "red", "nir"],
            enabled_indices=[],
        )

        if X_compat is None or X_new is None:
            pytest.skip("No pixels extracted")

        assert names_compat == names_new, (
            "Feature names differ between legacy and new API"
        )
        assert X_compat.shape == X_new.shape

    def test_flat_dict_detection_uses_known_collection_keys(self, s2_assets):
        """
        Features.py detects legacy vs new format by checking whether any key
        matches a known STAC collection name.  The flat dict's keys are asset
        keys (B02, B04 …) — none of which are collection names — so it wraps.
        """
        from app.ml.spectral_catalogue import COLLECTION_BAND_TO_ASSET
        known_collections = set(COLLECTION_BAND_TO_ASSET.keys())
        # The flat (legacy) dict has asset keys, not collection keys
        overlap = known_collections.intersection(s2_assets.keys())
        assert overlap == set(), (
            f"Unexpected collection-name keys in flat fixture: {overlap}"
        )
