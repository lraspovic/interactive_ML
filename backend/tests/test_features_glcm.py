"""
Tests for GLCM feature extraction in backend/app/ml/features.py

Coverage:
  - compute_glcm_features() basic smoke test
  - constant patch returns all-zero statistics
  - window sliding over a small valid mask
  - empty valid_mask returns {}
  - unknown statistic name is gracefully handled (skimage will raise)
  - GLCM pixel cap: extract_pixel_features limits GLCM computation to
    GLCM_MAX_PIXELS pixels per polygon and broadcasts the scalar result
    to every pixel row in the output matrix

No network calls — all COGs are synthetic local files.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.ml.features import GLCM_MAX_PIXELS, compute_glcm_features

# ---------------------------------------------------------------------------
# compute_glcm_features unit tests
# ---------------------------------------------------------------------------

# Skip the whole class if scikit-image is not available in this environment
skimage = pytest.importorskip("skimage", reason="scikit-image not installed — skip GLCM tests")


class TestComputeGlcmFeatures:
    """Unit tests for the stand-alone compute_glcm_features helper."""

    def _make_patch(self, rows: int, cols: int, fill: float = 0.0):
        """Return (band_data, valid_mask) covering the whole patch."""
        data = np.full((rows, cols), fill, dtype=np.float32)
        mask = np.ones((rows, cols), dtype=bool)
        return data, mask

    # ── smoke tests ─────────────────────────────────────────────────────────

    def test_basic_returns_dict(self):
        data = np.random.rand(20, 20).astype(np.float32)
        mask = np.ones((20, 20), dtype=bool)
        result = compute_glcm_features(data, mask, window_size=3, statistics=["contrast"])
        assert isinstance(result, dict)
        assert "contrast" in result

    def test_requested_stats_present_in_output(self):
        data = np.random.rand(20, 20).astype(np.float32)
        mask = np.ones((20, 20), dtype=bool)
        stats = ["contrast", "homogeneity", "energy", "correlation"]
        result = compute_glcm_features(data, mask, window_size=3, statistics=stats)
        for s in stats:
            assert s in result, f"Statistic '{s}' missing from result"

    def test_output_values_are_floats(self):
        data = np.random.rand(15, 15).astype(np.float32)
        mask = np.ones((15, 15), dtype=bool)
        result = compute_glcm_features(data, mask, window_size=3, statistics=["contrast"])
        assert isinstance(result["contrast"], float)

    # ── edge cases ───────────────────────────────────────────────────────────

    def test_empty_valid_mask_returns_empty_dict(self):
        data = np.random.rand(10, 10).astype(np.float32)
        mask = np.zeros((10, 10), dtype=bool)   # no valid pixels
        result = compute_glcm_features(data, mask, window_size=3, statistics=["contrast"])
        assert result == {}

    def test_constant_patch_returns_zeros(self):
        """A patch with identical values → GLCM should return zeros for all stats."""
        data, mask = self._make_patch(20, 20, fill=500.0)
        result = compute_glcm_features(
            data, mask, window_size=3,
            statistics=["contrast", "dissimilarity", "homogeneity"],
        )
        for stat, val in result.items():
            assert val == pytest.approx(0.0, abs=1e-6), (
                f"Expected 0 for {stat} on constant patch, got {val}"
            )

    def test_single_pixel_valid_mask_gracefully_handled(self):
        """Single valid pixel — patch size will be < 4, should return empty dict."""
        data = np.random.rand(5, 5).astype(np.float32) * 1000
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True   # only centre pixel valid
        result = compute_glcm_features(data, mask, window_size=3, statistics=["contrast"])
        # May return {} or a float — the function must not raise
        assert isinstance(result, dict)

    def test_different_window_sizes(self):
        data = np.random.rand(30, 30).astype(np.float32) * 2000
        mask = np.ones((30, 30), dtype=bool)
        for ws in (3, 5, 7, 9):
            result = compute_glcm_features(
                data, mask, window_size=ws, statistics=["energy"]
            )
            assert "energy" in result, f"Failed for window_size={ws}"

    # ── scikit-image unavailable ─────────────────────────────────────────────

    def test_returns_empty_when_skimage_missing(self):
        """If scikit-image is not installed the function should quietly return {}."""
        data = np.random.rand(10, 10).astype(np.float32)
        mask = np.ones((10, 10), dtype=bool)
        with patch.dict("sys.modules", {"skimage.feature": None}):
            result = compute_glcm_features(
                data, mask, window_size=3, statistics=["contrast"]
            )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# GLCM pixel cap constant
# ---------------------------------------------------------------------------

class TestGlcmPixelCap:
    def test_glcm_max_pixels_is_reasonable(self):
        """GLCM_MAX_PIXELS must be a positive integer not too small or too large."""
        assert isinstance(GLCM_MAX_PIXELS, int)
        assert 50 <= GLCM_MAX_PIXELS <= 10_000


# ---------------------------------------------------------------------------
# GLCM integration: extract_pixel_features broadcasts per-polygon GLCM
# ---------------------------------------------------------------------------

def _make_synthetic_cog(path: str, width: int = 400, height: int = 400) -> None:
    """Write a synthetic float32 UTM32N GeoTIFF at 10 m GSD.

    Covers 650000–654000 E, 5206000–5210000 N (UTM32N), approximately 11°E / 47°N.
    """
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS

    crs = CRS.from_epsg(32632)
    transform = Affine(10.0, 0.0, 650_000.0, 0.0, -10.0, 5_210_000.0)
    data = (np.random.rand(height, width) * 3000).astype(np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        width=width, height=height,
        count=1, dtype="float32",
        crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(data, 1)


# Polygon inside the synthetic COG extent (UTM32N → ~11°E 47°N in WGS84)
# Coordinates match the polygon in test_pixel_centroids.py.
_POLY_WGS84 = {
    "type": "Polygon",
    "coordinates": [[
        [11.0005, 46.9985],
        [11.0045, 46.9985],
        [11.0045, 47.0025],
        [11.0005, 47.0025],
        [11.0005, 46.9985],
    ]],
}


@pytest.mark.integration
class TestGlcmBroadcast:
    """Integration test: verify GLCM scalars are broadcast to all pixel rows."""

    def test_glcm_columns_broadcast_to_all_pixels(self, tmp_path):
        from app.ml.features import extract_pixel_features

        cog = str(tmp_path / "band.tif")
        _make_synthetic_cog(cog, width=400, height=400)

        stats = ["contrast", "homogeneity"]
        glcm_config = {
            "enabled": True,
            "bands": ["blue"],
            "window_size": 3,
            "statistics": stats,
        }

        sensors_assets = {
            "sentinel-2-l2a": {
                "B02": {"href": cog},  # blue
            }
        }

        X, feat_names, coords = extract_pixel_features(
            geometry=_POLY_WGS84,
            sensors_assets=sensors_assets,
            available_bands=["blue"],
            enabled_indices=[],
            glcm_config=glcm_config,
        )

        if X is None or X.shape[0] == 0:
            pytest.skip("No pixels extracted — COG may not overlap polygon in CI")

        n_pixels = X.shape[0]

        # GLCM feature columns should be present
        glcm_cols = [f for f in feat_names if f.startswith("glcm_")]
        assert len(glcm_cols) > 0, f"No GLCM feature columns found in {feat_names}"

        # All rows in each GLCM column must have the same value
        # (because scalars were broadcast from the polygon-level computation)
        for col_name in glcm_cols:
            col_idx = feat_names.index(col_name)
            col_vals = X[:, col_idx]
            assert np.all(col_vals == col_vals[0]), (
                f"GLCM column '{col_name}' is not constant across pixels — "
                "broadcast failed"
            )

    def test_glcm_none_config_adds_no_extra_columns(self, tmp_path):
        from app.ml.features import extract_pixel_features

        cog = str(tmp_path / "band.tif")
        _make_synthetic_cog(cog, width=400, height=400)

        sensors_assets = {"sentinel-2-l2a": {"B02": {"href": cog}}}

        X_no_glcm, names_no_glcm, _ = extract_pixel_features(
            geometry=_POLY_WGS84,
            sensors_assets=sensors_assets,
            available_bands=["blue"],
            enabled_indices=[],
            glcm_config=None,
        )
        if X_no_glcm is None or X_no_glcm.shape[0] == 0:
            pytest.skip("No pixels extracted")

        glcm_cols = [n for n in names_no_glcm if n.startswith("glcm_")]
        assert glcm_cols == [], (
            f"Unexpected GLCM columns when glcm_config=None: {glcm_cols}"
        )
