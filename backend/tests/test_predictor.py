"""
Tests for backend/app/ml/predictor.py

All tests use synthetic local GeoTIFFs — no network calls, no DB calls.
Focus areas:
  1. _read_band_part        — stays in native UTM, no EPSG:3857 reprojection
  2. extract_bbox_features  — correct snapping, utm_transform/native_crs passthrough
  3. write_prediction_cog   — UTM CRS, exact transform, class ID pixels, nodata mask, COG structure
  4. write_uncertainty_cog  — UTM CRS, exact transform, entropy values, nodata mask
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import rasterio.shutil as rio_shutil
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio import warp


# ---------------------------------------------------------------------------
# Shared constants — a 1 km × 1 km patch in UTM zone 32N (Bavaria, ~10°E 47°N)
# ---------------------------------------------------------------------------
UTM32N   = CRS.from_epsg(32632)
UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP = 600_000.0, 5_199_000.0, 601_000.0, 5_200_000.0
UTM_W, UTM_H = 100, 100
UTM_TRANSFORM = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, UTM_W, UTM_H)

# WGS84 bbox tightly covering the synthetic UTM patch — derived once at import time
_WGS84_BOUNDS = warp.transform_bounds(UTM32N, "epsg:4326", UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP)
WGS84_BBOX = list(_WGS84_BOUNDS)  # [west, south, east, north]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cog(path: str, *, fill: float = 1000.0, nodata: float = 0.0) -> None:
    """Write a tiny valid COG in UTM32N to *path*."""
    tmp = path + ".src.tif"
    data = np.full((UTM_H, UTM_W), fill, dtype=np.float32)
    with rasterio.open(
        tmp, "w",
        driver="GTiff",
        height=UTM_H, width=UTM_W,
        count=1, dtype="float32",
        crs=UTM32N,
        transform=UTM_TRANSFORM,
        nodata=nodata,
        tiled=True, blockxsize=64, blockysize=64,
    ) as ds:
        ds.write(data, 1)
        ds.build_overviews([2, 4], Resampling.bilinear)
        ds.update_tags(ns="rio_overview", resampling="bilinear")

    with rasterio.open(tmp) as src:
        rio_shutil.copy(
            src, path,
            driver="GTiff",
            copy_src_overviews=True,
            tiled=True, blockxsize=64, blockysize=64,
            compress="deflate",
        )
    Path(tmp).unlink()


@pytest.fixture()
def cog_path(tmp_path) -> str:
    p = str(tmp_path / "band.tif")
    _make_cog(p)
    return p


@pytest.fixture()
def cog_path_20m(tmp_path) -> str:
    """Simulate a 20 m band: same spatial extent, half the pixel count (50×50)."""
    tmp = str(tmp_path / "band20.src.tif")
    out = str(tmp_path / "band20.tif")
    w, h = 50, 50
    data = np.full((h, w), 2000.0, dtype=np.float32)
    t = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, w, h)
    with rasterio.open(
        tmp, "w",
        driver="GTiff",
        height=h, width=w,
        count=1, dtype="float32",
        crs=UTM32N,
        transform=t,
        nodata=0.0,
        tiled=True, blockxsize=32, blockysize=32,
    ) as ds:
        ds.write(data, 1)
        ds.build_overviews([2], Resampling.bilinear)
        ds.update_tags(ns="rio_overview", resampling="bilinear")
    with rasterio.open(tmp) as src:
        rio_shutil.copy(src, out, driver="GTiff", copy_src_overviews=True,
                        tiled=True, blockxsize=32, blockysize=32, compress="deflate")
    Path(tmp).unlink()
    return out


# ---------------------------------------------------------------------------
# 1. _read_band_part
# ---------------------------------------------------------------------------

class TestReadBandPart:
    """_read_band_part must stay in native UTM — no EPSG:3857 reprojection."""

    def test_returns_five_tuple(self, cog_path):
        from app.ml.predictor import _read_band_part
        result = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        assert len(result) == 5, "Must return (band_name, arr, mask, actual_native, native_crs)"

    def test_native_crs_is_utm_not_mercator(self, cog_path):
        from app.ml.predictor import _read_band_part
        _, arr, _, _, native_crs = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        assert arr is not None, "Expected a valid array"
        assert native_crs is not None
        assert native_crs.to_epsg() != 3857, "CRS must not be reprojected to EPSG:3857"
        assert native_crs.to_epsg() == 32632, "CRS should be UTM 32N (EPSG:32632)"

    def test_actual_native_bounds_in_utm_metres(self, cog_path):
        """actual_native should be in UTM metres (x >> 180, not in WGS84 degrees)."""
        from app.ml.predictor import _read_band_part
        _, _, _, actual_native, _ = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        left, bottom, right, top = actual_native
        # UTM easting for central Europe is ~600,000 m; degrees would be ~10
        assert left > 1000, f"Expected UTM easting (metres), got {left}"
        assert right > left
        assert top > bottom

    def test_actual_native_matches_utm_patch(self, cog_path):
        """Returned bounds must be within a few metres of the synthetic patch bounds."""
        from app.ml.predictor import _read_band_part
        _, arr, _, actual_native, _ = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        left, bottom, right, top = actual_native
        tol = 15.0  # one pixel at 10 m GSD
        assert abs(left   - UTM_LEFT)   < tol, f"left mismatch: {left} vs {UTM_LEFT}"
        assert abs(bottom - UTM_BOTTOM) < tol, f"bottom mismatch: {bottom} vs {UTM_BOTTOM}"
        assert abs(right  - UTM_RIGHT)  < tol, f"right mismatch: {right} vs {UTM_RIGHT}"
        assert abs(top    - UTM_TOP)    < tol, f"top mismatch: {top} vs {UTM_TOP}"

    def test_output_array_is_float32(self, cog_path):
        from app.ml.predictor import _read_band_part
        _, arr, _, _, _ = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        assert arr.dtype == np.float32

    def test_mask_matches_valid_pixels(self, cog_path):
        from app.ml.predictor import _read_band_part
        _, arr, mask, _, _ = _read_band_part(cog_path, WGS84_BBOX, "red", 512)
        # fill=1000 so every pixel is valid (nodata=0)
        assert mask.shape == arr.shape
        assert np.all(mask == 255), "All pixels should be valid (fill != nodata)"

    def test_max_size_respected(self, cog_path):
        """Requesting max_size=64 must not return an array larger than 64 in any dim."""
        from app.ml.predictor import _read_band_part
        _, arr, _, _, _ = _read_band_part(cog_path, WGS84_BBOX, "red", 64)
        assert arr is not None
        assert max(arr.shape) <= 64

    def test_out_of_bounds_bbox_returns_none(self, cog_path):
        from app.ml.predictor import _read_band_part
        # Antarctica bbox — completely outside our UTM32N synthetic file
        far_away = [-70.0, -80.0, -60.0, -70.0]
        _, arr, mask, actual_native, native_crs = _read_band_part(cog_path, far_away, "red", 512)
        assert arr is None


# ---------------------------------------------------------------------------
# 2. extract_bbox_features
# ---------------------------------------------------------------------------

class TestExtractBboxFeatures:
    """extract_bbox_features must return a consistent UTM grid after snapping."""

    def _make_signed_assets(self, band_hrefs: dict[str, str]) -> dict:
        """Build a fake signed_assets dict from {logical_name: local_path}."""
        from app.ml.features import S2_BAND_TO_ASSET
        assets = {}
        for logical, path in band_hrefs.items():
            asset_key = S2_BAND_TO_ASSET[logical]
            assets[asset_key] = {"href": path}
        return assets

    def test_returns_eight_tuple(self, cog_path):
        from app.ml.predictor import extract_bbox_features
        signed_assets = self._make_signed_assets({"red": cog_path})
        result = extract_bbox_features(WGS84_BBOX, signed_assets, ["red"], [])
        assert len(result) == 8

    def test_utm_transform_is_affine(self, cog_path):
        from app.ml.predictor import extract_bbox_features
        signed_assets = self._make_signed_assets({"red": cog_path})
        _, H, W, _, _, _, utm_transform, _ = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )
        assert isinstance(utm_transform, Affine)
        # Pixel dimensions encoded in the transform must match H, W
        xres = abs(utm_transform.a)
        yres = abs(utm_transform.e)
        assert xres > 0
        assert yres > 0
        # The transform should cover (W * xres, H * yres) metres
        assert abs(W * xres - (UTM_RIGHT - UTM_LEFT)) < 2.0
        assert abs(H * yres - (UTM_TOP - UTM_BOTTOM)) < 2.0

    def test_native_crs_is_utm(self, cog_path):
        from app.ml.predictor import extract_bbox_features
        signed_assets = self._make_signed_assets({"red": cog_path})
        _, _, _, _, _, _, _, native_crs = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )
        assert native_crs.to_epsg() == 32632

    def test_actual_bounds_wgs84_are_in_degrees(self, cog_path):
        from app.ml.predictor import extract_bbox_features
        signed_assets = self._make_signed_assets({"red": cog_path})
        _, _, _, _, _, actual_bounds, _, _ = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )
        west, south, east, north = actual_bounds
        assert -180 <= west < east <= 180
        assert  -90 <= south < north <= 90

    def test_snapping_gives_exact_transform(self):
        """
        When parallel band reads return arrays with slightly different pixel
        dimensions (common when window snapping rounds differently per band),
        after snapping to min(H, W) the utm_transform must equal
        from_bounds(*actual_native, min_W, min_H) — not inherited from any
        individual band's read.
        """
        from unittest.mock import patch
        from app.ml.features import S2_BAND_TO_ASSET
        from app.ml.predictor import extract_bbox_features

        # Simulate real-world sub-pixel rounding: red=100×100, nir=100×98
        H_A, W_A = 100, 100
        H_B, W_B = 100, 98
        actual_native = (UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP)

        def mock_read(href, bbox, band_name, max_size):
            if band_name == "red":
                arr  = np.ones((H_A, W_A), dtype=np.float32)
                mask = np.full((H_A, W_A), 255, dtype=np.uint8)
                return band_name, arr, mask, actual_native, UTM32N
            else:  # nir
                arr  = np.ones((H_B, W_B), dtype=np.float32)
                mask = np.full((H_B, W_B), 255, dtype=np.uint8)
                return band_name, arr, mask, actual_native, UTM32N

        signed_assets = {
            S2_BAND_TO_ASSET["red"]: {"href": "fake_red.tif"},
            S2_BAND_TO_ASSET["nir"]: {"href": "fake_nir.tif"},
        }

        with patch("app.ml.predictor._read_band_part", side_effect=mock_read):
            _, H, W, _, _, _, utm_transform, _ = extract_bbox_features(
                WGS84_BBOX, signed_assets, ["red", "nir"], []
            )

        assert H == 100, f"Expected H=100, got {H}"
        assert W == 98,  f"Expected W=98 (snapped to narrower band), got {W}"
        expected = from_bounds(*actual_native, W, H)
        assert abs(utm_transform.c - expected.c) < 1e-3, "transform origin X mismatch after snapping"
        assert abs(utm_transform.f - expected.f) < 1e-3, "transform origin Y mismatch after snapping"
        assert abs(utm_transform.a - expected.a) < 1e-3, "pixel width mismatch after snapping"
        assert abs(utm_transform.e - expected.e) < 1e-3, "pixel height mismatch after snapping"

    def test_feature_matrix_shape_matches_pixel_count(self, cog_path):
        from app.ml.predictor import extract_bbox_features
        signed_assets = self._make_signed_assets({"red": cog_path})
        X, H, W, valid_mask, feature_names, _, _, _ = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )
        assert X is not None
        assert X.shape == (H * W, len(feature_names))
        assert valid_mask.shape == (H * W,)

    def test_missing_bands_returns_none_X(self):
        from app.ml.predictor import extract_bbox_features
        # No assets at all
        X, H, W, _, _, _, _, _ = extract_bbox_features(
            WGS84_BBOX, {}, ["red"], []
        )
        assert X is None
        assert H == 0 and W == 0


# ---------------------------------------------------------------------------
# 3. write_prediction_cog
# ---------------------------------------------------------------------------

class TestWritePredictionCog:

    def _write(self, tmp_path, *, n_classes: int = 3, nodata_fraction: float = 0.0):
        from app.ml.predictor import write_prediction_cog
        H, W = 60, 80
        n_pixels = H * W
        rng = np.random.default_rng(42)
        predictions = rng.integers(1, n_classes + 1, size=n_pixels, dtype=np.int32)
        valid_mask = np.ones(n_pixels, dtype=bool)
        if nodata_fraction > 0:
            nodata_n = int(n_pixels * nodata_fraction)
            valid_mask[:nodata_n] = False
        transform = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, W, H)
        out = tmp_path / "pred.tif"
        write_prediction_cog(predictions, H, W, valid_mask, transform, UTM32N, str(out))
        return out, predictions, valid_mask, H, W

    def test_output_file_exists(self, tmp_path):
        out, *_ = self._write(tmp_path)
        assert out.exists()

    def test_crs_is_native_utm_not_mercator(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() != 3857, "CRS must not be EPSG:3857"
            assert ds.crs.to_epsg() == 32632

    def test_transform_matches_input(self, tmp_path):
        out, _, _, H, W = self._write(tmp_path)
        expected = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, W, H)
        with rasterio.open(out) as ds:
            t = ds.transform
            assert abs(t.c - expected.c) < 1e-3, "transform origin X mismatch"
            assert abs(t.f - expected.f) < 1e-3, "transform origin Y mismatch"
            assert abs(t.a - expected.a) < 1e-3, "pixel width mismatch"
            assert abs(t.e - expected.e) < 1e-3, "pixel height mismatch"

    def test_class_id_pixels_are_preserved(self, tmp_path):
        out, predictions, valid_mask, H, W = self._write(tmp_path)
        with rasterio.open(out) as ds:
            data = ds.read(1).astype(np.int32).ravel()
        # Valid pixels should match the original predictions
        assert np.all(data[valid_mask] == predictions[valid_mask])

    def test_nodata_mask_applied(self, tmp_path):
        out, _, valid_mask, H, W = self._write(tmp_path, nodata_fraction=0.1)
        with rasterio.open(out) as ds:
            assert ds.nodata == 0
            data = ds.read(1).ravel()
        nodata_pixels = ~valid_mask
        assert np.all(data[nodata_pixels] == 0), "Invalid pixels must be written as nodata=0"

    def test_dtype_is_uint16(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint16"

    def test_dimensions_match(self, tmp_path):
        out, _, _, H, W = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.height == H
            assert ds.width == W

    def test_cog_has_overviews(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert len(ds.overviews(1)) > 0, "COG must have at least one overview level"

    def test_cog_is_tiled(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            profile = ds.profile
            assert profile.get("tiled"), "COG must be tiled"


# ---------------------------------------------------------------------------
# 4. write_uncertainty_cog
# ---------------------------------------------------------------------------

class TestWriteUncertaintyCog:

    def _write(self, tmp_path, *, n_classes: int = 4, mode: str = "uniform"):
        """
        mode='uniform'  — max entropy (all classes equally likely → norm=1.0)
        mode='certain'  — one-hot (min entropy → norm≈0.0)
        """
        from app.ml.predictor import write_uncertainty_cog
        H, W = 60, 80
        n_pixels = H * W
        if mode == "uniform":
            probas = np.full((n_pixels, n_classes), 1.0 / n_classes, dtype=np.float32)
        else:  # certain
            probas = np.zeros((n_pixels, n_classes), dtype=np.float32)
            probas[:, 0] = 1.0
        valid_mask = np.ones(n_pixels, dtype=bool)
        transform = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, W, H)
        out = tmp_path / "unc.tif"
        write_uncertainty_cog(probas, H, W, valid_mask, transform, UTM32N, str(out))
        return out, valid_mask, H, W, n_pixels

    def test_output_file_exists(self, tmp_path):
        out, *_ = self._write(tmp_path)
        assert out.exists()

    def test_crs_is_native_utm_not_mercator(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() != 3857
            assert ds.crs.to_epsg() == 32632

    def test_dtype_is_float32(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "float32"

    def test_nodata_is_minus_one(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert ds.nodata == -1.0

    def test_uniform_probas_give_max_entropy(self, tmp_path):
        """Uniform class probabilities → normalised entropy should equal 1.0."""
        out, valid_mask, H, W, n_pixels = self._write(tmp_path, mode="uniform")
        with rasterio.open(out) as ds:
            data = ds.read(1).ravel()
        valid_data = data[valid_mask]
        assert np.allclose(valid_data, 1.0, atol=1e-5), \
            f"Expected entropy≈1.0 for uniform probas, got min={valid_data.min():.4f}"

    def test_certain_probas_give_zero_entropy(self, tmp_path):
        """One-hot class probabilities → normalised entropy should be ≈ 0."""
        out, valid_mask, H, W, n_pixels = self._write(tmp_path, mode="certain")
        with rasterio.open(out) as ds:
            data = ds.read(1).ravel()
        valid_data = data[valid_mask]
        assert np.allclose(valid_data, 0.0, atol=1e-5), \
            f"Expected entropy≈0.0 for one-hot probas, got max={valid_data.max():.4f}"

    def test_entropy_values_in_unit_interval(self, tmp_path):
        """All valid pixel values must lie in [0, 1]."""
        from app.ml.predictor import write_uncertainty_cog
        H, W = 40, 40
        n_pixels = H * W
        rng = np.random.default_rng(7)
        raw = rng.random((n_pixels, 3)).astype(np.float32)
        probas = raw / raw.sum(axis=1, keepdims=True)
        valid_mask = np.ones(n_pixels, dtype=bool)
        transform = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, W, H)
        out = tmp_path / "unc_range.tif"
        write_uncertainty_cog(probas, H, W, valid_mask, transform, UTM32N, str(out))
        with rasterio.open(out) as ds:
            data = ds.read(1).ravel()
        valid_data = data[valid_mask]
        assert np.all(valid_data >= 0.0), "Entropy values must be ≥ 0"
        assert np.all(valid_data <= 1.0), "Entropy values must be ≤ 1"

    def test_nodata_mask_applied(self, tmp_path):
        from app.ml.predictor import write_uncertainty_cog
        H, W = 40, 40
        n_pixels = H * W
        probas = np.full((n_pixels, 2), 0.5, dtype=np.float32)
        valid_mask = np.ones(n_pixels, dtype=bool)
        valid_mask[:100] = False  # first 100 pixels are invalid
        transform = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_RIGHT, UTM_TOP, W, H)
        out = tmp_path / "unc_nodata.tif"
        write_uncertainty_cog(probas, H, W, valid_mask, transform, UTM32N, str(out))
        with rasterio.open(out) as ds:
            data = ds.read(1).ravel()
        assert np.all(data[~valid_mask] == -1.0), "Invalid pixels must equal nodata=-1"

    def test_cog_has_overviews(self, tmp_path):
        out, *_ = self._write(tmp_path)
        with rasterio.open(out) as ds:
            assert len(ds.overviews(1)) > 0


# ---------------------------------------------------------------------------
# 5. Alignment invariant:
#    The WGS84 bounds derived from utm_transform + (H, W) must match the
#    WGS84 bounds reported in actual_bounds_wgs84 from extract_bbox_features.
# ---------------------------------------------------------------------------

class TestAlignmentInvariant:

    def test_utm_transform_and_wgs84_bounds_are_consistent(self, cog_path):
        """
        The WGS84 bounds derived from utm_transform + (H, W) must match
        actual_bounds_wgs84 returned by extract_bbox_features.
        """
        from app.ml.predictor import extract_bbox_features

        signed_assets = {}
        from app.ml.features import S2_BAND_TO_ASSET
        signed_assets[S2_BAND_TO_ASSET["red"]] = {"href": cog_path}

        _, H, W, _, _, actual_bounds, utm_transform, native_crs = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )

        # Derive bounds from the affine
        left_m  = utm_transform.c
        top_m   = utm_transform.f
        right_m = left_m + utm_transform.a * W
        bot_m   = top_m  + utm_transform.e * H   # e is negative

        derived_west, derived_south, derived_east, derived_north = warp.transform_bounds(
            native_crs, "epsg:4326", left_m, bot_m, right_m, top_m
        )
        rep_west, rep_south, rep_east, rep_north = actual_bounds

        tol = 1e-6  # degrees
        assert abs(derived_west  - rep_west)  < tol
        assert abs(derived_south - rep_south) < tol
        assert abs(derived_east  - rep_east)  < tol
        assert abs(derived_north - rep_north) < tol


# ---------------------------------------------------------------------------
# 6. Native resolution & exact pixel values
#    Pixel values returned by both _read_band_part and read_raw_bands_tiff
#    must equal the source DN values exactly (no interpolation).
# ---------------------------------------------------------------------------

class TestNativeResolutionAndExactValues:
    """
    Guarantee that:
    - _read_band_part returns the exact native pixel count (no GSD rounding)
    - Pixel values are unmodified source DNs (nearest-neighbour, not bilinear)
    - read_raw_bands_tiff uses full native resolution when max_size=None
    - The GSD encoded in utm_transform exactly matches native pixel spacing
    """

    FILL_VALUE = 1234.0  # distinctive value to catch any interpolation blending

    def _make_distinctive_cog(self, path: str, *, w: int = 100, h: int = 100) -> None:
        """COG filled with FILL_VALUE at native 10 m GSD."""
        _make_cog(path, fill=self.FILL_VALUE, nodata=0.0)

    def test_pixel_values_equal_source_fill(self, tmp_path):
        """All returned pixel values must equal the source fill — no interpolation."""
        from app.ml.predictor import _read_band_part

        p = str(tmp_path / "exact.tif")
        self._make_distinctive_cog(p)
        _, arr, _, _, _ = _read_band_part(p, WGS84_BBOX, "red", None)
        assert arr is not None
        assert np.all(arr == self.FILL_VALUE), (
            f"Expected all pixels = {self.FILL_VALUE}, got: unique={np.unique(arr)}"
        )

    def test_output_shape_matches_native_pixel_count(self, tmp_path):
        """Output dimensions must match the source's native window pixel count."""
        from app.ml.predictor import _read_band_part

        p = str(tmp_path / "native.tif")
        self._make_distinctive_cog(p)
        # The COG is UTM_W × UTM_H pixels covering exactly the WGS84_BBOX
        _, arr, _, _, _ = _read_band_part(p, WGS84_BBOX, "red", None)
        assert arr.shape == (UTM_H, UTM_W), (
            f"Expected ({UTM_H}, {UTM_W}), got {arr.shape}"
        )

    def test_native_gsd_is_exact_10m(self, tmp_path):
        """
        The transform returned by extract_bbox_features must encode exactly 10 m
        pixel spacing when the source is a 10 m COG.
        """
        from app.ml.predictor import extract_bbox_features
        from app.ml.features import S2_BAND_TO_ASSET

        p = str(tmp_path / "gsd.tif")
        self._make_distinctive_cog(p)
        signed_assets = {S2_BAND_TO_ASSET["red"]: {"href": p}}
        _, H, W, _, _, _, utm_transform, _ = extract_bbox_features(
            WGS84_BBOX, signed_assets, ["red"], []
        )
        xres = abs(utm_transform.a)
        yres = abs(utm_transform.e)
        # Spatial extent / pixel count → exactly native GSD
        extent_x = UTM_RIGHT - UTM_LEFT
        extent_y = UTM_TOP   - UTM_BOTTOM
        assert abs(xres - extent_x / W) < 1e-6, f"xres {xres} != {extent_x/W}"
        assert abs(yres - extent_y / H) < 1e-6, f"yres {yres} != {extent_y/H}"

    def test_max_size_none_returns_native_dimensions(self, tmp_path):
        """max_size=None must return the full native pixel count, not 512."""
        from app.ml.predictor import _read_band_part

        p = str(tmp_path / "full.tif")
        # Make a larger raster so 512-cap would have triggered under old code
        # 200×200 at 10 m covers 2 km × 2 km
        w, h = 200, 200
        data = np.full((h, w), self.FILL_VALUE, dtype=np.float32)
        t = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_LEFT + 2000, UTM_BOTTOM + 2000, w, h)
        with rasterio.open(
            p, "w", driver="GTiff", height=h, width=w, count=1,
            dtype="float32", crs=UTM32N, transform=t, nodata=0.0
        ) as ds:
            ds.write(data, 1)
        bbox = warp.transform_bounds(UTM32N, "epsg:4326", UTM_LEFT, UTM_BOTTOM, UTM_LEFT + 2000, UTM_BOTTOM + 2000)
        _, arr, _, _, _ = _read_band_part(p, list(bbox), "red", None)
        assert arr is not None
        assert arr.shape == (h, w), (
            f"max_size=None should return native ({h},{w}), got {arr.shape}"
        )

    def test_max_size_cap_still_works(self, tmp_path):
        """An explicit max_size cap must still limit output dimensions."""
        from app.ml.predictor import _read_band_part

        p = str(tmp_path / "capped.tif")
        # 200×200 native
        w, h = 200, 200
        data = np.full((h, w), self.FILL_VALUE, dtype=np.float32)
        t = from_bounds(UTM_LEFT, UTM_BOTTOM, UTM_LEFT + 2000, UTM_BOTTOM + 2000, w, h)
        with rasterio.open(
            p, "w", driver="GTiff", height=h, width=w, count=1,
            dtype="float32", crs=UTM32N, transform=t, nodata=0.0
        ) as ds:
            ds.write(data, 1)
        bbox = warp.transform_bounds(UTM32N, "epsg:4326", UTM_LEFT, UTM_BOTTOM, UTM_LEFT + 2000, UTM_BOTTOM + 2000)
        _, arr, _, _, _ = _read_band_part(p, list(bbox), "red", 64)
        assert arr is not None
        assert max(arr.shape) <= 64

    def test_raw_bands_tiff_native_resolution(self, tmp_path):
        """read_raw_bands_tiff with default max_size=None must use native dimensions."""
        import io
        from app.ml.predictor import read_raw_bands_tiff
        from app.ml.features import S2_BAND_TO_ASSET

        p = str(tmp_path / "raw.tif")
        self._make_distinctive_cog(p)
        signed_assets = {S2_BAND_TO_ASSET["red"]: {"href": p}}

        raw_bytes = read_raw_bands_tiff(WGS84_BBOX, signed_assets, ["red"])
        assert raw_bytes is not None
        with rasterio.open(io.BytesIO(raw_bytes)) as ds:
            assert ds.width  == UTM_W, f"Expected width {UTM_W}, got {ds.width}"
            assert ds.height == UTM_H, f"Expected height {UTM_H}, got {ds.height}"

    def test_raw_bands_tiff_exact_pixel_values(self, tmp_path):
        """Pixels in the raw bands TIFF must match the source fill value exactly."""
        import io
        from app.ml.predictor import read_raw_bands_tiff
        from app.ml.features import S2_BAND_TO_ASSET

        p = str(tmp_path / "rawval.tif")
        self._make_distinctive_cog(p)
        signed_assets = {S2_BAND_TO_ASSET["red"]: {"href": p}}

        raw_bytes = read_raw_bands_tiff(WGS84_BBOX, signed_assets, ["red"])
        with rasterio.open(io.BytesIO(raw_bytes)) as ds:
            arr = ds.read(1)
        valid = arr[arr != 0]  # exclude nodata
        assert len(valid) > 0
        assert np.all(valid == self.FILL_VALUE), (
            f"Expected all non-nodata pixels = {self.FILL_VALUE}, got unique={np.unique(valid)}"
        )

    def test_raw_bands_tiff_gsd_matches_source(self, tmp_path):
        """The GSD in the raw bands TIFF transform must equal the source pixel spacing."""
        import io
        from app.ml.predictor import read_raw_bands_tiff
        from app.ml.features import S2_BAND_TO_ASSET

        p = str(tmp_path / "rawgsd.tif")
        self._make_distinctive_cog(p)
        signed_assets = {S2_BAND_TO_ASSET["red"]: {"href": p}}

        raw_bytes = read_raw_bands_tiff(WGS84_BBOX, signed_assets, ["red"])
        with rasterio.open(io.BytesIO(raw_bytes)) as ds:
            xres = abs(ds.transform.a)
            yres = abs(ds.transform.e)
        # Source is 100×100 over 1000×1000 m → exactly 10 m
        assert abs(xres - 10.0) < 1e-3, f"Expected 10 m xres, got {xres}"
        assert abs(yres - 10.0) < 1e-3, f"Expected 10 m yres, got {yres}"
