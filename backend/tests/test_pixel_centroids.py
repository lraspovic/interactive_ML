"""
Tests for pixel centroid extraction and geopackage export.

Verifies:
- _read_band_pixels returns a 5-tuple including affine transform + native CRS
- extract_pixel_features returns (X, feature_names, coords_wgs84)
- pixel centroids are valid WGS84 coordinates inside the polygon bbox
- centroid count matches pixel count in X
- pixel_centroids serialise/deserialise correctly (round-trip)
- gpkg export uses Point geometry when centroids are available
- gpkg export falls back to polygon centroid for legacy rows (NULL centroids)
"""
from __future__ import annotations

import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_utm_cog(path: str, width: int = 400, height: int = 400) -> None:
    """Write a small synthetic UTM32N float32 GeoTIFF at 10 m GSD.

    Covers approximately 650000–654000 E, 5206000–5210000 N (UTM32N),
    which corresponds to roughly 11°E / 47°N in WGS84 — matching _POLY_WGS84.
    """
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS

    crs = CRS.from_epsg(32632)
    # Origin chosen so the test polygon (_POLY_WGS84) falls inside the raster.
    transform = Affine(10.0, 0.0, 650000.0, 0.0, -10.0, 5210000.0)
    # np.random.randint does not accept float dtype — generate int then cast.
    data = np.random.randint(100, 3000, (height, width)).astype(np.float32)

    with rasterio.open(
        path, "w",
        driver="GTiff",
        width=width, height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(data, 1)


# WGS84 polygon that fits inside the 600×600 m patch above
# (approx. 47.0°N, 11.0°E after projection)
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


# ---------------------------------------------------------------------------
# Tests for _read_band_pixels
# ---------------------------------------------------------------------------

class TestReadBandPixels:
    """Unit tests for the updated _read_band_pixels 5-tuple return."""

    def test_returns_five_tuple(self, tmp_path):
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        result = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert len(result) == 5, "must return 5-tuple (name, data, mask, transform, crs)"

    def test_band_name_preserved(self, tmp_path):
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        name, *_ = _read_band_pixels(cog, _POLY_WGS84, "nir")
        assert name == "nir"

    def test_data_and_mask_are_2d_arrays(self, tmp_path):
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        _, data, mask, _, _ = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert data is not None and mask is not None
        assert data.ndim == 2
        assert mask.ndim == 2
        assert data.shape == mask.shape

    def test_transform_is_not_none(self, tmp_path):
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        _, _, _, transform, _ = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert transform is not None

    def test_crs_is_not_3857(self, tmp_path):
        """CRS must be the native UTM CRS, not WebMercator."""
        from app.ml.features import _read_band_pixels
        from rasterio.crs import CRS

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        _, _, _, _, crs = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert crs is not None
        assert CRS.from_epsg(3857) != crs

    def test_crs_is_utm32n(self, tmp_path):
        from app.ml.features import _read_band_pixels
        from rasterio.crs import CRS

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        _, _, _, _, crs = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert crs == CRS.from_epsg(32632)

    def test_failure_returns_all_none(self):
        """Bad path must return (name, None, None, None, None)."""
        from app.ml.features import _read_band_pixels

        name, data, mask, transform, crs = _read_band_pixels(
            "/nonexistent/path.tif", _POLY_WGS84, "red"
        )
        assert name == "red"
        assert data is None
        assert mask is None
        assert transform is None
        assert crs is None

    def test_transform_origin_near_native_bounds(self, tmp_path):
        """The output transform west/north should equal the native bbox edges to <1 m."""
        import rasterio
        from rasterio import warp
        from rasterio.features import bounds as rio_bounds
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "band.tif")
        _make_utm_cog(cog)
        _, _, _, transform, crs = _read_band_pixels(cog, _POLY_WGS84, "red")

        aoi_bounds = rio_bounds(_POLY_WGS84)
        with rasterio.open(cog) as ds:
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)

        # transform_from_bounds sets c=west, f=north — should match native bbox to <1 m
        assert abs(transform.c - native_bounds[0]) < 1.0, "west origin should match native west"
        assert abs(transform.f - native_bounds[3]) < 1.0, "north origin should match native north"


# ---------------------------------------------------------------------------
# Tests for extract_pixel_features
# ---------------------------------------------------------------------------

class TestExtractPixelFeatures:
    """Integration tests using real (local temp) COG files."""

    def _make_signed_assets(self, tmp_path) -> dict:
        """Create two local band COGs and return a signed-asset-style dict."""
        red_path = str(tmp_path / "red.tif")
        nir_path = str(tmp_path / "nir.tif")
        _make_utm_cog(red_path)
        _make_utm_cog(nir_path)
        return {
            "B04": {"href": red_path},
            "B08": {"href": nir_path},
        }

    def test_returns_three_tuple(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        result = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert len(result) == 3

    def test_x_is_2d_float32(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        X, names, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert X is not None
        assert X.ndim == 2
        assert X.dtype == np.float32

    def test_feature_names_match_columns(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        X, names, _ = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert X.shape[1] == len(names)

    def test_coords_shape_matches_x(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        X, _, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert coords is not None
        assert coords.shape == (X.shape[0], 2), f"expected ({X.shape[0]}, 2), got {coords.shape}"

    def test_coords_dtype_is_float64(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        _, _, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert coords.dtype == np.float64

    def test_coords_are_valid_wgs84(self, tmp_path):
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        _, _, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        lons, lats = coords[:, 0], coords[:, 1]
        assert np.all(lons >= -180) and np.all(lons <= 180), "longitudes out of range"
        assert np.all(lats >= -90) and np.all(lats <= 90), "latitudes out of range"

    def test_centroids_inside_polygon_bbox(self, tmp_path):
        """Every pixel centroid must lie within the polygon's WGS84 bounding box."""
        from rasterio.features import bounds as rio_bounds
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        _, _, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        minx, miny, maxx, maxy = rio_bounds(_POLY_WGS84)
        lons, lats = coords[:, 0], coords[:, 1]
        # Allow a half-pixel margin (≈0.0001°) for edge pixels
        margin = 0.001
        assert np.all(lons >= minx - margin) and np.all(lons <= maxx + margin)
        assert np.all(lats >= miny - margin) and np.all(lats <= maxy + margin)

    def test_empty_assets_returns_none_tuple(self):
        from app.ml.features import extract_pixel_features

        X, names, coords = extract_pixel_features(
            _POLY_WGS84, {}, available_bands=["red"], enabled_indices=[]
        )
        assert X is None
        assert names == []
        assert coords is None

    def test_spectral_index_does_not_break_centroids(self, tmp_path):
        """Adding NDVI should not affect centroid count or validity."""
        from app.ml.features import extract_pixel_features

        assets = self._make_signed_assets(tmp_path)
        X, names, coords = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=["NDVI"]
        )
        assert "NDVI" in names
        assert X.shape[0] == coords.shape[0]


# ---------------------------------------------------------------------------
# Tests for pixel_centroids serialisation round-trip
# ---------------------------------------------------------------------------

class TestCentroidSerialisation:
    """Verify the numpy save/load round-trip used in the training pipeline."""

    def test_round_trip_float64(self):
        original = np.array([[11.001, 47.002], [11.003, 47.004]], dtype=np.float64)
        buf = io.BytesIO()
        np.save(buf, original)
        restored = np.load(io.BytesIO(buf.getvalue()))
        np.testing.assert_array_equal(original, restored)

    def test_dtype_preserved(self):
        original = np.random.rand(100, 2).astype(np.float64)
        buf = io.BytesIO()
        np.save(buf, original)
        restored = np.load(io.BytesIO(buf.getvalue()))
        assert restored.dtype == np.float64

    def test_shape_preserved(self):
        original = np.random.rand(42, 2).astype(np.float64)
        buf = io.BytesIO()
        np.save(buf, original)
        restored = np.load(io.BytesIO(buf.getvalue()))
        assert restored.shape == (42, 2)

    def test_values_preserved_within_tolerance(self):
        original = np.array([[10.123456789, 47.987654321]], dtype=np.float64)
        buf = io.BytesIO()
        np.save(buf, original)
        restored = np.load(io.BytesIO(buf.getvalue()))
        np.testing.assert_allclose(original, restored, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests for gpkg export geometry logic
# ---------------------------------------------------------------------------

class TestGpkgExportGeometry:
    """
    Test the pixel geometry selection logic used in download_features_gpkg,
    isolated from DB and HTTP calls.
    """

    def _build_pixel_records(
        self,
        X: np.ndarray,
        feature_names: list[str],
        geom,
        label: str,
        class_id: int,
        sample_id: str,
        coords_wgs84: np.ndarray | None,
    ) -> list[dict]:
        """Mirror the export logic from download_features_gpkg."""
        from shapely.geometry import Point

        pixel_records = []
        for px_idx in range(X.shape[0]):
            if coords_wgs84 is not None:
                lon, lat = coords_wgs84[px_idx]
                px_geom = Point(lon, lat)
            else:
                px_geom = geom.centroid

            px_row: dict = {
                "geometry": px_geom,
                "sample_id": sample_id,
                "pixel_idx": px_idx,
                "label": label,
                "class_id": class_id,
            }
            for i, name in enumerate(feature_names):
                px_row[name] = round(float(X[px_idx, i]), 6)
            pixel_records.append(px_row)
        return pixel_records

    def _sample_data(self):
        from shapely.geometry import shape

        np.random.seed(0)
        X = np.random.rand(10, 3).astype(np.float32)
        feature_names = ["red", "nir", "NDVI"]
        geom = shape(_POLY_WGS84)
        coords_wgs84 = np.column_stack([
            np.linspace(11.001, 11.004, 10),
            np.linspace(46.999, 47.002, 10),
        ]).astype(np.float64)
        return X, feature_names, geom, coords_wgs84

    def test_with_centroids_geometry_is_point(self):
        from shapely.geometry import Point

        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        for r in records:
            assert isinstance(r["geometry"], Point), "expected Point geometry"

    def test_with_centroids_correct_coordinates(self):
        from shapely.geometry import Point

        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        for i, r in enumerate(records):
            assert abs(r["geometry"].x - coords[i, 0]) < 1e-9
            assert abs(r["geometry"].y - coords[i, 1]) < 1e-9

    def test_without_centroids_geometry_is_polygon_centroid(self):
        from shapely.geometry import Point

        X, names, geom, _ = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", None)
        expected_centroid = geom.centroid
        for r in records:
            # Should be a point (centroid of polygon)
            assert isinstance(r["geometry"], Point)
            assert abs(r["geometry"].x - expected_centroid.x) < 1e-9
            assert abs(r["geometry"].y - expected_centroid.y) < 1e-9

    def test_pixel_count_matches_x_rows(self):
        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        assert len(records) == X.shape[0]

    def test_feature_values_correct(self):
        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        for px_idx, r in enumerate(records):
            for i, name in enumerate(names):
                assert abs(r[name] - round(float(X[px_idx, i]), 6)) < 1e-7

    def test_pixel_idx_assigned_sequentially(self):
        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        for i, r in enumerate(records):
            assert r["pixel_idx"] == i

    def test_geopandas_layer_geometry_type_is_point(self):
        """Full gpkg write round-trip: pixel_values layer must be a Point layer."""
        import geopandas as gpd

        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        assert gdf.geom_type.unique().tolist() == ["Point"], \
            f"unexpected geometry types: {gdf.geom_type.unique().tolist()}"

    def test_gpkg_roundtrip_preserves_point_type(self, tmp_path):
        """Write and re-read a .gpkg; pixel_values must still be a Point layer."""
        import geopandas as gpd

        X, names, geom, coords = self._sample_data()
        records = self._build_pixel_records(X, names, geom, "forest", 1, "sid-1", coords)
        pixel_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

        gpkg_path = str(tmp_path / "test.gpkg")
        pixel_gdf.to_file(gpkg_path, driver="GPKG", layer="pixel_values", engine="pyogrio")

        loaded = gpd.read_file(gpkg_path, layer="pixel_values")
        assert loaded.geom_type.unique().tolist() == ["Point"]
        assert len(loaded) == X.shape[0]

    def test_polygon_layer_remains_polygon(self, tmp_path):
        """The training_polygons layer must still be a Polygon layer."""
        import geopandas as gpd
        from shapely.geometry import shape

        geom = shape(_POLY_WGS84)
        np.random.seed(1)
        X = np.random.rand(5, 2).astype(np.float32)
        feature_names = ["red", "nir"]

        poly_records = [{
            "geometry": geom,
            "sample_id": "sid-1",
            "label": "forest",
            "class_id": 1,
            "n_pixels": 5,
            "red_mean": round(float(X[:, 0].mean()), 6),
            "nir_mean": round(float(X[:, 1].mean()), 6),
        }]
        poly_gdf = gpd.GeoDataFrame(poly_records, crs="EPSG:4326")

        gpkg_path = str(tmp_path / "test2.gpkg")
        poly_gdf.to_file(gpkg_path, driver="GPKG", layer="training_polygons", engine="pyogrio")
        loaded = gpd.read_file(gpkg_path, layer="training_polygons")
        assert loaded.geom_type.unique().tolist() == ["Polygon"]


# ---------------------------------------------------------------------------
# Tests for native resolution and exact pixel values (features.py path)
# ---------------------------------------------------------------------------

KNOWN_FILL = 777.0  # distinctive value — any blending would produce non-integer result


def _make_fill_cog(path: str, fill: float = KNOWN_FILL, w: int = 400, h: int = 400) -> None:
    """COG filled uniformly with *fill*, same spatial extent as _POLY_WGS84 + buffer."""
    import rasterio
    from affine import Affine
    from rasterio.crs import CRS

    crs = CRS.from_epsg(32632)
    transform = Affine(10.0, 0.0, 650000.0, 0.0, -10.0, 5210000.0)
    data = np.full((h, w), fill, dtype=np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h,
        count=1, dtype="float32", crs=crs, transform=transform, nodata=0
    ) as dst:
        dst.write(data, 1)


class TestNativeResolutionFeatures:
    """
    Verify that _read_band_pixels and extract_pixel_features preserve exact
    source DN values (nearest-neighbour, not bilinear) and use native
    window pixel count rather than a GSD-derived approximation.
    """

    def test_read_band_pixels_exact_fill_value(self, tmp_path):
        """All returned pixel values must equal the source fill — no blending."""
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "fill.tif")
        _make_fill_cog(cog)
        _, data, mask, _, _ = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert data is not None
        valid = data[mask]
        assert len(valid) > 0
        assert np.all(valid == KNOWN_FILL), (
            f"Expected all valid pixels = {KNOWN_FILL}, got unique={np.unique(valid)}"
        )

    def test_read_band_pixels_native_shape_matches_window(self, tmp_path):
        """
        Without a target_shape override, _read_band_pixels must return the
        native window pixel count — not a GSD-approximated count.
        """
        import rasterio
        import rasterio.features as rio_feat
        from rasterio import warp, windows as rio_windows
        from app.ml.features import _read_band_pixels

        cog = str(tmp_path / "native.tif")
        _make_fill_cog(cog)

        # Compute expected shape from the native window directly
        aoi_bounds = rio_feat.bounds(_POLY_WGS84)
        with rasterio.open(cog) as ds:
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            win = rio_windows.from_bounds(*native_bounds, transform=ds.transform)
            win = win.intersection(rio_windows.Window(0, 0, ds.width, ds.height))
            expected_h = max(1, round(win.height))
            expected_w = max(1, round(win.width))

        _, data, _, _, _ = _read_band_pixels(cog, _POLY_WGS84, "red")
        assert data is not None
        assert data.shape == (expected_h, expected_w), (
            f"Expected {(expected_h, expected_w)}, got {data.shape}"
        )

    def test_extract_pixel_features_values_match_fill(self, tmp_path):
        """
        All raw band feature values in X must equal the source fill value exactly.
        If bilinear resampling were used the values could be fractional.
        """
        from app.ml.features import extract_pixel_features

        red_path = str(tmp_path / "red_fill.tif")
        nir_path = str(tmp_path / "nir_fill.tif")
        _make_fill_cog(red_path, fill=KNOWN_FILL)
        _make_fill_cog(nir_path, fill=KNOWN_FILL + 100)
        assets = {
            "B04": {"href": red_path},
            "B08": {"href": nir_path},
        }
        X, names, _ = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red", "nir"], enabled_indices=[]
        )
        assert X is not None

        red_col = X[:, names.index("red")]
        nir_col = X[:, names.index("nir")]
        assert np.all(red_col == KNOWN_FILL), (
            f"red column: expected {KNOWN_FILL}, got unique={np.unique(red_col)}"
        )
        assert np.all(nir_col == KNOWN_FILL + 100), (
            f"nir column: expected {KNOWN_FILL + 100}, got unique={np.unique(nir_col)}"
        )

    def test_extract_pixel_features_target_shape_is_native(self, tmp_path):
        """
        The pixel count from extract_pixel_features must match the native window
        pixel count from the source COG, not a GSD-rounded approximation.
        """
        import rasterio
        import rasterio.features as rio_feat
        from rasterio import warp, windows as rio_windows
        from app.ml.features import extract_pixel_features

        cog = str(tmp_path / "shape.tif")
        _make_fill_cog(cog)
        assets = {"B04": {"href": cog}}

        aoi_bounds = rio_feat.bounds(_POLY_WGS84)
        with rasterio.open(cog) as ds:
            native_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            win = rio_windows.from_bounds(*native_bounds, transform=ds.transform)
            win = win.intersection(rio_windows.Window(0, 0, ds.width, ds.height))
            expected_h = max(1, round(win.height))
            expected_w = max(1, round(win.width))

        X, _, _ = extract_pixel_features(
            _POLY_WGS84, assets, available_bands=["red"], enabled_indices=[]
        )
        assert X is not None
        # n_valid_pixels <= expected_h * expected_w (some masked by polygon boundary)
        assert X.shape[0] <= expected_h * expected_w
        # At least 1 pixel must be valid for this to be a meaningful test
        assert X.shape[0] > 0
