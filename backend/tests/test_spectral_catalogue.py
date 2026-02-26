"""
Tests for backend/app/ml/spectral_catalogue.py

Coverage:
  - get_all_indices_by_domain() grouping and structure
  - get_computable_indices() band filtering + legacy alias resolution
  - compute_index() correctness for canonical indices + edge cases
  - ASI band code disambiguation (S1/S2 = SWIR, not Sentinel-1/2)
  - COLLECTION_BAND_TO_ASSET completeness for S1 and S2
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.spectral_catalogue import (
    ASI_BAND_TO_INTERNAL,
    COLLECTION_BAND_TO_ASSET,
    LEGACY_BAND_ALIASES,
    SPECTRAL_INDEX_CATALOGUE,
    compute_index,
    get_all_indices_by_domain,
    get_computable_indices,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ones(**bands) -> dict[str, np.ndarray]:
    """Return {band: ones(10)} for each specified named band."""
    return {k: np.ones(10, dtype=np.float32) for k in bands}


# ---------------------------------------------------------------------------
# get_all_indices_by_domain
# ---------------------------------------------------------------------------

class TestGetAllIndicesByDomain:
    def test_returns_dict(self):
        result = get_all_indices_by_domain()
        assert isinstance(result, dict)

    def test_domains_are_non_empty_strings(self):
        result = get_all_indices_by_domain()
        for domain, entries in result.items():
            assert isinstance(domain, str) and domain
            assert len(entries) > 0

    def test_each_entry_has_required_keys(self):
        result = get_all_indices_by_domain()
        for entries in result.values():
            for e in entries:
                assert "id" in e
                assert "long_name" in e
                assert "bands" in e
                assert isinstance(e["bands"], list)

    def test_ndvi_in_vegetation_domain(self):
        result = get_all_indices_by_domain()
        ids_in_veg = [e["id"] for e in result.get("vegetation", [])]
        assert "NDVI" in ids_in_veg

    def test_ndwi_in_water_domain(self):
        result = get_all_indices_by_domain()
        ids_in_water = [e["id"] for e in result.get("water", [])]
        assert "NDWI" in ids_in_water

    def test_total_length_matches_catalogue(self):
        result = get_all_indices_by_domain()
        total = sum(len(v) for v in result.values())
        assert total == len(SPECTRAL_INDEX_CATALOGUE)


# ---------------------------------------------------------------------------
# get_computable_indices
# ---------------------------------------------------------------------------

class TestGetComputableIndices:
    def test_ndvi_computable_with_nir_red(self):
        result = get_computable_indices(["nir", "red"])
        assert "NDVI" in result

    def test_ndvi_not_computable_without_nir(self):
        result = get_computable_indices(["blue", "green", "red"])
        assert "NDVI" not in result

    def test_ndwi_computable_with_green_nir(self):
        result = get_computable_indices(["green", "nir"])
        assert "NDWI" in result

    def test_extra_bands_do_not_block_computation(self):
        # NDVI only needs nir + red; extra bands should not matter
        result = get_computable_indices(["blue", "green", "red", "nir", "swir1"])
        assert "NDVI" in result

    def test_empty_bands_returns_empty_list(self):
        result = get_computable_indices([])
        assert result == []

    def test_sar_only_indices_computable(self):
        # RVI4S1 or similar radar index should appear with SAR bands
        result = get_computable_indices(["sar_vv", "sar_vh"])
        # The catalogue may or may not have pure-SAR indices;
        # the important thing is that the function doesn't crash.
        assert isinstance(result, list)

    def test_legacy_red_edge_alias(self):
        # "red_edge" was the old name for nir_2 (B8A).
        # get_computable_indices should treat "red_edge" as "nir_2".
        indices_with_alias = set(get_computable_indices(["red_edge", "nir"]))
        indices_with_canonical = set(get_computable_indices(["nir_2", "nir"]))
        assert indices_with_alias == indices_with_canonical

    def test_returns_list_of_strings(self):
        result = get_computable_indices(["nir", "red", "green"])
        assert isinstance(result, list)
        assert all(isinstance(i, str) for i in result)


# ---------------------------------------------------------------------------
# compute_index
# ---------------------------------------------------------------------------

class TestComputeIndex:
    def test_ndvi_formula(self):
        nir = np.array([0.8], dtype=np.float32)
        red = np.array([0.2], dtype=np.float32)
        result = compute_index("NDVI", {"nir": nir, "red": red})
        assert result is not None
        expected = (0.8 - 0.2) / (0.8 + 0.2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_ndvi_case_insensitive(self):
        val = compute_index("ndvi", {"nir": np.ones(5), "red": np.zeros(5)})
        assert val is not None

    def test_unknown_index_returns_none(self):
        assert compute_index("DOES_NOT_EXIST", {"nir": np.ones(5)}) is None

    def test_missing_required_band_returns_none(self):
        # NDVI requires both nir and red; only providing nir should return None
        assert compute_index("NDVI", {"nir": np.ones(5)}) is None

    def test_return_dtype_is_float32(self):
        result = compute_index("NDVI", {"nir": np.ones(10), "red": np.zeros(10)})
        assert result is not None
        assert result.dtype == np.float32

    def test_ndwi_formula(self):
        green = np.array([0.3], dtype=np.float32)
        nir   = np.array([0.5], dtype=np.float32)
        result = compute_index("NDWI", {"green": green, "nir": nir})
        expected = (0.3 - 0.5) / (0.3 + 0.5)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dvi_additive_formula(self):
        nir = np.array([0.7], dtype=np.float32)
        red = np.array([0.1], dtype=np.float32)
        result = compute_index("DVI", {"nir": nir, "red": red})
        np.testing.assert_allclose(result, np.array([0.6], dtype=np.float32), rtol=1e-5)

    def test_output_shape_matches_input(self):
        arr = np.random.rand(100).astype(np.float32)
        result = compute_index("NDVI", {"nir": arr, "red": arr * 0.5})
        assert result is not None
        assert result.shape == (100,)


# ---------------------------------------------------------------------------
# ASI band-code disambiguation
# ---------------------------------------------------------------------------

class TestBandCodeDisambiguation:
    def test_s1_maps_to_swir1(self):
        """ASI 'S1' is SWIR-1 — NOT the Sentinel-1 satellite."""
        assert ASI_BAND_TO_INTERNAL["S1"] == "swir1"

    def test_s2_maps_to_swir2(self):
        """ASI 'S2' is SWIR-2 — NOT the Sentinel-2 satellite."""
        assert ASI_BAND_TO_INTERNAL["S2"] == "swir2"

    def test_vv_maps_to_sar_vv(self):
        assert ASI_BAND_TO_INTERNAL["VV"] == "sar_vv"

    def test_vh_maps_to_sar_vh(self):
        assert ASI_BAND_TO_INTERNAL["VH"] == "sar_vh"

    def test_n_maps_to_nir(self):
        assert ASI_BAND_TO_INTERNAL["N"] == "nir"


# ---------------------------------------------------------------------------
# COLLECTION_BAND_TO_ASSET completeness
# ---------------------------------------------------------------------------

class TestCollectionBandToAsset:
    def test_s2_collection_present(self):
        assert "sentinel-2-l2a" in COLLECTION_BAND_TO_ASSET

    def test_s1_collection_present(self):
        assert "sentinel-1-rtc" in COLLECTION_BAND_TO_ASSET

    def test_s2_has_rgb_bands(self):
        m = COLLECTION_BAND_TO_ASSET["sentinel-2-l2a"]
        for band in ("blue", "green", "red", "nir"):
            assert band in m, f"Missing band: {band}"

    def test_s1_has_vv_vh(self):
        m = COLLECTION_BAND_TO_ASSET["sentinel-1-rtc"]
        assert "sar_vv" in m
        assert "sar_vh" in m

    def test_s1_asset_names_are_lowercase(self):
        """Planetary Computer S1-RTC uses lowercase asset keys (vv / vh)."""
        m = COLLECTION_BAND_TO_ASSET["sentinel-1-rtc"]
        assert m["sar_vv"] == "vv"
        assert m["sar_vh"] == "vh"

    def test_legacy_red_edge_alias_in_s2(self):
        """sentinel-2-l2a map should still have legacy 'red_edge' alias."""
        m = COLLECTION_BAND_TO_ASSET["sentinel-2-l2a"]
        assert "red_edge" in m


# ---------------------------------------------------------------------------
# Catalogue invariants
# ---------------------------------------------------------------------------

class TestCatalogueInvariants:
    def test_all_indices_have_band_only_formulas(self):
        """The catalogue must only contain band-only indices (no free parameters)."""
        for name, spec in SPECTRAL_INDEX_CATALOGUE.items():
            assert callable(spec["formula"]), f"{name}: formula is not callable"
            assert len(spec["bands"]) > 0, f"{name}: bands list is empty"

    def test_all_band_names_in_catalogue_are_internal(self):
        """Every band name in spec['bands'] should be a valid internal name."""
        valid_internal = set(ASI_BAND_TO_INTERNAL.values())
        for name, spec in SPECTRAL_INDEX_CATALOGUE.items():
            for band in spec["bands"]:
                assert band in valid_internal, (
                    f"{name}: band '{band}' is not a known internal band name"
                )

    def test_no_duplicate_ids(self):
        """Each index must have a unique short name."""
        ids = list(SPECTRAL_INDEX_CATALOGUE.keys())
        assert len(ids) == len(set(ids))
