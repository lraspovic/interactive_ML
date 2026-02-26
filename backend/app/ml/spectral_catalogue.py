"""
Spectral index catalogue derived from the Awesome Spectral Indices (ASI) project.
https://github.com/awesome-spectral-indices/spectral

Design choices
--------------
* Only **band-only** indices are included — indices whose `bands` list consists
  exclusively of real sensor bands (no free parameters such as L, gamma, alpha,
  sla, C1, C2, …).  This avoids the UX complexity of exposing per-index scalar
  constants to the user.
* The `kernel` domain (kNDVI, kEVI, …) is excluded — it requires precomputed
  kernel pseudo-bands that lie outside this pipeline.
* Thermal bands (T, T1) and platform-specific bands (Y, G1, A) that are not
  available from Sentinel-1/2 are retained in the catalogue but are gated by
  ``get_computable_indices`` — they will simply never be selectable when only
  S1/S2 bands are available.

ASI band code → internal logical band name
------------------------------------------
Note the potential confusion: in ASI, ``S1`` and ``S2`` are _band_ codes for
SWIR-1 and SWIR-2 respectively — they are NOT references to the Sentinel-1 or
Sentinel-2 satellites.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Band code mapping: ASI abbreviation → internal logical band name
# ---------------------------------------------------------------------------

ASI_BAND_TO_INTERNAL: dict[str, str] = {
    "B":   "blue",
    "G":   "green",
    "R":   "red",
    "N":   "nir",
    "S1":  "swir1",    # SWIR-1 (~1.6 µm) – NOT Sentinel-1 the satellite!
    "S2":  "swir2",    # SWIR-2 (~2.2 µm) – NOT Sentinel-2 the satellite!
    "RE1": "red_edge_1",
    "RE2": "red_edge_2",
    "RE3": "red_edge_3",
    "N2":  "nir_2",    # Sentinel-2 B8A (~865 nm)
    "VV":  "sar_vv",
    "VH":  "sar_vh",
    # Bands below are included for catalogue completeness but will not be
    # selectable until the corresponding sensor support is added.
    "T":   "thermal",
    "T1":  "thermal_1",
    "A":   "aerosol",
    "HH":  "sar_hh",
    "HV":  "sar_hv",
}

INTERNAL_TO_ASI_BAND: dict[str, str] = {v: k for k, v in ASI_BAND_TO_INTERNAL.items()}

# Set of band codes that the current sensor support can supply
SUPPORTED_ASI_BANDS: set[str] = {
    "B", "G", "R", "N", "S1", "S2",
    "RE1", "RE2", "RE3", "N2",
    "VV", "VH",
}

# ---------------------------------------------------------------------------
# STAC collection → internal band name → asset key
# ---------------------------------------------------------------------------

COLLECTION_BAND_TO_ASSET: dict[str, dict[str, str]] = {
    "sentinel-2-l2a": {
        "blue":        "B02",
        "green":       "B03",
        "red":         "B04",
        "red_edge_1":  "B05",
        "red_edge_2":  "B06",
        "red_edge_3":  "B07",
        "nir":         "B08",
        "nir_2":       "B8A",
        "red_edge":    "B8A",   # legacy alias kept for backward compatibility
        "swir1":       "B11",
        "swir2":       "B12",
    },
    "sentinel-1-rtc": {
        "sar_vv": "vv",
        "sar_vh": "vh",
    },
}

# Canonical band list per sensor (in display order)
SENSOR_BANDS: dict[str, list[str]] = {
    "sentinel-2-l2a": [
        "blue", "green", "red",
        "red_edge_1", "red_edge_2", "red_edge_3",
        "nir", "nir_2",
        "swir1", "swir2",
    ],
    "sentinel-1-rtc": [
        "sar_vv", "sar_vh",
    ],
}

# Bands that carry the legacy "red_edge" name from previous projects
LEGACY_BAND_ALIASES: dict[str, str] = {
    "red_edge": "nir_2",  # B8A was mislabelled as red_edge in older projects
}

# ---------------------------------------------------------------------------
# Spectral index catalogue
#
# Each entry:
#   "short_name": {
#       "long_name": str,
#       "domain": str,
#       "bands": list[str],          # internal logical band names
#       "formula": callable,         # fn(band_dict: dict[str, np.ndarray]) → np.ndarray
#   }
# ---------------------------------------------------------------------------

_EPS = 1e-10


def _mk(formula_str: str, **kw: Any) -> Any:
    """Lightweight helper — unused, kept for future dynamic eval."""
    return formula_str


# The catalogue is keyed by the ASI ``short_name`` (upper-case canonical form).
SPECTRAL_INDEX_CATALOGUE: dict[str, dict[str, Any]] = {}

def _register(short_name: str, long_name: str, domain: str, asi_bands: list[str], fn):
    """Add one index to the catalogue after translating ASI band codes."""
    internal_bands = [ASI_BAND_TO_INTERNAL[b] for b in asi_bands]
    SPECTRAL_INDEX_CATALOGUE[short_name] = {
        "long_name": long_name,
        "domain": domain,
        "bands": internal_bands,
        "formula": fn,
    }


# ── Vegetation ──────────────────────────────────────────────────────────────

_register("NDVI",   "Normalized Difference Vegetation Index",        "vegetation", ["N", "R"],
    lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS))

_register("DVI",    "Difference Vegetation Index",                   "vegetation", ["N", "R"],
    lambda b: b["nir"] - b["red"])

_register("SR",     "Simple Ratio",                                  "vegetation", ["N", "R"],
    lambda b: b["nir"] / (b["red"] + _EPS))

_register("GNDVI",  "Green Normalized Difference Vegetation Index",  "vegetation", ["N", "G"],
    lambda b: (b["nir"] - b["green"]) / (b["nir"] + b["green"] + _EPS))

_register("GRVI",   "Green Ratio Vegetation Index",                  "vegetation", ["N", "G"],
    lambda b: b["nir"] / (b["green"] + _EPS))

_register("IPVI",   "Infrared Percentage Vegetation Index",          "vegetation", ["N", "R"],
    lambda b: b["nir"] / (b["nir"] + b["red"] + _EPS))

_register("NLI",    "Non-Linear Vegetation Index",                   "vegetation", ["N", "R"],
    lambda b: (b["nir"] ** 2 - b["red"]) / (b["nir"] ** 2 + b["red"] + _EPS))

_register("RDVI",   "Renormalized Difference Vegetation Index",      "vegetation", ["N", "R"],
    lambda b: (b["nir"] - b["red"]) / ((b["nir"] + b["red"]) ** 0.5 + _EPS))

_register("MSAVI",  "Modified Soil-Adjusted Vegetation Index",       "vegetation", ["N", "R"],
    lambda b: 0.5 * (2 * b["nir"] + 1 - ((2 * b["nir"] + 1) ** 2 - 8 * (b["nir"] - b["red"])) ** 0.5))

_register("MSR",    "Modified Simple Ratio",                         "vegetation", ["N", "R"],
    lambda b: (b["nir"] / (b["red"] + _EPS) - 1) / ((b["nir"] / (b["red"] + _EPS) + 1) ** 0.5 + _EPS))

_register("TVI",    "Transformed Vegetation Index",                  "vegetation", ["N", "R"],
    lambda b: (((b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS)) + 0.5) ** 0.5)

_register("MGRVI",  "Modified Green Red Vegetation Index",           "vegetation", ["G", "R"],
    lambda b: (b["green"] ** 2 - b["red"] ** 2) / (b["green"] ** 2 + b["red"] ** 2 + _EPS))

_register("NGRDI",  "Normalized Green Red Difference Index",         "vegetation", ["G", "R"],
    lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] + _EPS))

_register("VARI",   "Visible Atmospherically Resistant Index",       "vegetation", ["G", "R", "B"],
    lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] - b["blue"] + _EPS))

_register("VIG",    "Vegetation Index Green",                        "vegetation", ["G", "R"],
    lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] + _EPS))

_register("ExG",    "Excess Green Index",                            "vegetation", ["G", "R", "B"],
    lambda b: 2 * b["green"] - b["red"] - b["blue"])

_register("ExGR",   "ExG - ExR Vegetation Index",                   "vegetation", ["G", "R", "B"],
    lambda b: (2.0 * b["green"] - b["red"] - b["blue"]) - (1.3 * b["red"] - b["green"]))

_register("ExR",    "Excess Red Index",                              "vegetation", ["R", "G"],
    lambda b: 1.3 * b["red"] - b["green"])

_register("GLI",    "Green Leaf Index",                              "vegetation", ["G", "R", "B"],
    lambda b: (2.0 * b["green"] - b["red"] - b["blue"]) / (2.0 * b["green"] + b["red"] + b["blue"] + _EPS))

_register("GCC",    "Green Chromatic Coordinate",                    "vegetation", ["G", "R", "B"],
    lambda b: b["green"] / (b["red"] + b["green"] + b["blue"] + _EPS))

_register("RCC",    "Red Chromatic Coordinate",                      "vegetation", ["R", "G", "B"],
    lambda b: b["red"] / (b["red"] + b["green"] + b["blue"] + _EPS))

_register("BCC",    "Blue Chromatic Coordinate",                     "vegetation", ["B", "R", "G"],
    lambda b: b["blue"] / (b["red"] + b["green"] + b["blue"] + _EPS))

_register("TGI",    "Triangular Greenness Index",                    "vegetation", ["R", "G", "B"],
    lambda b: -0.5 * (190 * (b["red"] - b["green"]) - 120 * (b["red"] - b["blue"])))

_register("GEMI",   "Global Environment Monitoring Index",           "vegetation", ["N", "R"],
    lambda b: (
        (2.0 * (b["nir"] ** 2 - b["red"] ** 2) + 1.5 * b["nir"] + 0.5 * b["red"])
        / (b["nir"] + b["red"] + 0.5)
    ) * (1.0 - 0.25 * (
        (2.0 * (b["nir"] ** 2 - b["red"] ** 2) + 1.5 * b["nir"] + 0.5 * b["red"])
        / (b["nir"] + b["red"] + 0.5)
    )) - (b["red"] - 0.125) / (1 - b["red"] + _EPS))

_register("NDII",   "Normalized Difference Infrared Index",          "vegetation", ["N", "S1"],
    lambda b: (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + _EPS))

_register("MSI",    "Moisture Stress Index",                         "vegetation", ["S1", "N"],
    lambda b: b["swir1"] / (b["nir"] + _EPS))

_register("NMDI",   "Normalized Multi-band Drought Index",           "vegetation", ["N", "S1", "S2"],
    lambda b: (b["nir"] - (b["swir1"] - b["swir2"])) / (b["nir"] + (b["swir1"] - b["swir2"]) + _EPS))

_register("GVMI",   "Global Vegetation Moisture Index",              "vegetation", ["N", "S2"],
    lambda b: ((b["nir"] + 0.1) - (b["swir2"] + 0.02)) / ((b["nir"] + 0.1) + (b["swir2"] + 0.02) + _EPS))

_register("SLAVI",  "Specific Leaf Area Vegetation Index",           "vegetation", ["N", "R", "S2"],
    lambda b: b["nir"] / (b["red"] + b["swir2"] + _EPS))

_register("MVI",    "Mangrove Vegetation Index",                     "vegetation", ["N", "G", "S1"],
    lambda b: (b["nir"] - b["green"]) / (b["swir1"] - b["green"] + _EPS))

_register("MI",     "Mangrove Index",                                "vegetation", ["N", "S1"],
    lambda b: (b["nir"] - b["swir1"]) / (b["nir"] * b["swir1"] + _EPS))

_register("DSWI5",  "Disease-Water Stress Index 5",                  "vegetation", ["N", "G", "S1", "R"],
    lambda b: (b["nir"] + b["green"]) / (b["swir1"] + b["red"] + _EPS))

_register("DSWI1",  "Disease-Water Stress Index 1",                  "vegetation", ["N", "S1"],
    lambda b: b["nir"] / (b["swir1"] + _EPS))

_register("DSWI4",  "Disease-Water Stress Index 4",                  "vegetation", ["G", "R"],
    lambda b: b["green"] / (b["red"] + _EPS))

_register("NIRv",   "Near-Infrared Reflectance of Vegetation",       "vegetation", ["N", "R"],
    lambda b: ((b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS)) * b["nir"])

_register("FCVI",   "Fluorescence Correction Vegetation Index",      "vegetation", ["N", "R", "G", "B"],
    lambda b: b["nir"] - ((b["red"] + b["green"] + b["blue"]) / 3.0))

_register("AVI",    "Advanced Vegetation Index",                     "vegetation", ["N", "R"],
    lambda b: (b["nir"] * (1.0 - b["red"]) * (b["nir"] - b["red"])) ** (1 / 3 + _EPS))

_register("BNDVI",  "Blue Normalized Difference Vegetation Index",   "vegetation", ["N", "B"],
    lambda b: (b["nir"] - b["blue"]) / (b["nir"] + b["blue"] + _EPS))

_register("GBNDVI", "Green-Blue NDVI",                               "vegetation", ["N", "G", "B"],
    lambda b: (b["nir"] - (b["green"] + b["blue"])) / (b["nir"] + b["green"] + b["blue"] + _EPS))

_register("GRNDVI", "Green-Red NDVI",                                "vegetation", ["N", "G", "R"],
    lambda b: (b["nir"] - (b["green"] + b["red"])) / (b["nir"] + b["green"] + b["red"] + _EPS))

_register("GARI",   "Green Atmospherically Resistant VI",            "vegetation", ["N", "G", "B", "R"],
    lambda b: (b["nir"] - (b["green"] - (b["blue"] - b["red"]))) / (b["nir"] - (b["green"] + (b["blue"] - b["red"])) + _EPS))

_register("NDYI",   "Normalized Difference Yellowness Index",        "vegetation", ["G", "B"],
    lambda b: (b["green"] - b["blue"]) / (b["green"] + b["blue"] + _EPS))

_register("DSI",    "Drought Stress Index",                          "vegetation", ["S1", "N"],
    lambda b: b["swir1"] / (b["nir"] + _EPS))

# Red-edge vegetation indices (Sentinel-2 specific)
_register("NDRE",   "Normalized Difference Red Edge",                "vegetation", ["N", "RE1"],
    lambda b: (b["nir"] - b["red_edge_1"]) / (b["nir"] + b["red_edge_1"] + _EPS))

_register("NDREI",  "Normalized Difference Red-Edge Index",          "vegetation", ["N", "RE1"],
    lambda b: (b["nir"] - b["red_edge_1"]) / (b["nir"] + b["red_edge_1"] + _EPS))

_register("CIRE",   "Chlorophyll Index Red Edge",                    "vegetation", ["N", "RE1"],
    lambda b: (b["nir"] / (b["red_edge_1"] + _EPS)) - 1)

_register("CIG",    "Chlorophyll Index Green",                       "vegetation", ["N", "G"],
    lambda b: (b["nir"] / (b["green"] + _EPS)) - 1)

_register("ARI",    "Anthocyanin Reflectance Index",                 "vegetation", ["G", "RE1"],
    lambda b: (1 / (b["green"] + _EPS)) - (1 / (b["red_edge_1"] + _EPS)))

_register("ARI2",   "Anthocyanin Reflectance Index 2",               "vegetation", ["N", "G", "RE1"],
    lambda b: b["nir"] * ((1 / (b["green"] + _EPS)) - (1 / (b["red_edge_1"] + _EPS))))

_register("NDVI705","Normalized Difference Vegetation Index (705/750)", "vegetation", ["RE2", "RE1"],
    lambda b: (b["red_edge_2"] - b["red_edge_1"]) / (b["red_edge_2"] + b["red_edge_1"] + _EPS))

_register("ND705",  "Normalized Difference (705 and 750 nm)",        "vegetation", ["RE2", "RE1"],
    lambda b: (b["red_edge_2"] - b["red_edge_1"]) / (b["red_edge_2"] + b["red_edge_1"] + _EPS))

_register("RENDVI", "Red Edge Normalized Difference VI",             "vegetation", ["RE2", "RE1"],
    lambda b: (b["red_edge_2"] - b["red_edge_1"]) / (b["red_edge_2"] + b["red_edge_1"] + _EPS))

_register("MSR705", "Modified Simple Ratio (705/750 nm)",            "vegetation", ["RE2", "RE1"],
    lambda b: (b["red_edge_2"] / (b["red_edge_1"] + _EPS) - 1) / ((b["red_edge_2"] / (b["red_edge_1"] + _EPS) + 1) ** 0.5 + _EPS))

_register("GM1",    "Gitelson and Merzlyak Index 1",                 "vegetation", ["RE2", "G"],
    lambda b: b["red_edge_2"] / (b["green"] + _EPS))

_register("GM2",    "Gitelson and Merzlyak Index 2",                 "vegetation", ["RE2", "RE1"],
    lambda b: b["red_edge_2"] / (b["red_edge_1"] + _EPS))

_register("IRECI",  "Inverted Red-Edge Chlorophyll Index",           "vegetation", ["RE3", "R", "RE1", "RE2"],
    lambda b: (b["red_edge_3"] - b["red"]) / (b["red_edge_1"] / (b["red_edge_2"] + _EPS)))

_register("MCARI",  "Modified Chlorophyll Absorption in Reflectance", "vegetation", ["RE1", "R", "G"],
    lambda b: ((b["red_edge_1"] - b["red"]) - 0.2 * (b["red_edge_1"] - b["green"])) * (b["red_edge_1"] / (b["red"] + _EPS)))

_register("TCI",    "Triangular Chlorophyll Index",                  "vegetation", ["RE1", "G", "R"],
    lambda b: 1.2 * (b["red_edge_1"] - b["green"]) - 1.5 * (b["red"] - b["green"]) * (b["red_edge_1"] / (b["red"] + _EPS)) ** 0.5)

_register("REDSI",  "Red-Edge Disease Stress Index",                 "vegetation", ["RE3", "R", "RE1"],
    lambda b: ((705.0 - 665.0) * (b["red_edge_3"] - b["red"]) - (783.0 - 665.0) * (b["red_edge_1"] - b["red"])) / (2.0 * b["red"] + _EPS))

_register("TRRVI",  "Transformed Red Range Vegetation Index",        "vegetation", ["RE2", "R", "N"],
    lambda b: ((b["red_edge_2"] - b["red"]) / (b["red_edge_2"] + b["red"] + _EPS)) / (((b["nir"] - b["red"]) / (b["nir"] + b["red"] + _EPS)) + 1.0))

_register("S2REP",  "Sentinel-2 Red-Edge Position",                  "vegetation", ["RE3", "R", "RE1", "RE2"],
    lambda b: 705.0 + 35.0 * ((((b["red_edge_3"] + b["red"]) / 2.0) - b["red_edge_1"]) / (b["red_edge_2"] - b["red_edge_1"] + _EPS)))

_register("SeLI",   "Sentinel-2 LAI Green Index",                   "vegetation", ["N2", "RE1"],
    lambda b: (b["nir_2"] - b["red_edge_1"]) / (b["nir_2"] + b["red_edge_1"] + _EPS))

_register("SR3",    "Simple Ratio (860, 550 and 708 nm)",            "vegetation", ["N2", "G", "RE1"],
    lambda b: b["nir_2"] / (b["green"] * b["red_edge_1"] + _EPS))

_register("TTVI",   "Transformed Triangular Vegetation Index",       "vegetation", ["RE3", "RE2", "N2"],
    lambda b: 0.5 * ((865.0 - 740.0) * (b["red_edge_3"] - b["red_edge_2"]) - (783.0 - 740) * (b["nir_2"] - b["red_edge_2"])))

_register("PSRI",   "Plant Senescing Reflectance Index",             "vegetation", ["R", "B", "RE2"],
    lambda b: (b["red"] - b["blue"]) / (b["red_edge_2"] + _EPS))

_register("RVI",    "Ratio Vegetation Index (RE2/R)",                "vegetation", ["RE2", "R"],
    lambda b: b["red_edge_2"] / (b["red"] + _EPS))

_register("VARI700","Visible Atmospherically Resistant Index 700 nm","vegetation", ["RE1", "R", "B"],
    lambda b: (b["red_edge_1"] - 1.7 * b["red"] + 0.7 * b["blue"]) / (b["red_edge_1"] + 1.3 * b["red"] - 1.3 * b["blue"] + _EPS))

_register("NRFIg",  "Normalized Rapeseed Flowering Index Green",     "vegetation", ["G", "S2"],
    lambda b: (b["green"] - b["swir2"]) / (b["green"] + b["swir2"] + _EPS))

_register("NRFIr",  "Normalized Rapeseed Flowering Index Red",       "vegetation", ["R", "S2"],
    lambda b: (b["red"] - b["swir2"]) / (b["red"] + b["swir2"] + _EPS))

_register("MNDVI",  "Modified NDVI",                                 "vegetation", ["N", "S2"],
    lambda b: (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"] + _EPS))

_register("NDPI",   "Normalized Difference Phenology Index (alpha=0.74)", "vegetation", ["N", "R", "S1"],
    lambda b: (b["nir"] - (0.74 * b["red"] + 0.26 * b["swir1"])) / (b["nir"] + (0.74 * b["red"] + 0.26 * b["swir1"]) + _EPS))

# ── Water ────────────────────────────────────────────────────────────────────

_register("NDWI",   "Normalized Difference Water Index",             "water", ["G", "N"],
    lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS))

_register("MNDWI",  "Modified Normalized Difference Water Index",    "water", ["G", "S1"],
    lambda b: (b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + _EPS))

_register("NDMI",   "Normalized Difference Moisture Index",          "water", ["N", "S1"],
    lambda b: (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + _EPS))

_register("LSWI",   "Land Surface Water Index",                      "water", ["N", "S1"],
    lambda b: (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + _EPS))

_register("AWEInsh","Automated Water Extraction Index (no shadow)",  "water", ["G", "S1", "N", "S2"],
    lambda b: 4.0 * (b["green"] - b["swir1"]) - 0.25 * b["nir"] + 2.75 * b["swir2"])

_register("AWEIsh", "Automated Water Extraction Index (shadow)",     "water", ["B", "G", "N", "S1", "S2"],
    lambda b: b["blue"] + 2.5 * b["green"] - 1.5 * (b["nir"] + b["swir1"]) - 0.25 * b["swir2"])

_register("SWM",    "Sentinel Water Mask",                           "water", ["B", "G", "N", "S1"],
    lambda b: (b["blue"] + b["green"]) / (b["nir"] + b["swir1"] + _EPS))

_register("MLSWI26","Modified LSWI (bands 2 and 6)",                 "water", ["N", "S1"],
    lambda b: (1.0 - b["nir"] - b["swir1"]) / (1.0 - b["nir"] + b["swir1"] + _EPS))

_register("MLSWI27","Modified LSWI (bands 2 and 7)",                 "water", ["N", "S2"],
    lambda b: (1.0 - b["nir"] - b["swir2"]) / (1.0 - b["nir"] + b["swir2"] + _EPS))

_register("WI1",    "Water Index 1",                                 "water", ["G", "S2"],
    lambda b: (b["green"] - b["swir2"]) / (b["green"] + b["swir2"] + _EPS))

_register("WI2",    "Water Index 2",                                 "water", ["B", "S2"],
    lambda b: (b["blue"] - b["swir2"]) / (b["blue"] + b["swir2"] + _EPS))

_register("WRI",    "Water Ratio Index",                             "water", ["G", "R", "N", "S1"],
    lambda b: (b["green"] + b["red"]) / (b["nir"] + b["swir1"] + _EPS))

_register("NDPonI", "Normalized Difference Pond Index",               "water", ["S1", "G"],
    lambda b: (b["swir1"] - b["green"]) / (b["swir1"] + b["green"] + _EPS))

_register("NDTI",   "Normalized Difference Turbidity Index",          "water", ["R", "G"],
    lambda b: (b["red"] - b["green"]) / (b["red"] + b["green"] + _EPS))

_register("MuWIR",  "Revised Multi-Spectral Water Index",             "water", ["B", "G", "N", "S2", "S1"],
    lambda b: (
        -4.0 * ((b["blue"] - b["green"]) / (b["blue"] + b["green"] + _EPS))
        + 2.0 * ((b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS))
        + 2.0 * ((b["green"] - b["swir2"]) / (b["green"] + b["swir2"] + _EPS))
        - ((b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + _EPS))
    ))

_register("S2WI",   "Sentinel-2 Water Index",                        "water", ["RE1", "S2"],
    lambda b: (b["red_edge_1"] - b["swir2"]) / (b["red_edge_1"] + b["swir2"] + _EPS))

_register("NDCI",   "Normalized Difference Chlorophyll Index",        "water", ["RE1", "R"],
    lambda b: (b["red_edge_1"] - b["red"]) / (b["red_edge_1"] + b["red"] + _EPS))

_register("PI",     "Plastic Index",                                  "water", ["N", "R"],
    lambda b: b["nir"] / (b["nir"] + b["red"] + _EPS))

_register("RNDVI",  "Reversed NDVI",                                  "water", ["R", "N"],
    lambda b: (b["red"] - b["nir"]) / (b["red"] + b["nir"] + _EPS))

_register("OSI",    "Oil Spill Index",                                "water", ["G", "R", "B"],
    lambda b: (b["green"] + b["red"]) / (b["blue"] + _EPS))

# ── Soil ─────────────────────────────────────────────────────────────────────

_register("BSI",    "Bare Soil Index",                                "soil", ["S1", "R", "N", "B"],
    lambda b: ((b["swir1"] + b["red"]) - (b["nir"] + b["blue"])) / ((b["swir1"] + b["red"]) + (b["nir"] + b["blue"]) + _EPS))

_register("MBI",    "Modified Bare Soil Index",                       "soil", ["S1", "S2", "N"],
    lambda b: ((b["swir1"] - b["swir2"] - b["nir"]) / (b["swir1"] + b["swir2"] + b["nir"] + _EPS)) + 0.5)

_register("EMBI",   "Enhanced Modified Bare Soil Index",              "soil", ["S1", "S2", "N", "G"],
    lambda b: (
        ((((b["swir1"] - b["swir2"] - b["nir"]) / (b["swir1"] + b["swir2"] + b["nir"] + _EPS)) + 0.5)
         - ((b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + _EPS)) - 0.5)
        /
        ((((b["swir1"] - b["swir2"] - b["nir"]) / (b["swir1"] + b["swir2"] + b["nir"] + _EPS)) + 0.5)
         + ((b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + _EPS)) + 1.5)
    ))

_register("NSDS",   "Normalized Shortwave Infrared Difference Soil-Moisture", "soil", ["S1", "S2"],
    lambda b: (b["swir1"] - b["swir2"]) / (b["swir1"] + b["swir2"] + _EPS))

_register("NSDSI1", "Normalized SWIR Difference Bare Soil Index 1",  "soil", ["S1", "S2"],
    lambda b: (b["swir1"] - b["swir2"]) / (b["swir1"] + _EPS))

_register("NSDSI2", "Normalized SWIR Difference Bare Soil Index 2",  "soil", ["S1", "S2"],
    lambda b: (b["swir1"] - b["swir2"]) / (b["swir2"] + _EPS))

_register("NSDSI3", "Normalized SWIR Difference Bare Soil Index 3",  "soil", ["S1", "S2"],
    lambda b: (b["swir1"] - b["swir2"]) / (b["swir1"] + b["swir2"] + _EPS))

_register("NDSoI",  "Normalized Difference Soil Index",               "soil", ["S2", "G"],
    lambda b: (b["swir2"] - b["green"]) / (b["swir2"] + b["green"] + _EPS))

_register("BITM",   "Landsat TM-based Brightness Index",              "soil", ["B", "G", "R"],
    lambda b: (((b["blue"] ** 2) + (b["green"] ** 2) + (b["red"] ** 2)) / 3.0) ** 0.5)

_register("BIXS",   "SPOT HRV XS-based Brightness Index",             "soil", ["G", "R"],
    lambda b: (((b["green"] ** 2) + (b["red"] ** 2)) / 2.0) ** 0.5)

_register("RI4XS",  "SPOT HRV XS-based Redness Index 4",              "soil", ["R", "G"],
    lambda b: (b["red"] ** 2.0) / (b["green"] ** 4.0 + _EPS))

# ── Burn ─────────────────────────────────────────────────────────────────────

_register("NBR",    "Normalized Burn Ratio",                          "burn", ["N", "S2"],
    lambda b: (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"] + _EPS))

_register("NBR2",   "Normalized Burn Ratio 2",                        "burn", ["S1", "S2"],
    lambda b: (b["swir1"] - b["swir2"]) / (b["swir1"] + b["swir2"] + _EPS))

_register("NBRSWIR","Normalized Burn Ratio SWIR",                     "burn", ["S2", "S1"],
    lambda b: (b["swir2"] - b["swir1"] - 0.02) / (b["swir2"] + b["swir1"] + 0.1))

_register("BAI",    "Burned Area Index",                              "burn", ["R", "N"],
    lambda b: 1.0 / ((0.1 - b["red"]) ** 2.0 + (0.06 - b["nir"]) ** 2.0 + _EPS))

_register("BAIM",   "Burned Area Index adapted to MODIS",             "burn", ["N", "S2"],
    lambda b: 1.0 / ((0.05 - b["nir"]) ** 2.0 + (0.2 - b["swir2"]) ** 2.0 + _EPS))

_register("CSI",    "Char Soil Index",                                "burn", ["N", "S2"],
    lambda b: b["nir"] / (b["swir2"] + _EPS))

_register("MIRBI",  "Mid-Infrared Burn Index",                        "burn", ["S2", "S1"],
    lambda b: 10.0 * b["swir2"] - 9.8 * b["swir1"] + 2.0)

_register("NDSWIR", "Normalized Difference SWIR",                     "burn", ["N", "S1"],
    lambda b: (b["nir"] - b["swir1"]) / (b["nir"] + b["swir1"] + _EPS))

_register("BAIS2",  "Burned Area Index for Sentinel-2",               "burn", ["RE2", "RE3", "N2", "R", "S2"],
    lambda b: (
        (1.0 - ((b["red_edge_2"] * b["red_edge_3"] * b["nir_2"]) / (b["red"] + _EPS)) ** 0.5)
        * (((b["swir2"] - b["nir_2"]) / (b["swir2"] + b["nir_2"] + _EPS) ** 0.5) + 1.0)
    ))

# ── Urban ────────────────────────────────────────────────────────────────────

_register("NDBI",   "Normalized Difference Built-Up Index",           "urban", ["S1", "N"],
    lambda b: (b["swir1"] - b["nir"]) / (b["swir1"] + b["nir"] + _EPS))

_register("UI",     "Urban Index",                                    "urban", ["S2", "N"],
    lambda b: (b["swir2"] - b["nir"]) / (b["swir2"] + b["nir"] + _EPS))

_register("BLFEI",  "Built-Up Land Features Extraction Index",        "urban", ["G", "R", "S2", "S1"],
    lambda b: (((b["green"] + b["red"] + b["swir2"]) / 3.0) - b["swir1"]) / (((b["green"] + b["red"] + b["swir2"]) / 3.0) + b["swir1"] + _EPS))

_register("BRBA",   "Band Ratio for Built-up Area",                   "urban", ["R", "S1"],
    lambda b: b["red"] / (b["swir1"] + _EPS))

_register("NBAI",   "Normalized Built-up Area Index",                 "urban", ["S2", "S1", "G"],
    lambda b: (b["swir2"] - b["swir1"] / (b["green"] + _EPS)) / (b["swir2"] + b["swir1"] / (b["green"] + _EPS) + _EPS))

_register("PISI",   "Perpendicular Impervious Surface Index",         "urban", ["B", "N"],
    lambda b: 0.8192 * b["blue"] - 0.5735 * b["nir"] + 0.0750)

_register("VgNIRBI","Visible Green-Based Built-Up Index",             "urban", ["G", "N"],
    lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS))

_register("VrNIRBI","Visible Red-Based Built-Up Index",               "urban", ["R", "N"],
    lambda b: (b["red"] - b["nir"]) / (b["red"] + b["nir"] + _EPS))

# ── Snow / Ice ───────────────────────────────────────────────────────────────

_register("NDSI",   "Normalized Difference Snow Index",               "snow", ["G", "S1"],
    lambda b: (b["green"] - b["swir1"]) / (b["green"] + b["swir1"] + _EPS))

_register("NDSII",  "Normalized Difference Snow Ice Index",            "snow", ["G", "N"],
    lambda b: (b["green"] - b["nir"]) / (b["green"] + b["nir"] + _EPS))

_register("NDSaII", "Normalized Difference Snow and Ice Index",        "snow", ["R", "S1"],
    lambda b: (b["red"] - b["swir1"]) / (b["red"] + b["swir1"] + _EPS))

_register("SWI",    "Snow Water Index",                               "snow", ["G", "S1", "S2"],
    lambda b: (b["green"] - (b["swir1"] + b["swir2"])) / (b["green"] + (b["swir1"] + b["swir2"]) + _EPS))

_register("NDGlaI", "Normalized Difference Glacier Index",            "snow", ["G", "R"],
    lambda b: (b["green"] - b["red"]) / (b["green"] + b["red"] + _EPS))

# ── Radar (Sentinel-1 only) ──────────────────────────────────────────────────

_register("DPDD",   "Dual-Pol Diagonal Distance",                     "radar", ["VV", "VH"],
    lambda b: (b["sar_vv"] + b["sar_vh"]) / (2.0 ** 0.5))

_register("NDPolI", "Normalized Difference Polarization Index",       "radar", ["VV", "VH"],
    lambda b: (b["sar_vv"] - b["sar_vh"]) / (b["sar_vv"] + b["sar_vh"] + _EPS))

_register("VVVHD",  "VV-VH Difference",                               "radar", ["VV", "VH"],
    lambda b: b["sar_vv"] - b["sar_vh"])

_register("VVVHR",  "VV-VH Ratio",                                    "radar", ["VV", "VH"],
    lambda b: b["sar_vv"] / (b["sar_vh"] + _EPS))

_register("DpRVIVV","Dual-Polarized Radar Vegetation Index (VV)",     "radar", ["VV", "VH"],
    lambda b: (4.0 * b["sar_vh"]) / (b["sar_vv"] + b["sar_vh"] + _EPS))

_register("RFDI",   "Radar Forest Degradation Index",                 "radar", ["VH", "VV"],
    lambda b: (b["sar_vh"] - b["sar_vv"]) / (b["sar_vh"] + b["sar_vv"] + _EPS))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_indices_by_domain() -> dict[str, list[dict]]:
    """Return the catalogue grouped by domain, serialisable to JSON.

    Each entry is::

        {"id": "NDVI", "long_name": "...", "bands": ["nir", "red"]}
    """
    result: dict[str, list[dict]] = {}
    for short_name, spec in SPECTRAL_INDEX_CATALOGUE.items():
        domain = spec["domain"]
        result.setdefault(domain, []).append({
            "id": short_name,
            "long_name": spec["long_name"],
            "bands": spec["bands"],
        })
    return result


def get_computable_indices(available_bands: list[str]) -> list[str]:
    """Return the short names of all indices computable from *available_bands*."""
    band_set = set(available_bands)
    # Normalise legacy aliases
    for old, new in LEGACY_BAND_ALIASES.items():
        if old in band_set:
            band_set.add(new)
    return [
        name
        for name, spec in SPECTRAL_INDEX_CATALOGUE.items()
        if set(spec["bands"]).issubset(band_set)
    ]


def compute_index(name: str, band_arrays: dict[str, "np.ndarray"]) -> "np.ndarray | None":
    """Compute a single spectral index.

    Parameters
    ----------
    name:         Upper-case index short name, e.g. ``"NDVI"``.
    band_arrays:  Dict of internal band name → 1D float32 pixel array.

    Returns None if the index is not in the catalogue or required bands
    are missing.
    """
    import numpy as np
    spec = SPECTRAL_INDEX_CATALOGUE.get(name.upper())
    if spec is None:
        return None
    if not set(spec["bands"]).issubset(band_arrays):
        return None
    try:
        result = spec["formula"](band_arrays)
        return np.asarray(result, dtype=np.float32)
    except Exception:
        return None
