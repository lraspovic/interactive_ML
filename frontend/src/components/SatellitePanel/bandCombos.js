/**
 * Band combination presets per sensor collection.
 * assets: Planetary Computer asset names
 * rescale: "min,max" used to stretch pixel values to uint8
 * color_formula: optional rio_tiler color enhancement
 */

// ── Sentinel-2 L2A ───────────────────────────────────────────────────────────
export const S2_BAND_COMBOS = [
  {
    id: 'true_color',
    label: 'True Color',
    assets: ['B04', 'B03', 'B02'],
    rescale: '0,3000',
    color_formula: null,
  },
  {
    id: 'false_color',
    label: 'False Color (NIR)',
    assets: ['B08', 'B04', 'B03'],
    rescale: '0,4000',
    color_formula: null,
  },
  {
    id: 'swir',
    label: 'SWIR Composite',
    assets: ['B12', 'B8A', 'B04'],
    rescale: '0,4000',
    color_formula: null,
  },
  {
    id: 'agriculture',
    label: 'Agriculture',
    assets: ['B11', 'B08', 'B02'],
    rescale: '0,4000',
    color_formula: null,
  },
  {
    id: 'geology',
    label: 'Geology',
    assets: ['B12', 'B11', 'B02'],
    rescale: '0,4000',
    color_formula: null,
  },
]

// ── Sentinel-1 RTC ───────────────────────────────────────────────────────────
// S1 RTC backscatter is stored in linear power scale (float32).
// Typical display range: 0–0.3 (surface/veg = 0.05–0.15, urban = up to 0.5+).
export const S1_BAND_COMBOS = [
  {
    id: 's1_vv_grey',
    label: 'VV (greyscale)',
    assets: ['vv'],
    rescale: '0,0.3',
    color_formula: null,
  },
  {
    id: 's1_vh_grey',
    label: 'VH (greyscale)',
    assets: ['vh'],
    rescale: '0,0.1',
    color_formula: null,
  },
  {
    id: 's1_dual_pol',
    label: 'Dual-pol RGB (VV/VH/VV)',
    assets: ['vv', 'vh', 'vv'],
    rescale: '0,0.3',
    color_formula: null,
  },
]

// ── Lookup by collection ─────────────────────────────────────────────────────
const COMBOS_BY_COLLECTION = {
  'sentinel-2-l2a': S2_BAND_COMBOS,
  'sentinel-1-rtc': S1_BAND_COMBOS,
}

/** Return the combo list for a given collection (falls back to S2). */
export function getCombosForCollection(collection) {
  return COMBOS_BY_COLLECTION[collection] ?? S2_BAND_COMBOS
}

/** Return the default (first) combo id for a collection. */
export function getDefaultComboId(collection) {
  return getCombosForCollection(collection)[0].id
}

// Backward-compat alias (S2 still used as default)
export const BAND_COMBOS = S2_BAND_COMBOS

// The backend URL (absolute) used to build tile requests.
const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Build a Leaflet TileLayer URL template that points at the backend
 * /imagery/stac/tiles/{z}/{x}/{y} endpoint.
 * Automatically selects from the correct combo list for scene.collection.
 */
export function buildSatTileUrl(scene, comboId) {
  const combos = getCombosForCollection(scene.collection)
  const combo = combos.find(c => c.id === comboId) ?? combos[0]
  const params = new URLSearchParams()
  params.set('item_id', scene.id)
  params.set('collection', scene.collection)
  combo.assets.forEach(a => params.append('assets', a))
  params.set('rescale', combo.rescale)
  if (combo.color_formula) params.set('color_formula', combo.color_formula)
  // Use absolute backend URL — Leaflet replaces {z}/{x}/{y} in the full string
  return `${BACKEND_URL}/imagery/stac/tiles/{z}/{x}/{y}?${params.toString()}`
}

/** Minimum zoom level before satellite search is permitted. */
export const MIN_SAT_ZOOM = 14
