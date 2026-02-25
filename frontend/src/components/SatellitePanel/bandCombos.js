/**
 * Sentinel-2 L2A band combination presets.
 * assets: Planetary Computer asset names
 * rescale: "min,max" for titiler
 * color_formula: optional titiler color enhancement
 */
export const BAND_COMBOS = [
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

// The backend URL (absolute) used to build tile requests.
const BACKEND_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Build a Leaflet TileLayer URL template that points at the backend
 * /imagery/stac/tiles/{z}/{x}/{y} endpoint.
 */
export function buildSatTileUrl(scene, comboId) {
  const combo = BAND_COMBOS.find(c => c.id === comboId) ?? BAND_COMBOS[0]
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
