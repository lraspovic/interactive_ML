const BASE = import.meta.env.VITE_API_URL || ''

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

export async function listProjects() {
  const r = await fetch(`${BASE}/projects`)
  if (!r.ok) throw new Error('Failed to fetch projects')
  return r.json()
}

export async function getProject(id) {
  const r = await fetch(`${BASE}/projects/${id}`)
  if (!r.ok) throw new Error('Failed to fetch project')
  return r.json()
}

export async function createProject(data) {
  const r = await fetch(`${BASE}/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to create project')
  }
  return r.json()
}

// ---------------------------------------------------------------------------
// Classes (legacy global endpoint kept for compatibility)
// ---------------------------------------------------------------------------

export async function fetchClasses() {
  const r = await fetch(`${BASE}/classes`)
  if (!r.ok) throw new Error('Failed to fetch classes')
  return r.json()
}

// ---------------------------------------------------------------------------
// Training samples
// ---------------------------------------------------------------------------

export async function fetchTrainingSamples(projectId) {
  const url = projectId
    ? `${BASE}/training-samples?project_id=${projectId}`
    : `${BASE}/training-samples`
  const r = await fetch(url)
  if (!r.ok) throw new Error('Failed to fetch training samples')
  return r.json()
}

export async function createTrainingSample({ geometry, label, class_id, project_id }) {
  const r = await fetch(`${BASE}/training-samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ geometry, label, class_id, project_id }),
  })
  if (!r.ok) throw new Error('Failed to create training sample')
  return r.json()
}

export async function deleteTrainingSample(id) {
  const r = await fetch(`${BASE}/training-samples/${id}`, { method: 'DELETE' })
  if (!r.ok) throw new Error('Failed to delete training sample')
}

// ---------------------------------------------------------------------------
// Satellite imagery / STAC
// ---------------------------------------------------------------------------

/**
 * Search Planetary Computer for satellite scenes.
 * @param {object} params
 * @param {string} params.bbox          "minlon,minlat,maxlon,maxlat"
 * @param {string} [params.collection]  default "sentinel-2-l2a"
 * @param {string} [params.date_from]   ISO date
 * @param {string} [params.date_to]     ISO date
 * @param {number} [params.max_cloud]   0-100, default 20
 * @param {number} [params.limit]       default 12
 */
export async function searchStacScenes(params) {
  const query = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => v != null && query.set(k, v))
  const r = await fetch(`${BASE}/imagery/stac-search?${query}`)
  if (!r.ok) throw new Error('STAC search failed')
  return r.json()
}

// ---------------------------------------------------------------------------
// Model training
// ---------------------------------------------------------------------------

/**
 * Kick off model training for a project using a specific satellite scene.
 * @param {{ project_id: string, item_id: string, collection: string }} body
 */
export async function triggerTrain(body) {
  const r = await fetch(`${BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to start training')
  }
  return r.json()
}

/**
 * Poll training job status for a project.
 * @param {string} projectId
 */
export async function getTrainStatus(projectId) {
  const r = await fetch(`${BASE}/train/status?project_id=${projectId}`)
  if (!r.ok) throw new Error('Failed to fetch train status')
  return r.json()
}

/**
 * Fetch per-polygon feature means for a project + scene (only polygons
 * already extracted and cached in training_features).
 */
export async function getFeatureMeans(projectId, itemId, collection = 'sentinel-2-l2a') {
  const q = new URLSearchParams({ project_id: projectId, item_id: itemId, collection })
  const r = await fetch(`${BASE}/train/features?${q}`)
  if (!r.ok) throw new Error('Failed to fetch feature means')
  return r.json()
}

export function getFeatureDownloadUrl(projectId, itemId, collection = 'sentinel-2-l2a') {
  const q = new URLSearchParams({ project_id: projectId, item_id: itemId, collection })
  return `${BASE}/train/features/download?${q}`
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/**
 * Kick off a prediction job.
 * @param {{ project_id: string, item_id: string, collection: string, bbox: number[] }} body
 */
export async function triggerPredict(body) {
  const r = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to start prediction')
  }
  return r.json()
}

/**
 * Poll prediction job status for a project.
 * @param {string} projectId
 */
export async function getPredictStatus(projectId) {
  const r = await fetch(`${BASE}/predict/status?project_id=${projectId}`)
  if (!r.ok) throw new Error('Failed to fetch predict status')
  return r.json()
}

/** Returns a Leaflet TileLayer URL template for the prediction classification map. */
export function getPredictTileUrl(projectId, bust) {
  return `${BASE}/predict/tiles/{z}/{x}/{y}?project_id=${projectId}&t=${bust}`
}

/** Returns a Leaflet TileLayer URL template for the uncertainty heatmap. */
export function getUncertaintyTileUrl(projectId, bust) {
  return `${BASE}/predict/uncertainty/tiles/{z}/{x}/{y}?project_id=${projectId}&t=${bust}`
}

/** Returns the download URL for the prediction COG (uint16 class IDs, EPSG:3857). */
export function getPredictDownloadUrl(projectId) {
  return `${BASE}/predict/download/${projectId}`
}

/** Returns the download URL for the uncertainty COG (float32 normalised entropy, EPSG:3857). */
export function getUncertaintyDownloadUrl(projectId) {
  return `${BASE}/predict/uncertainty/download/${projectId}`
}

/** Fetch the full spectral index catalogue grouped by domain from the backend. */
export async function getSpectralIndices() {
  const r = await fetch(`${BASE}/imagery/spectral-indices`)
  if (!r.ok) throw new Error('Failed to fetch spectral indices')
  return r.json()
}

/** Returns the download URL for the debug GeoTIFF download (raw bands used for last prediction). */
export function getDebugTiffUrl(projectId) {
  return `${BASE}/predict/debug-tiff/${projectId}`
}
