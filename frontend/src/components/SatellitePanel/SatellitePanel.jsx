import React, { useState } from 'react'
import './SatellitePanel.css'
import { useAppContext } from '../../context/AppContext'
import { searchStacScenes } from '../../services/api'
import { getCombosForCollection, getDefaultComboId, MIN_SAT_ZOOM } from './bandCombos'

const DEFAULT_MAX_CLOUD = 20
const DEFAULT_COLLECTION = 'sentinel-2-l2a'

const SENSOR_LABEL = {
  'sentinel-2-l2a': 'Sentinel-2 L2A',
  'sentinel-1-rtc': 'Sentinel-1 RTC',
}

export default function SatellitePanel() {
  const {
    mapZoom, mapBounds,
    activeScene, setActiveScene,
    activeBandCombo, setActiveBandCombo,
    satelliteOpacity, setSatelliteOpacity,
    activeProject,
  } = useAppContext()

  const [open, setOpen] = useState(false)
  const [scenes, setScenes] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [maxCloud, setMaxCloud] = useState(DEFAULT_MAX_CLOUD)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [collection, setCollection] = useState(DEFAULT_COLLECTION)

  // Sensor types enabled for the active project (may be undefined for old projects)
  const projectSensors = activeProject?.sensors?.map(s => s.sensor_type) ?? [DEFAULT_COLLECTION]
  // Show toggle only if the project has more than one sensor type
  const showSensorToggle = projectSensors.length > 1

  // Sentinel-1 has no cloud cover metadata — hide the control when searching S1
  const isS1 = collection === 'sentinel-1-rtc'

  const canSearch = mapZoom >= MIN_SAT_ZOOM && mapBounds !== null

  const handleSearch = async () => {
    if (!canSearch) return
    setLoading(true)
    setError(null)
    try {
      const { minLon, minLat, maxLon, maxLat } = mapBounds
      const results = await searchStacScenes({
        bbox: `${minLon},${minLat},${maxLon},${maxLat}`,
        collection,
        max_cloud: isS1 ? undefined : maxCloud,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        limit: 12,
      })
      setScenes(results)
      if (results.length === 0) setError('No scenes found. Try relaxing the date range.')
    } catch (e) {
      setError(e.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const switchCollection = (col) => {
    setCollection(col)
    setScenes([])
    setError(null)
    // Reset to the first valid combo for the new collection
    setActiveBandCombo(getDefaultComboId(col))
  }

  const selectScene = (scene) => {
    if (activeScene?.id === scene.id) {
      setActiveScene(null)
    } else {
      setActiveScene(scene)
      // Reset to first valid combo for this scene's collection
      setActiveBandCombo(getDefaultComboId(scene.collection))
    }
  }

  const clearSatellite = () => {
    setActiveScene(null)
    setScenes([])
  }

  return (
    <div className="sat-panel">
      {/* Section header */}
      <button className="sat-panel-header" onClick={() => setOpen(o => !o)}>
        <span className="sat-panel-icon">🛰</span>
        <span className="sat-panel-title">Satellite Imagery</span>
        {activeScene && <span className="sat-panel-badge">1 active</span>}
        <span className={`sat-panel-chevron${open ? ' open' : ''}`}>›</span>
      </button>

      {activeScene && (
        <div className="sat-active-scene-bar">
          <span className="sat-active-scene-dot" />
          <div className="sat-active-scene-details">
            <span className="sat-active-scene-name" title={activeScene.id}>{activeScene.id}</span>
            <span className="sat-active-scene-date">{activeScene.date}</span>
          </div>
          {activeScene.cloud_pct >= 0 && (
            <span className="sat-active-scene-cloud">☁ {activeScene.cloud_pct}%</span>
          )}
        </div>
      )}

      {open && (
        <div className="sat-panel-body">
          {/* Zoom gate */}
          {!canSearch && (
            <div className="sat-zoom-hint">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              Zoom to level {MIN_SAT_ZOOM}+ to enable satellite search
              <br /><span className="sat-zoom-current">Current zoom: {mapZoom}</span>
            </div>
          )}

          {canSearch && (
            <>
              {/* Sensor toggle (only shown when project has multiple sensors) */}
              {showSensorToggle && (
                <div className="sat-sensor-toggle">
                  {projectSensors.map(col => (
                    <button
                      key={col}
                      className={`sat-sensor-btn${collection === col ? ' active' : ''}`}
                      onClick={() => switchCollection(col)}
                    >
                      {SENSOR_LABEL[col] ?? col}
                    </button>
                  ))}
                </div>
              )}

              {/* Search controls */}
              <div className="sat-controls">
                {!isS1 && (
                  <div className="sat-row">
                    <label>Cloud cover ≤</label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={maxCloud}
                      onChange={e => setMaxCloud(Number(e.target.value))}
                      className="sat-input sat-input--sm"
                    />
                    <span>%</span>
                  </div>
                )}
                <div className="sat-row">
                  <label>From</label>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={e => setDateFrom(e.target.value)}
                    className="sat-input"
                  />
                </div>
                <div className="sat-row">
                  <label>To</label>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={e => setDateTo(e.target.value)}
                    className="sat-input"
                  />
                </div>
                <button className="sat-search-btn" onClick={handleSearch} disabled={loading}>
                  {loading ? 'Searching…' : 'Search Scenes'}
                </button>
              </div>

              {error && <div className="sat-error">{error}</div>}

              {/* Scene list */}
              {scenes.length > 0 && (
                <div className="sat-scenes">
                  {scenes.map(scene => (
                    <button
                      key={scene.id}
                      className={`sat-scene-card${activeScene?.id === scene.id ? ' active' : ''}`}
                      onClick={() => selectScene(scene)}
                      title={scene.id}
                    >
                      {scene.thumbnail
                        ? <img src={scene.thumbnail} alt="" className="sat-scene-thumb" />
                        : <div className="sat-scene-thumb sat-scene-thumb--placeholder">
                            {isS1 ? 'S1' : 'S2'}
                          </div>
                      }
                      <div className="sat-scene-meta">
                        <span className="sat-scene-date">{scene.date}</span>
                        {!isS1 && (
                          <span className={`sat-scene-cloud${scene.cloud_pct > 30 ? ' high' : ''}`}>
                            ☁ {scene.cloud_pct >= 0 ? `${scene.cloud_pct}%` : '—'}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {/* Active scene controls */}
              {activeScene && (
                <div className="sat-active-controls">
                  <div className="sat-divider" />
                  <div className="sat-field">
                    <label>Band Combination</label>
                    <select
                      value={activeBandCombo}
                      onChange={e => setActiveBandCombo(e.target.value)}
                      className="sat-select"
                    >
                      {getCombosForCollection(activeScene.collection).map(c => (
                        <option key={c.id} value={c.id}>{c.label}</option>
                      ))}
                    </select>
                  </div>
                  <div className="sat-field">
                    <label>Opacity — {Math.round(satelliteOpacity * 100)}%</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={satelliteOpacity}
                      onChange={e => setSatelliteOpacity(Number(e.target.value))}
                      className="sat-range"
                    />
                  </div>
                  <button className="sat-clear-btn" onClick={clearSatellite}>
                    ✕ Remove Layer
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
