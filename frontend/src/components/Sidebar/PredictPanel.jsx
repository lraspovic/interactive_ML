import { useState, useEffect, useRef, useCallback } from 'react'
import './PredictPanel.css'
import { useAppContext } from '../../context/AppContext'
import { triggerPredict, getPredictStatus } from '../../services/api'

export default function PredictPanel() {
  const {
    activeProject,
    activeScene,
    classes,
    mapBounds,
    prediction,
    setPrediction,
    predictionOpacity,
    setPredictionOpacity,
    uncertaintyOpacity,
    setUncertaintyOpacity,
    showToast,
  } = useAppContext()

  const [open, setOpen] = useState(true)
  const [job, setJob] = useState(null)   // null | { status, progress, bbox, timestamp, error }
  const pollRef = useRef(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const pollStatus = useCallback(async () => {
    if (!activeProject?.id) return
    try {
      const data = await getPredictStatus(activeProject.id)
      setJob(data)
      if (data.status === 'done') {
        stopPolling()
        setPrediction({ bbox: data.bbox, timestamp: data.timestamp })
        showToast('Prediction complete')
      } else if (data.status === 'failed') {
        stopPolling()
        showToast(data.error || 'Prediction failed', 'error')
      }
    } catch {
      stopPolling()
    }
  }, [activeProject?.id, stopPolling, setPrediction, showToast])

  useEffect(() => {
    if (job?.status === 'running' && !pollRef.current) {
      pollRef.current = setInterval(pollStatus, 2000)
    }
    return () => {}
  }, [job?.status, pollStatus])

  useEffect(() => () => stopPolling(), [stopPolling])

  // Restore any existing prediction state on mount
  useEffect(() => {
    if (activeProject?.id) pollStatus()
  }, [activeProject?.id, pollStatus])

  async function handlePredict() {
    if (!activeProject || !activeScene || !mapBounds) return
    const bbox = [mapBounds.minLon, mapBounds.minLat, mapBounds.maxLon, mapBounds.maxLat]
    try {
      await triggerPredict({
        project_id: activeProject.id,
        item_id: activeScene.id,
        collection: activeScene.collection,
        bbox,
      })
      setJob({ status: 'running', progress: 0, bbox, timestamp: null, error: null })
      pollRef.current = setInterval(pollStatus, 2000)
    } catch (e) {
      showToast(e.message || 'Failed to start prediction', 'error')
    }
  }

  const isRunning = job?.status === 'running'
  const hasPrediction = job?.status === 'done' || prediction != null
  const canPredict = activeProject && activeScene && mapBounds && !isRunning

  return (
    <div className="predict-panel">
      <button className="predict-panel-header" onClick={() => setOpen(o => !o)}>
        <span className="predict-panel-icon">🗺️</span>
        <span className="predict-panel-title">Prediction</span>
        {hasPrediction && <span className="predict-panel-badge">Ready</span>}
        <span className="predict-panel-chevron">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="predict-panel-body">
          {/* Requirements hints */}
          {!activeScene && (
            <p className="predict-hint">🛰️ Select a satellite scene to run prediction.</p>
          )}
          {!mapBounds && activeScene && (
            <p className="predict-hint">🗺️ Pan the map to set a viewport first.</p>
          )}

          {/* Progress bar */}
          {isRunning && (
            <div className="predict-progress-wrap">
              <div
                className="predict-progress-bar"
                style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
              />
              <span className="predict-progress-label">
                Running… {Math.round((job.progress ?? 0) * 100)}%
              </span>
            </div>
          )}

          {/* Predict button */}
          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={!canPredict}
            title={!activeScene ? 'Select a scene first' : !mapBounds ? 'Pan the map first' : 'Predict on current viewport'}
          >
            {isRunning ? '⏳ Predicting…' : '▶ Run Prediction'}
          </button>

          {/* Overlays + legend — only shown when a prediction exists */}
          {hasPrediction && (
            <>
              <div className="predict-opacity-row">
                <label>Classification</label>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={predictionOpacity}
                  onChange={e => setPredictionOpacity(+e.target.value)}
                />
                <span>{Math.round(predictionOpacity * 100)}%</span>
              </div>

              <div className="predict-opacity-row">
                <label>Uncertainty</label>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={uncertaintyOpacity}
                  onChange={e => setUncertaintyOpacity(+e.target.value)}
                />
                <span>{Math.round(uncertaintyOpacity * 100)}%</span>
              </div>

              {/* Legend */}
              {classes.length > 0 && (
                <div className="predict-legend">
                  <span className="predict-legend-title">Legend</span>
                  <ul className="predict-legend-list">
                    {classes.map(cls => (
                      <li key={cls.id} className="predict-legend-item">
                        <span className="predict-legend-swatch" style={{ background: cls.color }} />
                        <span className="predict-legend-name">{cls.name}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}

          {/* Error */}
          {job?.status === 'failed' && (
            <p className="predict-error">{job.error}</p>
          )}
        </div>
      )}
    </div>
  )
}
