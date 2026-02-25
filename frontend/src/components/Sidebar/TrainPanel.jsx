import { useState, useEffect, useRef, useCallback } from 'react'
import './TrainPanel.css'
import { useAppContext } from '../../context/AppContext'
import { triggerTrain, getTrainStatus } from '../../services/api'

export default function TrainPanel() {
  const { activeProject, activeScene, samples, showToast } = useAppContext()
  const [open, setOpen] = useState(true)
  const [job, setJob] = useState(null)   // null | { status, progress, metrics, error }
  const pollRef = useRef(null)

  // Count unique classes that actually have samples
  const classesWithSamples = new Set(
    samples.features.map(f => f.properties.class_id).filter(Boolean)
  )

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const pollStatus = useCallback(async () => {
    if (!activeProject?.id) return
    try {
      const data = await getTrainStatus(activeProject.id)
      setJob(data)
      if (data.status === 'done' || data.status === 'failed' || data.status === 'idle') {
        stopPolling()
        if (data.status === 'done') {
          showToast(`Training complete — test acc ${data.metrics?.test_accuracy != null ? (data.metrics.test_accuracy * 100).toFixed(1) : (data.metrics?.train_accuracy * 100).toFixed(1)}%`)
        } else if (data.status === 'failed') {
          showToast(data.error || 'Training failed', 'error')
        }
      }
    } catch {
      stopPolling()
    }
  }, [activeProject?.id, stopPolling, showToast])

  // Start polling whenever a job transitions to "running"
  useEffect(() => {
    if (job?.status === 'running' && !pollRef.current) {
      pollRef.current = setInterval(pollStatus, 2000)
    }
    return () => {}
  }, [job?.status, pollStatus])

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [stopPolling])

  // Fetch status on mount so we pick up any in-progress job from before
  useEffect(() => {
    if (activeProject?.id) pollStatus()
  }, [activeProject?.id, pollStatus])

  async function handleTrain() {
    if (!activeProject || !activeScene) return
    try {
      await triggerTrain({
        project_id: activeProject.id,
        item_id: activeScene.id,
        collection: activeScene.collection,
      })
      setJob({ status: 'running', progress: 0, metrics: null, error: null })
      pollRef.current = setInterval(pollStatus, 2000)
    } catch (e) {
      showToast(e.message || 'Failed to start training', 'error')
    }
  }

  const isRunning = job?.status === 'running'
  const canTrain = activeProject && activeScene && classesWithSamples.size >= 2 && !isRunning

  return (
    <div className="train-panel">
      <button className="train-panel-header" onClick={() => setOpen(o => !o)}>
        <span className="train-panel-icon">🤖</span>
        <span className="train-panel-title">Train Model</span>
        {job?.status === 'done' && (
          <span className="train-panel-badge">
            {(job.metrics.accuracy * 100).toFixed(1)}%
          </span>
        )}
        <span className="train-panel-chevron">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="train-panel-body">

          {/* Requirement hints */}
          {!activeScene && (
            <p className="train-hint">
              🛰️ Select a satellite scene above to use as the imagery source.
            </p>
          )}
          {activeScene && classesWithSamples.size < 2 && (
            <p className="train-hint">
              ✏️ Draw polygons for at least 2 classes on the map.
            </p>
          )}

          {/* Scene info */}
          {activeScene && (
            <div className="train-scene-info">
              <span className="train-scene-label">Scene</span>
              <span className="train-scene-value" title={activeScene.id}>
                {activeScene.date} · ☁ {activeScene.cloud_pct}%
              </span>
            </div>
          )}

          {/* Sample summary */}
          <div className="train-scene-info">
            <span className="train-scene-label">Polygons</span>
            <span className="train-scene-value">
              {samples.features.length} drawn · {classesWithSamples.size} classes
            </span>
          </div>

          {/* Progress bar */}
          {isRunning && (
            <div className="train-progress-wrap">
              <div
                className="train-progress-bar"
                style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }}
              />
              <span className="train-progress-label">
                Extracting pixels… {Math.round((job.progress ?? 0) * 100)}%
              </span>
            </div>
          )}

          {/* Metrics */}
          {job?.status === 'done' && job.metrics && (
            <div className="train-metrics">
              <div className="train-metric-row">
                <span>Train acc</span>
                <strong>{(job.metrics.train_accuracy * 100).toFixed(1)}%</strong>
              </div>
              <div className="train-metric-row">
                <span>Test acc</span>
                <strong style={{ color: job.metrics.test_accuracy != null ? 'var(--accent)' : 'var(--text-secondary)' }}>
                  {job.metrics.test_accuracy != null
                    ? `${(job.metrics.test_accuracy * 100).toFixed(1)}%`
                    : '— (≤1 poly/class)'}
                </strong>
              </div>
              <div className="train-metric-row">
                <span>Polygons</span>
                <strong>{job.metrics.n_train_polygons} train / {job.metrics.n_test_polygons} test</strong>
              </div>
              <div className="train-metric-row">
                <span>Pixels</span>
                <strong>{job.metrics.n_train?.toLocaleString()} train / {job.metrics.n_test?.toLocaleString()} test</strong>
              </div>
              <div className="train-metric-row">
                <span>Features</span>
                <strong>{job.metrics.feature_names?.join(', ') || job.metrics.n_features}</strong>
              </div>
              <div className="train-metric-row">
                <span>Model</span>
                <strong>{job.metrics.model_type?.replace('_', ' ')}</strong>
              </div>
            </div>
          )}

          {/* Error */}
          {job?.status === 'failed' && (
            <p className="train-error">{job.error}</p>
          )}

          {/* Train button */}
          <button
            className="train-btn"
            onClick={handleTrain}
            disabled={!canTrain}
            title={!activeScene ? 'Select a scene first' : !canTrain ? 'Need ≥2 classes with polygons' : 'Start training'}
          >
            {isRunning ? '⏳ Training…' : '▶ Train Model'}
          </button>
        </div>
      )}
    </div>
  )
}
