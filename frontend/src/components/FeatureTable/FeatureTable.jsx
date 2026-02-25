import { useState, useEffect, useCallback } from 'react'
import './FeatureTable.css'
import { useAppContext } from '../../context/AppContext'
import { getFeatureMeans, getFeatureDownloadUrl } from '../../services/api'

export default function FeatureTable() {
  const { activeProject, activeScene } = useAppContext()
  const [open, setOpen] = useState(false)
  const [data, setData] = useState(null)   // null | { features: [], item_id }
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    if (!activeProject?.id || !activeScene?.id) {
      setData(null)
      return
    }
    setLoading(true)
    try {
      const result = await getFeatureMeans(
        activeProject.id,
        activeScene.id,
        activeScene.collection,
      )
      setData(result)
      if (result.features.length > 0) setOpen(true)
    } catch {
      // no-op — table just stays empty
    } finally {
      setLoading(false)
    }
  }, [activeProject?.id, activeScene?.id, activeScene?.collection])

  // Refresh whenever the scene changes
  useEffect(() => { refresh() }, [refresh])

  // Derive column headers from first row's means keys
  const featureNames = data?.features?.[0]
    ? Object.keys(data.features[0].means)
    : []
  const rows = data?.features ?? []
  const hasData = rows.length > 0

  return (
    <div className={`feature-table-drawer${open ? ' feature-table-drawer--open' : ''}`}>
      {/* ── Toggle handle ── */}
      <button className="feature-table-handle" onClick={() => setOpen(o => !o)}>
        <span className="feature-table-handle-icon">📊</span>
        <span className="feature-table-handle-title">
          Feature Means
          {hasData && <span className="feature-table-badge">{rows.length} polygons</span>}
          {activeScene && !hasData && !loading && (
            <span className="feature-table-hint">train first to populate</span>
          )}
          {loading && <span className="feature-table-hint">loading…</span>}
        </span>
        <span className="feature-table-handle-refresh" title="Refresh" onClick={e => { e.stopPropagation(); refresh() }}>↻</span>
        {hasData && activeScene && (
          <a
            className="feature-table-download"
            href={getFeatureDownloadUrl(activeProject.id, activeScene.id, activeScene.collection)}
            download
            title="Download as GeoPackage"
            onClick={e => e.stopPropagation()}
          >⬇ GPKG</a>
        )}
        <span className="feature-table-chevron">{open ? '▾' : '▴'}</span>
      </button>

      {/* ── Table ── */}
      {open && (
        <div className="feature-table-scroll">
          {!activeScene && (
            <p className="feature-table-empty">Select a satellite scene to view extracted features.</p>
          )}
          {activeScene && !hasData && !loading && (
            <p className="feature-table-empty">No features extracted yet for this scene. Run training to extract pixel values from your polygons.</p>
          )}
          {hasData && (
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Polygon</th>
                  <th>Class ID</th>
                  <th className="feature-table-num">Pixels</th>
                  {featureNames.map(n => (
                    <th key={n} className="feature-table-num">{n}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(row => (
                  <tr key={row.sample_id}>
                    <td className="feature-table-label">{row.label}</td>
                    <td className="feature-table-num">{row.class_id ?? '—'}</td>
                    <td className="feature-table-num">{row.n_pixels.toLocaleString()}</td>
                    {featureNames.map(n => (
                      <td key={n} className="feature-table-num feature-table-value">
                        {row.means[n] != null ? row.means[n].toFixed(4) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}
