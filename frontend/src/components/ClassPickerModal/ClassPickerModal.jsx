import { useState } from 'react'
import './ClassPickerModal.css'
import { useAppContext } from '../../context/AppContext'
import { createTrainingSample } from '../../services/api'

export default function ClassPickerModal() {
  const { classes, pendingGeometry, setPendingGeometry, refreshSamples, activeProject } = useAppContext()
  const [selectedId, setSelectedId] = useState(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  if (!pendingGeometry) return null

  function handleCancel() {
    setPendingGeometry(null)
    setSelectedId(null)
    setError(null)
  }

  async function handleSave() {
    if (!selectedId) return
    const cls = classes.find((c) => c.id === selectedId)
    setSaving(true)
    setError(null)
    try {
      await createTrainingSample({
        geometry: pendingGeometry,
        label: cls.name,
        class_id: cls.id,
        project_id: activeProject?.id ?? null,
      })
      await refreshSamples()
      setPendingGeometry(null)
      setSelectedId(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={handleCancel}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3>Assign Class</h3>
        <p className="modal-subtitle">Select a land cover class for this polygon</p>

        <div className="modal-classes">
          {classes.map((cls) => (
            <button
              key={cls.id}
              className={`modal-class-btn ${selectedId === cls.id ? 'active' : ''}`}
              onClick={() => setSelectedId(cls.id)}
            >
              <span className="swatch" style={{ background: cls.color }} />
              <span>{cls.name}</span>
            </button>
          ))}
        </div>

        {error && <p style={{ color: '#f87171', fontSize: 12, marginBottom: 12 }}>{error}</p>}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={handleCancel} disabled={saving}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={!selectedId || saving}
          >
            {saving ? 'Saving…' : 'Save Sample'}
          </button>
        </div>
      </div>
    </div>
  )
}
