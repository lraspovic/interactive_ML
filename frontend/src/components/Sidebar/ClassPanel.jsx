import { useState } from 'react'
import './ClassPanel.css'
import { useAppContext } from '../../context/AppContext'

export default function ClassPanel() {
  const { classes, samples } = useAppContext()
  const [open, setOpen] = useState(true)

  // Count samples per class_id
  const countByClass = samples.features.reduce((acc, f) => {
    const id = f.properties.class_id
    acc[id] = (acc[id] || 0) + 1
    return acc
  }, {})

  const totalSamples = samples.features.length

  if (!classes.length) return null

  return (
    <div className="class-panel">
      <button className="class-panel-header" onClick={() => setOpen(o => !o)}>
        <span className="class-panel-icon">🗂️</span>
        <span className="class-panel-title">Land Cover Classes</span>
        {totalSamples > 0 && (
          <span className="class-panel-badge">{totalSamples}</span>
        )}
        <span className="class-panel-chevron">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="class-panel-body">
          <ul className="class-list">
            {classes.map((cls) => (
              <li key={cls.id} className="class-item">
                <span className="class-swatch" style={{ background: cls.color }} />
                <span className="class-name">{cls.name}</span>
                {countByClass[cls.id] ? (
                  <span className="class-count">{countByClass[cls.id]}</span>
                ) : null}
              </li>
            ))}
          </ul>
          <p className="class-hint">
            Draw a polygon on the map, then assign a class to save it as a training sample.
          </p>
        </div>
      )}
    </div>
  )
}
