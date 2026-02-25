import React from 'react'

export default function Step4Classes({ formData, setFormData, errors }) {
  const classes = formData.classes

  const update = (index, field, value) => {
    setFormData(prev => {
      const next = prev.classes.map((c, i) => i === index ? { ...c, [field]: value } : c)
      return { ...prev, classes: next }
    })
  }

  const add = () => {
    setFormData(prev => ({
      ...prev,
      classes: [...prev.classes, { name: '', color: '#888888' }],
    }))
  }

  const remove = (index) => {
    setFormData(prev => ({
      ...prev,
      classes: prev.classes.filter((_, i) => i !== index),
    }))
  }

  return (
    <div className="step-section">
      <h3>Class Scheme</h3>
      <p className="hint">
        Define the land cover classes for this project. At least 2 classes are required.
      </p>

      <div className="class-list">
        {classes.map((cls, i) => (
          <div key={i} className="class-row">
            <input
              type="color"
              className="class-color-input"
              value={cls.color}
              onChange={e => update(i, 'color', e.target.value)}
              title="Pick colour"
            />
            <input
              type="text"
              className="class-name-input"
              value={cls.name}
              onChange={e => update(i, 'name', e.target.value)}
              placeholder={`Class ${i + 1}`}
            />
            <button
              className="class-remove-btn"
              onClick={() => remove(i)}
              title="Remove"
              disabled={classes.length <= 2}
            >
              ×
            </button>
          </div>
        ))}

        <button className="add-class-btn" onClick={add}>
          + Add Class
        </button>
      </div>

      {errors.classes && <span className="error-msg">{errors.classes}</span>}
    </div>
  )
}
