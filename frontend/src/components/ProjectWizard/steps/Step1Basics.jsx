import React from 'react'

export default function Step1Basics({ formData, setFormData, errors }) {
  const set = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  return (
    <div className="step-section">
      <h3>Project Basics</h3>

      <div className="field">
        <label>Project Name *</label>
        <input
          type="text"
          value={formData.name}
          onChange={e => set('name', e.target.value)}
          placeholder="e.g. Vegetation Mapping 2024"
          autoFocus
        />
        {errors.name && <span className="error-msg">{errors.name}</span>}
      </div>

      <div className="field">
        <label>Description</label>
        <textarea
          value={formData.description}
          onChange={e => set('description', e.target.value)}
          placeholder="Optional — describe the purpose of this project"
        />
      </div>

      <div className="field">
        <label>Task Type</label>
        <div className="radio-group">
          {[
            { id: 'classification', label: 'Classification' },
            { id: 'regression',     label: 'Regression' },
          ].map(opt => (
            <label key={opt.id} className={`radio-btn${formData.task_type === opt.id ? ' selected' : ''}`}>
              <input
                type="radio"
                name="task_type"
                value={opt.id}
                checked={formData.task_type === opt.id}
                onChange={() => set('task_type', opt.id)}
              />
              {opt.label}
            </label>
          ))}
        </div>
      </div>
    </div>
  )
}
