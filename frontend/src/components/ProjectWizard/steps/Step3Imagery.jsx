import React from 'react'
import { BAND_OPTIONS, SPECTRAL_INDICES } from '../wizardConstants'

export default function Step3Imagery({ formData, setFormData, errors }) {
  const set = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const toggleBand = (id) => {
    const current = formData.available_bands
    const next = current.includes(id) ? current.filter(b => b !== id) : [...current, id]
    // remove indices that no longer have required bands
    const validIndices = formData.enabled_indices.filter(idx => {
      const req = SPECTRAL_INDICES.find(s => s.id === idx)?.required ?? []
      return req.every(b => next.includes(b))
    })
    setFormData(prev => ({ ...prev, available_bands: next, enabled_indices: validIndices }))
  }

  const toggleIndex = (id) => {
    const current = formData.enabled_indices
    const next = current.includes(id) ? current.filter(i => i !== id) : [...current, id]
    set('enabled_indices', next)
  }

  return (
    <div className="step-section">
      <h3>Imagery Source</h3>

      <div className="field">
        <label>Tile URL Template</label>
        <input
          type="url"
          value={formData.imagery_url}
          onChange={e => set('imagery_url', e.target.value)}
          placeholder="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {errors.imagery_url && <span className="error-msg">{errors.imagery_url}</span>}
      </div>

      <div className="field">
        <label>Available Bands</label>
        <div className="checkbox-grid">
          {BAND_OPTIONS.map(band => (
            <label
              key={band.id}
              className={`check-item${formData.available_bands.includes(band.id) ? ' checked' : ''}`}
            >
              <input
                type="checkbox"
                checked={formData.available_bands.includes(band.id)}
                onChange={() => toggleBand(band.id)}
              />
              {band.label}
            </label>
          ))}
        </div>
      </div>

      <div className="field">
        <label>Spectral Indices</label>
        <div className="checkbox-grid">
          {SPECTRAL_INDICES.map(idx => {
            const canEnable = idx.required.every(b => formData.available_bands.includes(b))
            const isOn = formData.enabled_indices.includes(idx.id)
            return (
              <label
                key={idx.id}
                className={`check-item${isOn ? ' checked' : ''}${!canEnable ? ' disabled' : ''}`}
                title={!canEnable ? `Requires: ${idx.required.join(', ')}` : idx.description}
              >
                <input
                  type="checkbox"
                  checked={isOn}
                  disabled={!canEnable}
                  onChange={() => canEnable && toggleIndex(idx.id)}
                />
                {idx.label}
                <span className="check-badge">{idx.description}</span>
              </label>
            )
          })}
        </div>
      </div>

      <div className="field" style={{ maxWidth: 200 }}>
        <label>Resolution (m/px)</label>
        <input
          type="number"
          min="1"
          value={formData.resolution_m}
          onChange={e => set('resolution_m', parseFloat(e.target.value) || 10)}
        />
      </div>
    </div>
  )
}
