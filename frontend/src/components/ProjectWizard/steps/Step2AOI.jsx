import React from 'react'

export default function Step2AOI({ formData, setFormData, errors }) {
  const bbox = formData.aoi_bbox ?? { minLon: '', minLat: '', maxLon: '', maxLat: '' }

  const set = (field, value) => {
    setFormData(prev => ({
      ...prev,
      aoi_bbox: { ...bbox, [field]: value },
    }))
  }

  return (
    <div className="step-section">
      <h3>Area of Interest</h3>
      <p className="hint">
        Define the geographic bounding box for your project. Predictions will be clipped
        to this area. You can skip this step and define it later.
      </p>

      <div className="two-col">
        <div className="field">
          <label>Min Latitude</label>
          <input
            type="number"
            step="any"
            value={bbox.minLat}
            onChange={e => set('minLat', e.target.value)}
            placeholder="-90"
          />
        </div>
        <div className="field">
          <label>Max Latitude</label>
          <input
            type="number"
            step="any"
            value={bbox.maxLat}
            onChange={e => set('maxLat', e.target.value)}
            placeholder="90"
          />
        </div>
        <div className="field">
          <label>Min Longitude</label>
          <input
            type="number"
            step="any"
            value={bbox.minLon}
            onChange={e => set('minLon', e.target.value)}
            placeholder="-180"
          />
        </div>
        <div className="field">
          <label>Max Longitude</label>
          <input
            type="number"
            step="any"
            value={bbox.maxLon}
            onChange={e => set('maxLon', e.target.value)}
            placeholder="180"
          />
        </div>
      </div>

      {errors.aoi_bbox && <span className="error-msg">{errors.aoi_bbox}</span>}
    </div>
  )
}
