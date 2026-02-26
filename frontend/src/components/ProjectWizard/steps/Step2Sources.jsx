import React from 'react'
import { SENSOR_OPTIONS, SENSOR_BANDS } from '../wizardConstants'

/**
 * Step 2 — Imagery Sources
 *
 * Lets the user:
 *   - toggle which sensors they want (at least one required)
 *   - pick which bands to extract per sensor (clickable chips, no checkboxes)
 *   - enter the display tile URL (XYZ/TMS format, only used for the map base layer)
 */
export default function Step2Sources({ formData, setFormData, errors }) {
  const { sensors, imagery_url } = formData

  // ── helpers ────────────────────────────────────────────────────────────────

  /** Is a given sensor_type currently enabled? */
  const isSensorEnabled = (sensorType) =>
    sensors.some(s => s.sensor_type === sensorType)

  /** Get the band list for an enabled sensor (or empty array). */
  const getSensorBands = (sensorType) =>
    sensors.find(s => s.sensor_type === sensorType)?.bands ?? []

  /**
   * Toggle a sensor on/off. Removing a sensor also removes its bands from the
   * sensors list; adding it back starts with all bands selected.
   */
  const toggleSensor = (sensorType) => {
    setFormData(prev => {
      const already = prev.sensors.some(s => s.sensor_type === sensorType)
      if (already) {
        // must keep at least one sensor
        if (prev.sensors.length === 1) return prev
        return { ...prev, sensors: prev.sensors.filter(s => s.sensor_type !== sensorType) }
      }
      // add sensor with all its bands selected by default
      const allBands = SENSOR_BANDS[sensorType]?.map(b => b.id) ?? []
      return { ...prev, sensors: [...prev.sensors, { sensor_type: sensorType, bands: allBands }] }
    })
  }

  /**
   * Toggle a single band for a given sensor. Removing all bands of a sensor
   * automatically disables it.
   */
  const toggleBand = (sensorType, bandId) => {
    setFormData(prev => {
      const updated = prev.sensors.map(s => {
        if (s.sensor_type !== sensorType) return s
        const has = s.bands.includes(bandId)
        const newBands = has ? s.bands.filter(b => b !== bandId) : [...s.bands, bandId]
        return { ...s, bands: newBands }
      })
      // remove sensors that now have zero bands
      const filtered = updated.filter(s => s.bands.length > 0)
      // if that removed the last sensor, undo the change
      if (filtered.length === 0) return prev
      return { ...prev, sensors: filtered }
    })
  }

  // ── render ──────────────────────────────────────────────────────────────────

  return (
    <div className="step-section">
      <h3>Imagery Sources</h3>

      {/* Display tile URL */}
      <div className="field">
        <label>Display Tile URL <span className="field-hint">(XYZ/TMS format, shown on the map)</span></label>
        <input
          type="url"
          value={imagery_url}
          onChange={e => setFormData(prev => ({ ...prev, imagery_url: e.target.value }))}
          placeholder="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          className={errors?.imagery_url ? 'input-error' : ''}
        />
        {errors?.imagery_url && <span className="field-error">{errors.imagery_url}</span>}
        <span className="field-hint">Leave as OpenStreetMap default for RGB-only projects.</span>
      </div>

      {/* Sensor toggles */}
      <div className="field">
        <label>Sensors</label>
        <span className="field-hint">
          Click a sensor card to enable it. Then select which bands to extract for ML features.
        </span>
        {errors?.sensors && <span className="field-error">{errors.sensors}</span>}

        <div className="sensor-list">
          {SENSOR_OPTIONS.map(opt => {
            const enabled = isSensorEnabled(opt.id)
            const activeBands = getSensorBands(opt.id)
            const availableBands = SENSOR_BANDS[opt.id] ?? []

            return (
              <div key={opt.id} className={`sensor-block${enabled ? ' sensor-block--active' : ''}`}>
                {/* Sensor header — whole row is the toggle */}
                <div className="sensor-toggle-row" onClick={() => toggleSensor(opt.id)}>
                  <div className={`sensor-dot${enabled ? ' on' : ''}`} />
                  <span className="sensor-label">{opt.label}</span>
                  <span className="sensor-status">{enabled ? 'Enabled' : 'Click to enable'}</span>
                </div>

                {/* Band chips — only shown when sensor is enabled */}
                {enabled && (
                  <div className="sensor-bands">
                    <span className="field-hint">Bands to extract:</span>
                    <div className="band-grid">
                      {availableBands.map(band => (
                        <button
                          key={band.id}
                          type="button"
                          className={`band-chip${activeBands.includes(band.id) ? ' active' : ''}`}
                          onClick={() => toggleBand(opt.id, band.id)}
                        >
                          {band.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
