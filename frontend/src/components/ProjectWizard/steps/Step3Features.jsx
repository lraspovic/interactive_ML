import React, { useState, useEffect } from 'react'
import { getSpectralIndices } from '../../../services/api'
import { GLCM_STATISTICS, GLCM_WINDOW_SIZES } from '../wizardConstants'

/**
 * Step 3 — Feature Selection
 *
 * Two sections:
 *   1. Spectral Indices — fetched from backend GET /imagery/spectral-indices
 *      Grouped by domain, each domain has a "Select all" button.
 *      Indices are disabled if the project's available bands don't cover the
 *      required bands.
 *
 *   2. GLCM Textures — per raw band, configurable window & statistics.
 */
export default function Step3Features({ formData, setFormData, errors }) {
  const [indexCatalogue, setIndexCatalogue] = useState({}) // { domain: [{ id, formula, required_bands, description }] }
  const [loadingIndices, setLoadingIndices] = useState(true)
  const [glcmJsonError, setGlcmJsonError] = useState(null)
  const [openDomain, setOpenDomain] = useState(null)

  // Derive the union of all bands selected across all sensors
  const allBands = Array.from(
    new Set(formData.sensors.flatMap(s => s.bands))
  )

  // ── Fetch spectral index catalogue ─────────────────────────────────────────
  useEffect(() => {
    setLoadingIndices(true)
    getSpectralIndices()
      .then(data => {
        const catalogue = data ?? {}
        setIndexCatalogue(catalogue)
        // Auto-open first domain
        const firstDomain = Object.keys(catalogue).sort()[0]
        if (firstDomain) setOpenDomain(firstDomain)
      })
      .catch(() => setIndexCatalogue({}))
      .finally(() => setLoadingIndices(false))
  }, [])

  // ── Spectral index helpers ─────────────────────────────────────────────────

  const isComputable = (index) =>
    (index.required_bands ?? []).every(b => allBands.includes(b))

  const toggleIndex = (id) => {
    setFormData(prev => {
      const has = prev.enabled_indices.includes(id)
      return {
        ...prev,
        enabled_indices: has
          ? prev.enabled_indices.filter(i => i !== id)
          : [...prev.enabled_indices, id],
      }
    })
  }

  const selectAllInDomain = (indices) => {
    const computable = indices.filter(isComputable).map(i => i.id)
    setFormData(prev => ({
      ...prev,
      enabled_indices: Array.from(new Set([...prev.enabled_indices, ...computable])),
    }))
  }

  const clearAllInDomain = (indices) => {
    const ids = new Set(indices.map(i => i.id))
    setFormData(prev => ({
      ...prev,
      enabled_indices: prev.enabled_indices.filter(i => !ids.has(i)),
    }))
  }

  // ── GLCM helpers ───────────────────────────────────────────────────────────

  const glcmEnabled = !!formData.glcm_config

  const toggleGlcm = () => {
    if (glcmEnabled) {
      setFormData(prev => ({ ...prev, glcm_config: null }))
    } else {
      setFormData(prev => ({
        ...prev,
        glcm_config: {
          enabled: true,
          bands: allBands.slice(),
          window_size: 5,
          statistics: GLCM_STATISTICS.filter(s => s !== 'ASM'),
        },
      }))
    }
    setGlcmJsonError(null)
  }

  const setGlcmField = (field, value) => {
    setFormData(prev => ({
      ...prev,
      glcm_config: { ...(prev.glcm_config ?? {}), [field]: value },
    }))
  }

  const toggleGlcmBand = (bandId) => {
    const current = formData.glcm_config?.bands ?? []
    const updated = current.includes(bandId)
      ? current.filter(b => b !== bandId)
      : [...current, bandId]
    setGlcmField('bands', updated)
  }

  const toggleGlcmStat = (stat) => {
    const current = formData.glcm_config?.statistics ?? []
    const updated = current.includes(stat)
      ? current.filter(s => s !== stat)
      : [...current, stat]
    setGlcmField('statistics', updated)
  }

  // ── render ──────────────────────────────────────────────────────────────────

  const domains = Object.keys(indexCatalogue).sort()

  return (
    <div className="step-section">
      <h3>Feature Selection</h3>

      {allBands.length === 0 && (
        <div className="info-banner">
          No bands selected — go back to Step 2 to choose sensor bands.
        </div>
      )}

      {/* ── Spectral Indices ─────────────────────────────────────────────── */}
      <div className="field">
        <label>Spectral Indices</label>
        <span className="field-hint">
          Derived band ratios computed before ML training. Only indices whose required
          bands are in your selection can be enabled.
        </span>

        {loadingIndices ? (
          <div className="field-hint">Loading catalogue…</div>
        ) : domains.length === 0 ? (
          <div className="field-hint">No indices returned from server.</div>
        ) : (
          <div className="index-domain-list">
            {domains.map(domain => {
              const indices = indexCatalogue[domain] ?? []
              const domainComputable = indices.filter(isComputable)
              const domainSelected = indices.filter(i => formData.enabled_indices.includes(i.id))
              const isOpen = openDomain === domain

              return (
                <div key={domain} className={`index-domain${isOpen ? ' index-domain--open' : ''}`}>
                  {/* Domain header row */}
                  <div
                    className="index-domain-header"
                    onClick={() => setOpenDomain(isOpen ? null : domain)}
                  >
                    <span className="domain-chevron">{isOpen ? '▾' : '▸'}</span>
                    <span className="domain-name">{domain}</span>
                    <span className="domain-count">
                      {domainSelected.length}/{domainComputable.length} selected
                    </span>
                    <div className="domain-actions" onClick={e => e.stopPropagation()}>
                      <button
                        className="btn-link"
                        onClick={() => selectAllInDomain(indices)}
                        disabled={domainComputable.length === 0}
                        title="Select all computable indices in this domain"
                      >
                        Select all
                      </button>
                      <button
                        className="btn-link"
                        onClick={() => clearAllInDomain(indices)}
                        disabled={domainSelected.length === 0}
                      >
                        Clear
                      </button>
                    </div>
                  </div>

                  {/* Index chips — only when open */}
                  {isOpen && (
                    <div className="index-grid">
                      {indices.map(idx => {
                        const computable = isComputable(idx)
                        const checked = formData.enabled_indices.includes(idx.id)
                        const missing = (idx.required_bands ?? []).filter(b => !allBands.includes(b))
                        return (
                          <button
                            key={idx.id}
                            type="button"
                            className={`index-chip${checked ? ' active' : ''}${!computable ? ' disabled' : ''}`}
                            onClick={() => computable && toggleIndex(idx.id)}
                            title={
                              !computable
                                ? `Requires: ${missing.join(', ')}`
                                : (idx.description ?? idx.id)
                            }
                          >
                            <span className="index-id">{idx.id}</span>
                            {idx.formula && (
                              <span className="index-formula">{idx.formula}</span>
                            )}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* ── GLCM Textures ────────────────────────────────────────────────── */}
      <div className="field">
        <div className="glcm-toggle-row" onClick={toggleGlcm}>
          <div className={`sensor-dot${glcmEnabled ? ' on' : ''}`} />
          <span>GLCM Texture Features</span>
          <span className="sensor-status">{glcmEnabled ? 'Enabled' : 'Click to enable'}</span>
        </div>
        <span className="field-hint">
          Grey-Level Co-occurrence Matrix statistics computed per raw band polygon.
          Adds spatial texture information to the feature vector.
        </span>

        {glcmEnabled && formData.glcm_config && (
          <div className="glcm-config">
            {/* Bands */}
            <div className="field">
              <label>Bands for GLCM <span className="field-hint">(raw bands only)</span></label>
              <div className="band-grid">
                {allBands.map(band => (
                  <button
                    key={band}
                    type="button"
                    className={`band-chip${(formData.glcm_config.bands ?? []).includes(band) ? ' active' : ''}`}
                    onClick={() => toggleGlcmBand(band)}
                  >
                    {band}
                  </button>
                ))}
              </div>
            </div>

            {/* Window size */}
            <div className="field">
              <label>Window Size</label>
              <select
                value={formData.glcm_config.window_size ?? 5}
                onChange={e => setGlcmField('window_size', Number(e.target.value))}
              >
                {GLCM_WINDOW_SIZES.map(s => (
                  <option key={s} value={s}>{s} × {s}</option>
                ))}
              </select>
            </div>

            {/* Statistics */}
            <div className="field">
              <label>Statistics</label>
              <div className="band-grid">
                {GLCM_STATISTICS.map(stat => (
                  <button
                    key={stat}
                    type="button"
                    className={`band-chip${(formData.glcm_config.statistics ?? []).includes(stat) ? ' active' : ''}`}
                    onClick={() => toggleGlcmStat(stat)}
                  >
                    {stat}
                  </button>
                ))}
              </div>
            </div>

            {glcmJsonError && <span className="field-error">{glcmJsonError}</span>}
          </div>
        )}
      </div>
    </div>
  )
}
