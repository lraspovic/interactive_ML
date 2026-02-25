import React from 'react'
import { MODEL_FAMILIES, HYPERPARAMS } from '../wizardConstants'

export default function Step5Model({ formData, setFormData }) {
  const set = (field, value) => setFormData(prev => ({ ...prev, [field]: value }))

  const setFamily = (family) => {
    const defaultType = MODEL_FAMILIES[family][0].id
    const defaults = buildDefaults(defaultType)
    setFormData(prev => ({ ...prev, model_family: family, model_type: defaultType, model_params: defaults }))
  }

  const setType = (type) => {
    const defaults = buildDefaults(type)
    setFormData(prev => ({ ...prev, model_type: type, model_params: defaults }))
  }

  const setParam = (id, value) => {
    setFormData(prev => ({ ...prev, model_params: { ...prev.model_params, [id]: value } }))
  }

  function buildDefaults(type) {
    const params = HYPERPARAMS[type] ?? []
    return Object.fromEntries(params.map(p => [p.id, p.default ?? '']))
  }

  const params = HYPERPARAMS[formData.model_type] ?? []

  return (
    <div className="step-section">
      <h3>Model Configuration</h3>

      <div className="field">
        <label>Model Family</label>
        <div className="model-family-toggle">
          {Object.keys(MODEL_FAMILIES).map(fam => (
            <button
              key={fam}
              className={`model-family-btn${formData.model_family === fam ? ' selected' : ''}`}
              onClick={() => setFamily(fam)}
            >
              {fam === 'classical' ? 'Classical ML' : 'Deep Learning'}
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label>Model Type</label>
        <select value={formData.model_type} onChange={e => setType(e.target.value)}>
          {MODEL_FAMILIES[formData.model_family].map(m => (
            <option key={m.id} value={m.id}>{m.label}</option>
          ))}
        </select>
      </div>

      {params.length > 0 && (
        <div className="field">
          <label>Hyperparameters</label>
          <div className="param-grid">
            {params.map(p => (
              <div className="field" key={p.id}>
                <label>{p.label}</label>
                {p.type === 'select' ? (
                  <select
                    value={formData.model_params[p.id] ?? p.default}
                    onChange={e => setParam(p.id, e.target.value)}
                  >
                    {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={formData.model_params[p.id] ?? ''}
                    onChange={e => setParam(p.id, e.target.value === '' ? null : parseFloat(e.target.value))}
                    placeholder={p.placeholder ?? ''}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
