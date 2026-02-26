import React, { useState } from 'react'
import './ProjectWizard.css'
import { DEFAULT_CLASSES, DEFAULT_SENSORS } from './wizardConstants'
import { createProject } from '../../services/api'
import { useAppContext } from '../../context/AppContext'
import Step1Basics   from './steps/Step1Basics'
import Step2Sources  from './steps/Step2Sources'
import Step3Features from './steps/Step3Features'
import Step4Classes  from './steps/Step4Classes'
import Step5Model    from './steps/Step5Model'

const STEPS = [
  { label: 'Basics' },
  { label: 'Sources' },
  { label: 'Features' },
  { label: 'Classes' },
  { label: 'Model' },
]

const INITIAL_FORM = {
  // Step 1
  name: '',
  description: '',
  task_type: 'classification',
  // Step 2 — sources
  imagery_url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  sensors: DEFAULT_SENSORS.map(s => ({ ...s, bands: [...s.bands] })),
  // Step 3 — features
  enabled_indices: [],
  glcm_config: null,
  // Step 4
  classes: DEFAULT_CLASSES.map(c => ({ ...c })),
  // Step 5
  model_family: 'classical',
  model_type: 'random_forest',
  model_params: { n_estimators: 100, max_depth: null },
}

export default function ProjectWizard() {
  const { setShowWizard, loadProject, refreshProjects, showToast } = useAppContext()
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState(INITIAL_FORM)
  const [errors, setErrors] = useState({})
  const [submitting, setSubmitting] = useState(false)

  const validate = (s) => {
    const errs = {}
    if (s === 1) {
      if (!formData.name.trim()) errs.name = 'Project name is required.'
    }
    if (s === 2) {
      if (!formData.imagery_url.trim()) errs.imagery_url = 'Tile URL is required.'
      if (!formData.sensors.length) errs.sensors = 'Select at least one sensor.'
      const anyBands = formData.sensors.some(sc => sc.bands.length > 0)
      if (!anyBands) errs.sensors = 'Select at least one band.'
    }
    if (s === 4) {
      if (formData.classes.length < 2) errs.classes = 'At least 2 classes are required.'
      if (formData.classes.some(c => !c.name.trim())) errs.classes = 'All classes must have a name.'
    }
    return errs
  }

  const next = () => {
    const errs = validate(step)
    if (Object.keys(errs).length) { setErrors(errs); return }
    setErrors({})
    setStep(s => s + 1)
  }

  const back = () => { setErrors({}); setStep(s => s - 1) }

  const submit = async () => {
    const errs = validate(5)
    if (Object.keys(errs).length) { setErrors(errs); return }

    setSubmitting(true)
    try {
      // Derive the flat available_bands list as the union of all sensor bands
      // (required by the backend for backward-compat with feature extraction).
      const seen = new Set()
      const available_bands = []
      for (const sc of formData.sensors) {
        for (const b of sc.bands) {
          if (!seen.has(b)) { seen.add(b); available_bands.push(b) }
        }
      }

      const payload = {
        name: formData.name.trim(),
        description: formData.description.trim() || null,
        task_type: formData.task_type,
        imagery_url: formData.imagery_url.trim(),
        sensors: formData.sensors,
        available_bands,
        enabled_indices: formData.enabled_indices,
        glcm_config: formData.glcm_config || null,
        classes: formData.classes.map((c, i) => ({ name: c.name.trim(), color: c.color, display_order: i })),
        model_config: {
          family: formData.model_family,
          type: formData.model_type,
          params: formData.model_params,
        },
      }
      const project = await createProject(payload)
      await refreshProjects()
      showToast(`Project "${project.name}" created!`, 'success')
      loadProject(project.id)
      setShowWizard(false)
    } catch (err) {
      showToast(err?.message ?? 'Failed to create project.', 'error')
    } finally {
      setSubmitting(false)
    }
  }

  const stepComponent = [
    <Step1Basics   formData={formData} setFormData={setFormData} errors={errors} />,
    <Step2Sources  formData={formData} setFormData={setFormData} errors={errors} />,
    <Step3Features formData={formData} setFormData={setFormData} errors={errors} />,
    <Step4Classes  formData={formData} setFormData={setFormData} errors={errors} />,
    <Step5Model    formData={formData} setFormData={setFormData} errors={errors} />,
  ][step - 1]

  return (
    <div className="wizard-overlay" onClick={e => e.target === e.currentTarget && setShowWizard(false)}>
      <div className="wizard-card">

        {/* Header + step indicator */}
        <div className="wizard-header">
          <h2>New Project — Step {step} of {STEPS.length}: {STEPS[step - 1].label}</h2>
          <div className="wizard-steps">
            {STEPS.map((s, i) => {
              const n = i + 1
              const done   = n < step
              const active = n === step
              return (
                <React.Fragment key={n}>
                  <div className="wizard-step-item">
                    <div className={`step-pill${active ? ' active' : done ? ' done' : ''}`}>
                      {done ? '✓' : n}
                    </div>
                  </div>
                  {i < STEPS.length - 1 && (
                    <div className={`step-connector${done ? ' done' : ''}`} />
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </div>

        {/* Step body */}
        <div className="wizard-body">
          {stepComponent}
        </div>

        {/* Footer */}
        <div className="wizard-footer">
          <div className="wizard-footer-left">
            <button className="btn-ghost" onClick={() => setShowWizard(false)}>
              Cancel
            </button>
          </div>
          <div className="wizard-footer-right">
            {step > 1 && (
              <button className="btn-ghost" onClick={back}>
                ← Back
              </button>
            )}
            {step < STEPS.length ? (
              <button className="btn-primary" onClick={next}>
                Next →
              </button>
            ) : (
              <button className="btn-success" onClick={submit} disabled={submitting}>
                {submitting ? 'Creating…' : 'Create Project'}
              </button>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}
