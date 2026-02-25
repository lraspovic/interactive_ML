import React, { useState } from 'react'
import './ProjectWizard.css'
import { DEFAULT_CLASSES } from './wizardConstants'
import { createProject } from '../../services/api'
import { useAppContext } from '../../context/AppContext'
import Step1Basics  from './steps/Step1Basics'
import Step2AOI     from './steps/Step2AOI'
import Step3Imagery from './steps/Step3Imagery'
import Step4Classes from './steps/Step4Classes'
import Step5Model   from './steps/Step5Model'

const STEPS = [
  { label: 'Basics' },
  { label: 'AOI' },
  { label: 'Imagery' },
  { label: 'Classes' },
  { label: 'Model' },
]

const INITIAL_FORM = {
  name: '',
  description: '',
  task_type: 'classification',
  aoi_bbox: null,
  imagery_url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  available_bands: ['blue', 'green', 'red'],
  enabled_indices: [],
  resolution_m: 10,
  classes: DEFAULT_CLASSES.map(c => ({ ...c })),
  model_family: 'classical',
  model_type: 'random_forest',
  model_params: { n_estimators: 100, max_depth: null },
}

function bboxToGeoJSON(bbox) {
  if (!bbox) return null
  const { minLon, minLat, maxLon, maxLat } = bbox
  const coords = [minLon, minLat, maxLon, maxLat].map(Number)
  if (coords.some(n => isNaN(n))) return null
  const [wn, s, e, n] = [coords[0], coords[1], coords[2], coords[3]]
  return {
    type: 'Polygon',
    coordinates: [[[wn, s], [e, s], [e, n], [wn, n], [wn, s]]],
  }
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
      const b = formData.aoi_bbox
      if (b && Object.values(b).some(v => v !== '')) {
        const { minLat, maxLat, minLon, maxLon } = b
        if (Number(minLat) >= Number(maxLat)) errs.aoi_bbox = 'Min latitude must be less than max latitude.'
        if (Number(minLon) >= Number(maxLon)) errs.aoi_bbox = 'Min longitude must be less than max longitude.'
      }
    }
    if (s === 3) {
      if (!formData.imagery_url.trim()) errs.imagery_url = 'Imagery URL is required.'
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
      const aoi_geometry = bboxToGeoJSON(formData.aoi_bbox)
      const payload = {
        name: formData.name.trim(),
        description: formData.description.trim() || null,
        task_type: formData.task_type,
        aoi_geometry,
        imagery_url: formData.imagery_url.trim(),
        available_bands: formData.available_bands,
        enabled_indices: formData.enabled_indices,
        resolution_m: formData.resolution_m,
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
    <Step1Basics  formData={formData} setFormData={setFormData} errors={errors} />,
    <Step2AOI     formData={formData} setFormData={setFormData} errors={errors} />,
    <Step3Imagery formData={formData} setFormData={setFormData} errors={errors} />,
    <Step4Classes formData={formData} setFormData={setFormData} errors={errors} />,
    <Step5Model   formData={formData} setFormData={setFormData} errors={errors} />,
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
