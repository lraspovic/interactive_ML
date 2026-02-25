import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { fetchTrainingSamples, listProjects, getProject } from '../services/api'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  // null = loading, [] = no projects, [...] = projects list
  const [projects, setProjects] = useState(null)
  const [activeProject, setActiveProject] = useState(null)
  const [showWizard, setShowWizard] = useState(false)

  const [samples, setSamples] = useState({ type: 'FeatureCollection', features: [] })

  // Geometry (GeoJSON) captured from Geoman that awaits class assignment
  const [pendingGeometry, setPendingGeometry] = useState(null)
  // { id, label } of the sample awaiting delete confirmation
  const [pendingDeletion, setPendingDeletion] = useState(null)
  // { message, type } for transient notifications
  const [toast, setToast] = useState(null)

  // Classes come directly from the active project — no separate fetch needed
  const classes = activeProject?.classes ?? []

  // ── Map viewport (reported by MapEventHandler) ──────────────────────────
  const [mapZoom, setMapZoom] = useState(5)
  const [mapBounds, setMapBounds] = useState(null) // {minLon,minLat,maxLon,maxLat}

  // ── Satellite imagery ────────────────────────────────────────────────────
  const [activeScene, setActiveScene] = useState(null)   // scene obj from stac-search
  const [activeBandCombo, setActiveBandCombo] = useState('true_color')
  const [satelliteOpacity, setSatelliteOpacity] = useState(1.0)

  // ── Prediction overlays ──────────────────────────────────────────────────
  // prediction: null | { bbox, timestamp } — set when a prediction job succeeds
  const [prediction, setPrediction] = useState(null)
  const [predictionOpacity, setPredictionOpacity] = useState(0.7)
  const [uncertaintyOpacity, setUncertaintyOpacity] = useState(0.0)

  // Load projects list on startup
  const refreshProjects = useCallback(() => {
    return listProjects().then(setProjects).catch(console.error)
  }, [])

  useEffect(() => {
    refreshProjects()
  }, [refreshProjects])

  // Fetch training samples scoped to active project
  const refreshSamples = useCallback(() => {
    const id = activeProject?.id ?? null
    fetchTrainingSamples(id).then(setSamples).catch(console.error)
  }, [activeProject?.id])

  useEffect(() => {
    refreshSamples()
  }, [refreshSamples])

  async function loadProject(projectId) {
    const project = await getProject(projectId)
    setActiveProject(project)
    setShowWizard(false)
    // Clear all per-project overlay state when switching projects
    setPrediction(null)
    setActiveScene(null)
    setActiveBandCombo('true_color')
  }

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3000)
  }, [])

  return (
    <AppContext.Provider
      value={{
        // project management
        projects,
        activeProject,
        setActiveProject,
        loadProject,
        refreshProjects,
        showWizard,
        setShowWizard,
        // map / annotation
        classes,
        samples,
        refreshSamples,
        pendingGeometry,
        setPendingGeometry,
        pendingDeletion,
        setPendingDeletion,
        // map viewport
        mapZoom,
        setMapZoom,
        mapBounds,
        setMapBounds,
        // satellite imagery
        activeScene,
        setActiveScene,
        activeBandCombo,
        setActiveBandCombo,
        satelliteOpacity,
        setSatelliteOpacity,
        // prediction overlays
        prediction,
        setPrediction,
        predictionOpacity,
        setPredictionOpacity,
        uncertaintyOpacity,
        setUncertaintyOpacity,
        // notifications
        toast,
        showToast,
      }}
    >
      {children}
    </AppContext.Provider>
  )
}

export function useAppContext() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useAppContext must be used inside AppProvider')
  return ctx
}

