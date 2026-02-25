import './Sidebar.css'
import ClassPanel from './ClassPanel'
import TrainPanel from './TrainPanel'
import PredictPanel from './PredictPanel'
import { useAppContext } from '../../context/AppContext'
import SatellitePanel from '../SatellitePanel/SatellitePanel'

export default function Sidebar() {
  const { activeProject, setActiveProject, setShowWizard } = useAppContext()

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          {/* Layers icon */}
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          </svg>
        </div>
        <div className="sidebar-title">
          <h1>LC Mapper</h1>
          <span>Land Cover Mapping</span>
        </div>
      </div>

      {activeProject && (
        <div className="sidebar-project-bar">
          <div className="sidebar-project-info">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="7" height="7" rx="1"/>
              <rect x="14" y="3" width="7" height="7" rx="1"/>
              <rect x="3" y="14" width="7" height="7" rx="1"/>
              <rect x="14" y="14" width="7" height="7" rx="1"/>
            </svg>
            <span title={activeProject.name}>{activeProject.name}</span>
          </div>
          <button
            className="sidebar-project-switch"
            onClick={() => setActiveProject(null)}
            title="Switch project"
          >
            ⇄
          </button>
        </div>
      )}

      <div className="sidebar-body">
        <ClassPanel />
        <div style={{ height: 12 }} />
        <SatellitePanel />
        <div style={{ height: 12 }} />
        <TrainPanel />
        <div style={{ height: 12 }} />
        <PredictPanel />
      </div>
    </aside>
  )
}
