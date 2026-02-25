import './ProjectLanding.css'
import { useAppContext } from '../../context/AppContext'

export default function ProjectLanding() {
  const { projects, loadProject, setShowWizard } = useAppContext()

  const loading = projects === null

  return (
    <div className="landing-overlay">
      <div className="landing-header">
        <div className="landing-logo">
          <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" /></svg>
        </div>
        <h1>LC Mapper</h1>
        <p>Interactive machine learning for remote sensing image classification</p>
      </div>

      <div className="landing-body">
        {loading ? (
          <p style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>Loading…</p>
        ) : (
          <>
            <div className="landing-section-header">
              <h2>Existing Projects</h2>
            </div>

            {projects.length === 0 ? (
              <div className="landing-empty">
                No projects yet. Create your first project to get started.
              </div>
            ) : (
              <div className="project-cards">
                {projects.map((p) => (
                  <button
                    key={p.id}
                    className="project-card"
                    onClick={() => loadProject(p.id)}
                  >
                    <div className="project-card-name">{p.name}</div>
                    <div className="project-card-meta">
                      <span className="project-card-badge">{p.task_type}</span>
                      <span>{p.class_count} classes</span>
                    </div>
                  </button>
                ))}
              </div>
            )}

            <button className="landing-create-btn" onClick={() => setShowWizard(true)}>
              + Create New Project
            </button>
          </>
        )}
      </div>
    </div>
  )
}
