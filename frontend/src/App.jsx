import './index.css'
import { AppProvider, useAppContext } from './context/AppContext'
import Sidebar from './components/Sidebar/Sidebar'
import MapView from './components/Map/MapView'
import FeatureTable from './components/FeatureTable/FeatureTable'
import ClassPickerModal from './components/ClassPickerModal/ClassPickerModal'
import ConfirmDeleteModal from './components/ConfirmDeleteModal/ConfirmDeleteModal'
import Toast from './components/Toast/Toast'
import ProjectLanding from './components/ProjectWizard/ProjectLanding'
import ProjectWizard from './components/ProjectWizard/ProjectWizard'

function AppShell() {
  const { activeProject, showWizard } = useAppContext()

  if (!activeProject) {
    return (
      <>
        <ProjectLanding />
        {showWizard && <ProjectWizard />}
        <Toast />
      </>
    )
  }

  return (
    <>
      <div style={{ display: 'flex', height: '100%', width: '100%' }}>
        <Sidebar />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          <MapView />
          <FeatureTable />
        </div>
      </div>
      <ClassPickerModal />
      <ConfirmDeleteModal />
      {showWizard && <ProjectWizard />}
      <Toast />
    </>
  )
}

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  )
}
