import './Toast.css'
import { useAppContext } from '../../context/AppContext'

export default function Toast() {
  const { toast } = useAppContext()
  if (!toast) return null
  return (
    <div className={`toast toast-${toast.type}`}>
      {toast.type === 'success' ? '✓' : '✕'} {toast.message}
    </div>
  )
}
