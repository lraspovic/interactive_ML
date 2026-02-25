import { useState } from 'react'
import '../ClassPickerModal/ClassPickerModal.css'
import './ConfirmDeleteModal.css'
import { useAppContext } from '../../context/AppContext'
import { deleteTrainingSample } from '../../services/api'

export default function ConfirmDeleteModal() {
  const { pendingDeletion, setPendingDeletion, refreshSamples, showToast } = useAppContext()
  const [deleting, setDeleting] = useState(false)

  if (!pendingDeletion) return null

  function handleCancel() {
    // Polygon was already restored via refreshSamples() in DrawingHandler
    setPendingDeletion(null)
  }

  async function handleConfirm() {
    setDeleting(true)
    try {
      await deleteTrainingSample(pendingDeletion.id)
      await refreshSamples()
      showToast(`"${pendingDeletion.label}" sample deleted.`, 'success')
      setPendingDeletion(null)
    } catch (e) {
      showToast('Failed to delete sample.', 'error')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={handleCancel}>
      <div className="modal confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-icon">🗑️</div>
        <h3>Delete Sample?</h3>
        <p className="modal-subtitle">
          Are you sure you want to delete the{' '}
          <strong style={{ color: '#e8eaf0' }}>{pendingDeletion.label}</strong> polygon?
          <br />
          This action cannot be undone.
        </p>
        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={handleCancel} disabled={deleting}>
            Cancel
          </button>
          <button className="btn btn-danger" onClick={handleConfirm} disabled={deleting}>
            {deleting ? 'Deleting…' : 'Yes, Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}
