import { useEffect } from 'react'
import { useMap } from 'react-leaflet'
import { useAppContext } from '../../context/AppContext'

/**
 * Listens for Geoman's pm:create and pm:remove events.
 * - create: captures drawn geometry → opens ClassPickerModal
 * - remove: stores pending deletion → opens ConfirmDeleteModal
 *   (Geoman already removed the layer visually; refreshSamples on cancel restores it)
 */
export default function DrawingHandler() {
  const map = useMap()
  const { setPendingGeometry, setPendingDeletion, refreshSamples } = useAppContext()

  useEffect(() => {
    function onLayerCreated({ layer }) {
      const geojson = layer.toGeoJSON()
      map.removeLayer(layer)
      setPendingGeometry(geojson.geometry)
    }

    function onLayerRemoved({ layer }) {
      const id = layer.feature?.id
      const label = layer.feature?.properties?.label ?? 'this polygon'
      if (!id) {
        // Not a saved sample (e.g. drawn but not yet saved) — nothing to do
        return
      }
      // Immediately refresh so the polygon comes back if the user cancels
      refreshSamples()
      setPendingDeletion({ id, label })
    }

    map.on('pm:create', onLayerCreated)
    map.on('pm:remove', onLayerRemoved)
    return () => {
      map.off('pm:create', onLayerCreated)
      map.off('pm:remove', onLayerRemoved)
    }
  }, [map, setPendingGeometry, setPendingDeletion, refreshSamples])

  return null
}
