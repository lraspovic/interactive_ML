import { useEffect } from 'react'
import { useMap } from 'react-leaflet'
import '@geoman-io/leaflet-geoman-free'
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css'

/**
 * Activates the Geoman drawing toolbar on the Leaflet map instance.
 * Must be rendered as a child of <MapContainer>.
 */
export default function GeomanControl() {
  const map = useMap()

  useEffect(() => {
    if (!map.pm) return

    map.pm.addControls({
      position: 'topleft',
      drawMarker: false,
      drawCircleMarker: false,
      drawPolyline: false,
      drawCircle: false,
      drawText: false,
      drawPolygon: true,
      drawRectangle: true,
      editMode: true,
      dragMode: true,
      cutPolygon: false,
      removalMode: true,
      rotateMode: false,
    })

    map.pm.setGlobalOptions({
      snappable: true,
      snapDistance: 10,
    })

    // Clean up controls on unmount
    return () => {
      map.pm.removeControls()
    }
  }, [map])

  return null
}
