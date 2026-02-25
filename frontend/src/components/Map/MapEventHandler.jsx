/**
 * Invisible map child component that listens for zoom/move events
 * and forwards the current viewport to AppContext.
 */
import { useEffect } from 'react'
import { useMapEvents } from 'react-leaflet'
import { useAppContext } from '../../context/AppContext'

export default function MapEventHandler() {
  const { setMapZoom, setMapBounds } = useAppContext()

  const reportViewport = (map) => {
    setMapZoom(map.getZoom())
    const b = map.getBounds()
    setMapBounds({
      minLon: b.getWest(),
      minLat: b.getSouth(),
      maxLon: b.getEast(),
      maxLat: b.getNorth(),
    })
  }

  const map = useMapEvents({
    zoomend: () => reportViewport(map),
    moveend: () => reportViewport(map),
  })

  // Report initial viewport once mounted
  useEffect(() => {
    reportViewport(map)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return null
}
