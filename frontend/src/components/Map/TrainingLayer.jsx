import { GeoJSON } from 'react-leaflet'
import { useAppContext } from '../../context/AppContext'

/**
 * Renders all saved training samples as colored polygons on the map.
 * A new key forces react-leaflet to remount the GeoJSON layer whenever
 * the samples or palette changes.
 */
export default function TrainingLayer() {
  const { samples, classes } = useAppContext()

  const colorById = Object.fromEntries(classes.map((c) => [c.id, c.color]))

  if (!samples.features.length) return null

  return (
    <GeoJSON
      key={samples.features.length + JSON.stringify(colorById)}
      data={samples}
      style={(feature) => {
        const color = colorById[feature.properties.class_id] ?? '#ffffff'
        return {
          color,
          weight: 2,
          opacity: 0.9,
          fillColor: color,
          fillOpacity: 0.25,
        }
      }}
      onEachFeature={(feature, layer) => {
        const { label, created_at } = feature.properties
        layer.bindTooltip(
          `<strong>${label}</strong><br/><small>${new Date(created_at).toLocaleString()}</small>`,
          { sticky: true }
        )
      }}
    />
  )
}
