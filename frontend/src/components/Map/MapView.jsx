import { MapContainer, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./MapView.css";
import GeomanControl from "./GeomanControl";
import DrawingHandler from "./DrawingHandler";
import TrainingLayer from "./TrainingLayer";
import MapEventHandler from "./MapEventHandler";
import { useAppContext } from "../../context/AppContext";
import { buildSatTileUrl, MIN_SAT_ZOOM } from "../SatellitePanel/bandCombos";
import { getPredictTileUrl, getUncertaintyTileUrl } from "../../services/api";

// Default center: roughly central Europe / global overview
const DEFAULT_CENTER = [48.0, 16.0];
const DEFAULT_ZOOM = 5;

export default function MapView() {
  const {
    activeProject,
    activeScene,
    activeBandCombo,
    satelliteOpacity,
    mapZoom,
    prediction,
    predictionOpacity,
    uncertaintyOpacity,
  } = useAppContext();
  const satTileUrl =
    activeScene && mapZoom >= MIN_SAT_ZOOM
      ? buildSatTileUrl(activeScene, activeBandCombo)
      : null;

  return (
    <div className="map-container">
      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxZoom={19}
        />
        {satTileUrl && (
          <TileLayer
            key={satTileUrl}
            url={satTileUrl}
            opacity={satelliteOpacity}
            maxZoom={19}
            tileSize={256}
            keepBuffer={4}
            updateWhenIdle={true}
            updateInterval={300}
            attribution="Imagery &copy; Microsoft Planetary Computer / ESA Sentinel-2"
          />
        )}
        {prediction?.timestamp && (
          <TileLayer
            key={`pred-${prediction.timestamp}`}
            url={getPredictTileUrl(activeProject?.id, prediction.timestamp)}
            opacity={predictionOpacity}
            zIndex={400}
            tileSize={256}
          />
        )}
        {prediction?.timestamp && (
          <TileLayer
            key={`unc-${prediction.timestamp}`}
            url={getUncertaintyTileUrl(activeProject?.id, prediction.timestamp)}
            opacity={uncertaintyOpacity}
            zIndex={401}
            tileSize={256}
          />
        )}
        <MapEventHandler />
        <GeomanControl />
        <DrawingHandler />
        <TrainingLayer />
      </MapContainer>
    </div>
  );
}
