import { MapContainer, TileLayer, ImageOverlay } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const BOUNDS: [[number, number], [number, number]] = [
  [18.89, -156.07],
  [20.277, -154.799],
];

interface DemoMapProps {
  date: string;
  opacity: number;
  basePath: string;
}

export default function DemoMap({ date, opacity, basePath }: DemoMapProps) {
  return (
    <MapContainer
      bounds={BOUNDS}
      style={{ height: '480px', width: '100%' }}
      scrollWheelZoom={false}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {/* IMERG layer — fades out as opacity increases */}
      <ImageOverlay
        url={`${basePath}/samples/${date}/imerg.png`}
        bounds={BOUNDS}
        opacity={1 - opacity}
      />
      {/* Prediction layer — fades in as opacity increases */}
      <ImageOverlay
        url={`${basePath}/samples/${date}/pred.png`}
        bounds={BOUNDS}
        opacity={opacity}
      />
    </MapContainer>
  );
}
