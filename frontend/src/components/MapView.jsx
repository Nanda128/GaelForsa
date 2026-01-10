import React, { useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { formatDateTime } from '../utils/helpers';
import { STATUS_COLORS, STATUS_LABELS, MAP_CONFIG } from '../constants/status';
import { filterTurbinesByStatus } from '../utils/validation';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

function createCustomIcon(status) {
    const color = STATUS_COLORS[status] || '#666';
    return L.divIcon({
        className: 'custom-marker',
        html: `<div style="
            width: 20px;
            height: 20px;
            background-color: ${color};
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        "></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
        popupAnchor: [0, -10]
    });
}

function MapUpdater() {
    const map = useMap();
    
    React.useEffect(() => {
        // const start = Date.now();
        const timer = setTimeout(() => {
            map.invalidateSize();
            // console.log('Map invalidated after', Date.now() - start, 'ms');
        }, 100);
        // Tried 50ms but 100ms works better
        return () => clearTimeout(timer);
    }, [map]);

    React.useEffect(() => {
        const handleResize = () => {
            setTimeout(() => map.invalidateSize(), 100);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, [map]);

    return null;
}

function MapView({ turbines = [], statusFilters, onTurbineClick }) {
    const filteredTurbines = useMemo(() => {
        const result = filterTurbinesByStatus(turbines, statusFilters);
        // console.log('Filtered turbines:', result.length, 'from', turbines.length);
        return result;
    }, [turbines, statusFilters]);

    if (!turbines || turbines.length === 0) {
        return (
            <main className="map-container">
                <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-tertiary)' }}>
                    No turbines available
                </div>
            </main>
        );
    }

    return (
        <main className="map-container">
            <MapContainer
                center={MAP_CONFIG.center}
                zoom={MAP_CONFIG.zoom}
                style={{ height: '100%', width: '100%', zIndex: 1 }}
                zoomControl={true}
                scrollWheelZoom={true}
            >
                <TileLayer
                    url={MAP_CONFIG.tileUrl}
                    attribution={MAP_CONFIG.attribution}
                />
                <MapUpdater />
                {filteredTurbines.map(turbine => {
                    const lat = parseFloat(turbine.latitude);
                    const lon = parseFloat(turbine.longitude);
                    
                    return (
                        <Marker
                            key={turbine.id}
                            position={[lat, lon]}
                            icon={createCustomIcon(turbine.status)}
                        >
                            <Popup>
                                <div className="turbine-popup">
                                    <h3>{turbine.name}</h3>
                                    <span className={`status ${turbine.status}`}>
                                        {STATUS_LABELS[turbine.status] || 'Unknown'}
                                    </span>
                                    <div className="last-updated">
                                        <strong>Last Updated:</strong><br />
                                        {formatDateTime(turbine.updated_at)}
                                    </div>
                                    <button
                                        className="popup-details-btn"
                                        onClick={() => onTurbineClick(turbine)}
                                    >
                                        View Details
                                    </button>
                                </div>
                            </Popup>
                        </Marker>
                    );
                })}
            </MapContainer>
        </main>
    );
}

export default MapView;
