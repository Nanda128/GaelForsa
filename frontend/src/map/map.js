// Initialize the map and configure it to focus on Ireland
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export const MAP_CONFIG = {
    center: [53.5, -8.0], // that's Ireland
    zoom: 7,
    maxZoom: 19,
    tileUrl: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
};

/**
 * Initializes the Leaflet map
 * @param {string} containerId - The ID of the container element
 * @returns {L.Map} The initialized map instance
 */
export function initMap(containerId = 'map') {
    const map = L.map(containerId).setView(MAP_CONFIG.center, MAP_CONFIG.zoom);

    L.tileLayer(MAP_CONFIG.tileUrl, {
        maxZoom: MAP_CONFIG.maxZoom,
        attribution: MAP_CONFIG.attribution
    }).addTo(map);

    return map;
}

