// Turbine marker logic for Leaflet map and connect to the backend API

import L from 'leaflet';
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';
import 'leaflet.markercluster/dist/MarkerCluster.Default.css';
import {getApiBaseUrl} from '../utils/api.js';
import {showLoading, hideLoading, showError} from '../utils/ui.js';
import {escapeHtml, formatDateTime} from '../utils/helpers.js';

/**
 * @typedef {Object} Turbine
 * @property {number} id - Turbine ID
 * @property {string} name - Turbine name
 * @property {number} latitude - Latitude coordinate
 * @property {number} longitude - Longitude coordinate
 * @property {'green'|'yellow'|'red'} status - Turbine status
 * @property {string} updated_at - ISO datetime string of last update
 * @property {string} created_at - ISO datetime string of creation
 */

export const STATUS_COLORS = {
    green: '#28a745', yellow: '#ffc107', red: '#dc3545'
};

/**
 * Configuration options for marker clustering
 * @type {Object}
 */
export const CLUSTER_CONFIG = {
    chunkedLoading: true,           //Load markers in chunks for better performance
    maxClusterRadius: 50,           //Cluster markers within 50px at current zoom
    spiderfyOnMaxZoom: true,        //Spread out overlapping markers at max zoom
    showCoverageOnHover: false,     //Disable coverage polygon on hover for performance
    zoomToBoundsOnClick: true,      //Zoom to cluster bounds on click
    disableClusteringAtZoom: 16     //Disable clustering at high zoom levels
};

let markerClusterGroup = null;

/**
 * Get human-readable status label
 * @param {'green'|'yellow'|'red'} status - Turbine status
 * @returns {string} Status label
 */
function getStatusLabel(status) {
    const labels = {
        green: 'Healthy', yellow: 'Warning', red: 'Critical'
    };
    return labels[status] || 'Unknown';
}

/**
 * Create popup HTML content for a turbine
 * @param {Turbine} turbine - The turbine object
 * @returns {string} Turbine popup content
 */
function createPopupContent(turbine) {
    const statusLabel = getStatusLabel(turbine.status);
    const lastUpdated = formatDateTime(turbine.updated_at);

    return `
        <div class="turbine-popup">
            <h3>${escapeHtml(turbine.name)}</h3>
            <span class="status ${turbine.status}">${statusLabel}</span>
            <div class="last-updated">
                <strong>Last Updated:</strong><br>
                ${lastUpdated}
            </div>
        </div>
    `;
}

/**
 * Create a custom circle marker with the specified color
 * @param {Turbine} turbine - The turbine object
 * @returns {L.CircleMarker} The created marker
 */
function createTurbineMarker(turbine) {
    const color = STATUS_COLORS[turbine.status] || STATUS_COLORS.green;

    const marker = L.circleMarker([turbine.latitude, turbine.longitude], {
        radius: 12, fillColor: color, color: '#fff', weight: 2, opacity: 1, fillOpacity: 0.8
    });

    const popupContent = createPopupContent(turbine);
    marker.bindPopup(popupContent);

    return marker;
}

/**
 * Clear all turbine markers from the map
 * @param {L.Map} map - The Leaflet map instance
 */
function clearMarkers(map) {
    if (markerClusterGroup) {
        map.removeLayer(markerClusterGroup);
        markerClusterGroup = null;
    }
}

/**
 * Fetch turbines from API and add markers to map
 * @param {L.Map} map - The Leaflet map instance
 */
export async function loadTurbines(map) {
    showLoading();

    try {
        const apiBaseUrl = getApiBaseUrl();
        const response = await fetch(`${apiBaseUrl}/turbines/`);

        if (!response.ok) {
            console.error('Error loading turbines:', `HTTP error! status: ${response.status}`);
            showError('Failed to load turbines. Please check the API connection.');
            return;
        }

        const data = await response.json();

        const turbines = data.results || data;
        clearMarkers(map);

        markerClusterGroup = L.markerClusterGroup(CLUSTER_CONFIG);

        turbines.forEach(turbine => {
            if (turbine.latitude && turbine.longitude) {
                const marker = createTurbineMarker(turbine);
                markerClusterGroup.addLayer(marker);
            }
        });

        map.addLayer(markerClusterGroup);

        if (markerClusterGroup.getLayers().length > 0) {
            map.fitBounds(markerClusterGroup.getBounds().pad(0.1));
        }

        console.log(`Loaded ${turbines.length} turbines`);

    } catch (error) {
        console.error('Error loading turbines:', error);
        showError('Failed to load turbines. Please check the API connection.');
    } finally {
        hideLoading();
    }
}

