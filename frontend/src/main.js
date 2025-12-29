// GaelFórsa Turbine Map Entry Point
import {initMap, loadTurbines} from './map/index.js';
import './styles/main.css';

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    const map = initMap();
    loadTurbines(map).catch((error) => {
        console.error('Failed to load turbines:', error);
    });
});
