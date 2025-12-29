// GaelFórsa Turbine Map Entry Point
import {initMap, loadTurbines, filterTurbines, focusOnTurbine, initSidebar, updateSidebarTurbines} from './map/index.js';
import './styles/main.css';

document.addEventListener('DOMContentLoaded', async () => {
    const map = initMap();

    initSidebar({
        onTurbineClick: (turbine) => {
            focusOnTurbine(map, turbine);
        },
        onFilterChange: (statusFilters) => {
            filterTurbines(map, statusFilters);
        }
    });

    try {
        const turbines = await loadTurbines(map);
        updateSidebarTurbines(turbines);
    } catch (error) {
        console.error('Failed to load turbines:', error);
    }
});
