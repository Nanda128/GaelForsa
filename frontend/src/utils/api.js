import { get, post, patch } from './apiClient';

function handleResponse(response) {
    if (Array.isArray(response)) {
        return response;
    }
    return response.results || response;
}

export async function fetchTurbines() {
    const data = await get('/turbines/');
    return handleResponse(data);
}

export async function fetchTurbine(turbineId) {
    return await get(`/turbines/${turbineId}/`);
}

export async function fetchTurbineAlerts(turbineId) {
    const data = await get(`/turbines/${turbineId}/alerts/`);
    return handleResponse(data);
}

export async function fetchTurbineHealthPredictions(turbineId, limit = 10) {
    const data = await get(`/turbines/${turbineId}/health-predictions/`, { limit });
    return handleResponse(data);
}

export async function fetchTurbineMaintenanceEvents(turbineId, limit = 10) {
    const data = await get(`/turbines/${turbineId}/maintenance-events/`, { limit });
    return handleResponse(data);
}

export async function fetchTurbineLogs(turbineId, limit = 100) {
    const data = await get(`/turbines/${turbineId}/logs/`, { limit });
    return handleResponse(data);
}

export async function acknowledgeAlert(turbineId, alertId) {
    return await patch(`/turbines/${turbineId}/alerts/${alertId}/`, {
        acknowledged: true,
        acknowledged_at: new Date().toISOString()
    });
}

export async function fetchFarmStats() {
    return await get('/farm/stats/');
}

export async function fetchFarmPowerSummary(days = 7) {
    return await get('/farm/power-summary/', { days });
}

export async function fetchFarmHealthTrends(days = 30) {
    return await get('/farm/health-trends/', { days });
}

export async function fetchFarmGridImpact() {
    return await get('/farm/grid-impact/');
}

export async function fetchFarmEconomics() {
    return await get('/farm/economics/');
}
