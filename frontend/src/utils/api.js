import { get, patch } from './apiClient';

function handleResponse(response) {
    if (Array.isArray(response)) {
        return response;
    }
    return response.results || response;
}

export async function fetchTurbines() {
    const data = await get('/turbines');
    return handleResponse(data);
}

export async function fetchTurbine(turbineId) {
    return await get(`/turbines/${turbineId}`);
}

export async function fetchTurbineLogs(turbineId, limit = 100) {
    const data = await get(`/turbines/${turbineId}/logs`, { limit });
    return data.rows || [];
}

export async function fetchFarmStats() {
    try {
        const turbines = await fetchTurbines();

        const totalTurbines = turbines.length;
        const operationalTurbines = turbines.filter(t => t.status === 'operational' || t.status === 'normal').length;
        const avgHealthScore = turbines.length > 0
            ? turbines.reduce((sum, t) => sum + (t.health_score || 0), 0) / turbines.length
            : 0;

        return {
            total_turbines: totalTurbines,
            operational_turbines: operationalTurbines,
            average_health_score: Math.round(avgHealthScore * 100) / 100,
            turbines_by_status: turbines.reduce((acc, t) => {
                const status = t.status || 'unknown';
                acc[status] = (acc[status] || 0) + 1;
                return acc;
            }, {})
        };
    } catch (error) {
        console.error('Failed to fetch farm stats:', error);
        return null;
    }
}

export async function fetchFarmPowerSummary(days = 7) {
    // Power summary would require additional backend endpoint
    // For now, return placeholder structure
    try {
        const turbines = await fetchTurbines();
        return {
            days,
            by_turbine: turbines.map(t => ({
                turbine_id: t.id,
                name: t.name,
                status: t.status,
                health_score: t.health_score
            }))
        };
    } catch (error) {
        console.error('Failed to fetch power summary:', error);
        return null;
    }
}

export async function fetchFarmHealthTrends(days = 30) {
    // Health trends would require additional backend endpoint
    // For now, return placeholder structure
    try {
        const turbines = await fetchTurbines();
        const avgHealth = turbines.length > 0
            ? turbines.reduce((sum, t) => sum + (t.health_score || 0), 0) / turbines.length
            : 0;

        return {
            days,
            trends: [{
                date: new Date().toISOString().split('T')[0],
                average_health: Math.round(avgHealth * 100) / 100,
                turbine_count: turbines.length
            }]
        };
    } catch (error) {
        console.error('Failed to fetch health trends:', error);
        return null;
    }
}

export async function fetchFarmGridImpact() {
    // Grid impact would require additional backend endpoint
    // For now, compute from available turbine data
    try {
        const turbines = await fetchTurbines();
        const total = turbines.length;
        const operational = turbines.filter(t =>
            t.status === 'operational' || t.status === 'normal' || t.status === 'green'
        ).length;
        const critical = turbines.filter(t =>
            t.status === 'critical' || t.status === 'red'
        ).length;

        // Estimated power values (placeholder calculations)
        const activePowerPerTurbine = 2500; // kW per operational turbine
        const totalActivePower = operational * activePowerPerTurbine;
        const reactivePower = totalActivePower * 0.3; // Typical power factor estimate

        return {
            system_availability: {
                total_turbines: total,
                operational_turbines: operational,
                percentage: total > 0 ? (operational / total) * 100 : 0
            },
            power_metrics: {
                total_active_power_kw: totalActivePower,
                estimated_reactive_power_kvar: reactivePower
            },
            grid_stability: {
                critical_alerts_count: critical
            }
        };
    } catch (error) {
        console.error('Failed to fetch grid impact:', error);
        return null;
    }
}

export async function fetchFarmEconomics() {
    // Economics would require additional backend endpoint
    // For now, compute from available turbine data
    try {
        const turbines = await fetchTurbines();
        const total = turbines.length;
        const operational = turbines.filter(t =>
            t.status === 'operational' || t.status === 'normal' || t.status === 'green'
        ).length;
        const maintenance = turbines.filter(t => t.status === 'maintenance').length;
        const downtime = total - operational;

        // Estimated values (placeholder calculations)
        const periodDays = 30;
        const revenuePerTurbinePerDay = 500; // EUR
        const maintenanceCostPerEvent = 2000; // EUR
        const downtimeCostPerTurbinePerDay = 300; // EUR

        const totalRevenue = operational * revenuePerTurbinePerDay * periodDays;
        const maintenanceCost = maintenance * maintenanceCostPerEvent;
        const downtimeCost = downtime * downtimeCostPerTurbinePerDay * periodDays;
        const totalCost = maintenanceCost + downtimeCost;
        const preventiveSavings = maintenance * 500; // Savings from preventive maintenance
        const netBenefit = totalRevenue - totalCost + preventiveSavings;
        const roi = totalCost > 0 ? ((totalRevenue - totalCost) / totalCost) * 100 : 0;

        return {
            revenue: {
                total_revenue_eur: totalRevenue,
                period_days: periodDays
            },
            costs: {
                maintenance_cost_eur: maintenanceCost,
                downtime_cost_eur: downtimeCost,
                total_cost_eur: totalCost
            },
            savings: {
                estimated_preventive_savings_eur: preventiveSavings,
                net_benefit_eur: netBenefit
            },
            metrics: {
                roi_percentage: roi
            }
        };
    } catch (error) {
        console.error('Failed to fetch economics:', error);
        return null;
    }
}

// Turbine alerts - returns alerts for a specific turbine(?)
// Backend endpoint: GET /turbines/{turbineId}/alerts (not yet implemented)
export async function fetchTurbineAlerts(turbineId) {
    try {
        const data = await get(`/turbines/${turbineId}/alerts`);
        return handleResponse(data);
    } catch (error) {
        // Endpoint may not exist yet - return empty array
        console.warn('fetchTurbineAlerts: endpoint not available, returning empty array');
        return [];
    }
}

// Turbine health predictions - returns ML predictions for a specific turbine(?)
// Backend endpoint: GET /turbines/{turbineId}/predictions (not yet implemented)
export async function fetchTurbineHealthPredictions(turbineId, limit = 10) {
    try {
        const data = await get(`/turbines/${turbineId}/predictions`, { limit });
        return handleResponse(data);
    } catch (error) {
        // Endpoint may not exist yet - return empty array
        console.warn('fetchTurbineHealthPredictions: endpoint not available, returning empty array');
        return [];
    }
}

// Turbine maintenance events - returns maintenance history for a specific turbine(?)
// Backend endpoint: GET /turbines/{turbineId}/maintenance (not yet implemented)
export async function fetchTurbineMaintenanceEvents(turbineId, limit = 10) {
    try {
        const data = await get(`/turbines/${turbineId}/maintenance`, { limit });
        return handleResponse(data);
    } catch (error) {
        // Endpoint may not exist yet - return empty array
        console.warn('fetchTurbineMaintenanceEvents: endpoint not available, returning empty array');
        return [];
    }
}

// Acknowledge an alert for a specific turbine(?)
// Backend endpoint: PATCH /turbines/{turbineId}/alerts/{alertId}/acknowledge (not yet implemented)
export async function acknowledgeAlert(turbineId, alertId) {
    try {
        return await patch(`/turbines/${turbineId}/alerts/${alertId}/acknowledge`, {});
    } catch (error) {
        console.warn('acknowledgeAlert: endpoint not available');
        throw error;
    }
}

